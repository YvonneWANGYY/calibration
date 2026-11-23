#!/usr/bin/env python3
"""Train Conditional RQ-Spline Coupling Flow (NSF) and sample theta combinations."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch
from kfold_split import member_splits, validate_configuration
from torch import save as save_checkpoint

from crqsf_core import ConditionalRQSplineFlow, FlowConfig, parse_hidden_dims
from crqsf_data import (
    DEFAULT_METRIC_KEYS,
    DEFAULT_THETA_KEYS,
    ProfileMinMaxScaler,
    ProfileStandardizer,
    MaskedContextStandardizer,
    Standardizer,
    ThetaCoordinateTransform,
    build_masked_tensors,
    build_profile_tensors,
    build_tensors,
    compute_metric_distance_weights,
    compute_theta_neighbor_weights,
    build_prior_over_proposal_sample_weights,
    make_loader,
    parse_csv_keys,
    prepare_profile_context,
    read_metrics_jsonl,
    split_dataset_indices,
)
from crqsf_profile_encoder import ProfileConditionedFlow, ProfileEncoder, ProfileEncoderConfig


DEFAULT_TARGET_RUN_DIR = ""

PROFILE_FIELD_NAMES = ["density", "pressure", "velocity", "specific_internal_energy"]
THETA_INTERNAL_MIN = 1.0e-8


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            return batch[0], None, batch[1], None
        if len(batch) == 3:
            second, third = batch[1], batch[2]
            if getattr(second, "ndim", 0) == 2 and second.shape[-1] == 1:
                return batch[0], batch[1], batch[2], None
            return batch[0], None, batch[1], batch[2]
        if len(batch) == 4:
            return batch[0], batch[1], batch[2], batch[3]
    raise ValueError("Unexpected batch structure")


def model_log_prob(model, theta: torch.Tensor, context: torch.Tensor, summary_features: torch.Tensor | None = None) -> torch.Tensor:
    if summary_features is None:
        return model.log_prob(theta, context)
    return model.log_prob(theta, context, summary_features)


def ensemble_mixture_log_prob(member_log_probs: torch.Tensor) -> torch.Tensor:
    if member_log_probs.ndim != 2:
        raise ValueError("member_log_probs must have shape (n_members, n_samples)")
    if member_log_probs.shape[0] < 1:
        raise ValueError("member_log_probs must contain at least one ensemble member")
    return torch.logsumexp(member_log_probs, dim=0) - math.log(float(member_log_probs.shape[0]))


def parse_ensemble_seeds(seed_text: str, base_seed: int, ensemble_size: int) -> List[int]:
    if int(ensemble_size) <= 0:
        raise ValueError("--ensemble_size must be > 0")
    if not seed_text:
        return [int(base_seed) + offset for offset in range(int(ensemble_size))]
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if len(seeds) != int(ensemble_size):
        raise ValueError("--ensemble_seeds length must match --ensemble_size")
    return seeds


def make_round_robin_ensemble_sample_fn(member_sample_fns: List[Callable[[int], torch.Tensor]]) -> Callable[[int], torch.Tensor]:
    if not member_sample_fns:
        raise ValueError("member_sample_fns must contain at least one sampler")

    def sample(n: int) -> torch.Tensor:
        n = int(n)
        if n <= 0:
            raise ValueError("sample count must be positive")
        assignments = [idx % len(member_sample_fns) for idx in range(n)]
        chunks: Dict[int, torch.Tensor] = {}
        for member_idx in sorted(set(assignments)):
            chunks[member_idx] = member_sample_fns[member_idx](assignments.count(member_idx))
        counters = {member_idx: 0 for member_idx in chunks}
        rows = []
        for member_idx in assignments:
            sample_idx = counters[member_idx]
            rows.append(chunks[member_idx][sample_idx])
            counters[member_idx] += 1
        return torch.stack(rows, dim=0)

    return sample


def weighted_nll_from_batch(
    model,
    theta: torch.Tensor,
    context: torch.Tensor,
    summary_features: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_prob = model_log_prob(model, theta, context, summary_features)
    if not torch.isfinite(log_prob).all():
        raise FloatingPointError("Non-finite log_prob encountered during training")
    per_sample_nll = -log_prob
    if sample_weights is None:
        loss = per_sample_nll.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite NLL encountered during training")
        return loss
    weights = sample_weights / sample_weights.sum().clamp_min(1.0e-8)
    loss = (per_sample_nll * weights).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite weighted NLL encountered during training")
    return loss


def _ensure_finite_gradients(model) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"Non-finite gradient encountered in parameter '{name}' during training")


def _ensure_finite_parameters(model) -> None:
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"Non-finite parameter encountered in '{name}' during training")


def train_one_epoch(model, loader, optimizer: torch.optim.Optimizer, device: torch.device, grad_clip: float) -> float:
    model.train()
    running = 0.0
    count = 0.0
    for batch_idx, batch in enumerate(loader):
        x_ctx, x_summary, x_theta, sample_weights = unpack_batch(batch)
        x_ctx = x_ctx.to(device)
        x_theta = x_theta.to(device)
        if x_summary is not None:
            x_summary = x_summary.to(device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        try:
            loss = weighted_nll_from_batch(
                model,
                x_theta,
                x_ctx,
                summary_features=x_summary,
                sample_weights=sample_weights,
            )
        except FloatingPointError as exc:
            raise FloatingPointError(f"{exc} (train batch_idx={batch_idx})") from exc
        loss.backward()
        _ensure_finite_gradients(model)
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        _ensure_finite_parameters(model)
        denom = float(sample_weights.sum().item()) if sample_weights is not None else float(x_ctx.shape[0])
        running += float(loss.item()) * denom
        count += denom
    return running / max(1.0, count)


@torch.no_grad()
def eval_nll(model, loader, device: torch.device) -> float:
    model.eval()
    running = 0.0
    count = 0.0
    for batch_idx, batch in enumerate(loader):
        x_ctx, x_summary, x_theta, sample_weights = unpack_batch(batch)
        x_ctx = x_ctx.to(device)
        x_theta = x_theta.to(device)
        if x_summary is not None:
            x_summary = x_summary.to(device)
        if sample_weights is not None:
            sample_weights = sample_weights.to(device)
        try:
            loss = weighted_nll_from_batch(
                model,
                x_theta,
                x_ctx,
                summary_features=x_summary,
                sample_weights=sample_weights,
            )
        except FloatingPointError as exc:
            raise FloatingPointError(f"{exc} (eval batch_idx={batch_idx})") from exc
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite evaluation NLL encountered during training")
        denom = float(sample_weights.sum().item()) if sample_weights is not None else float(x_ctx.shape[0])
        running += float(loss.item()) * denom
        count += denom
    return running / max(1.0, count)


def train_model_to_best(
    model,
    train_loader,
    val_loader,
    test_loader,
    *,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    epochs: int,
    early_stopping: bool,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
    device: torch.device,
    member_label: str = "",
) -> Tuple[List[Dict[str, float]], float | None]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stopped_early = False
    epochs_without_improvement = 0
    prefix = f"{member_label} " if member_label else ""
    for epoch in range(1, epochs + 1):
        train_nll = train_one_epoch(model, train_loader, optimizer, device=device, grad_clip=grad_clip)
        val_nll = eval_nll(model, val_loader, device=device)
        history.append({"epoch": epoch, "train_nll": train_nll, "val_nll": val_nll})
        if val_nll < (best_val - early_stopping_min_delta):
            best_val = val_nll
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(f"{prefix}Epoch {epoch:4d}/{epochs} | train_nll={train_nll:.6f} | val_nll={val_nll:.6f}")
        if early_stopping and epochs_without_improvement >= early_stopping_patience:
            stopped_early = True
            print(
                f"{prefix}Early stopping at epoch {epoch} "
                f"(best_epoch={best_epoch}, best_val_nll={best_val:.6f}, "
                f"patience={early_stopping_patience}, min_delta={early_stopping_min_delta})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_nll = eval_nll(model, test_loader, device=device) if test_loader is not None else None

    if history:
        history[-1]["best_epoch"] = int(best_epoch)
        history[-1]["best_val_nll"] = float(best_val)
        history[-1]["stopped_early"] = bool(stopped_early)
        history[-1]["early_stopping_patience"] = int(early_stopping_patience)
        history[-1]["early_stopping_min_delta"] = float(early_stopping_min_delta)
        history[-1]["test_nll"] = None if test_nll is None else float(test_nll)

    return history, test_nll


def parse_target_metrics(args, metric_keys: List[str], device: torch.device) -> torch.Tensor:
    if args.target_metrics:
        vals = [float(v.strip()) for v in args.target_metrics.split(",") if v.strip()]
        if len(vals) != len(metric_keys):
            raise ValueError(f"--target_metrics expects {len(metric_keys)} values, got {len(vals)}")
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)

    if args.target_metrics_json:
        payload = json.loads(Path(args.target_metrics_json).read_text())
        metrics_payload = payload.get("metrics", payload)
        if not isinstance(metrics_payload, dict):
            raise ValueError(f"--target_metrics_json must contain a dict or a 'metrics' dict: {args.target_metrics_json}")
        vals = [float(metrics_payload.get(k, 0.0)) for k in metric_keys]
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)

    if args.target_run_dir:
        target_run_dir = args.target_run_dir
        if not Path(target_run_dir).exists() and "referenceeSim" in target_run_dir:
            alt = target_run_dir.replace("referenceeSim", "referenceSim")
            if Path(alt).exists():
                target_run_dir = alt

        target_run_path = Path(target_run_dir)
        metrics_json_path = target_run_path / "metrics.json"
        if metrics_json_path.exists():
            payload = json.loads(metrics_json_path.read_text())
            metrics = payload.get("metrics", {})
            if any(k not in metrics for k in metric_keys):
                from profile_analysis import compute_metrics_from_run
                metrics, _ = compute_metrics_from_run(target_run_dir)
        else:
            from profile_analysis import compute_metrics_from_run
            metrics, _ = compute_metrics_from_run(target_run_dir)
        vals = [float(metrics.get(k, 0.0)) for k in metric_keys]
        print(f"Using target metrics derived from run_dir={target_run_dir}")
        return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)

    return torch.zeros(1, len(metric_keys), dtype=torch.float32, device=device)


def resolve_metrics_file(metrics_file_arg: str) -> str:
    path = Path(metrics_file_arg)
    if path.exists():
        return str(path)
    return metrics_file_arg


def build_scalar_target_weights(
    x_raw: torch.Tensor,
    target_metrics_raw: torch.Tensor,
    ctx_std: Standardizer,
    strength: float,
    floor: float,
    metric_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if metric_mask is None:
        return compute_metric_distance_weights(
            metrics=x_raw,
            target_metrics=target_metrics_raw,
            scale=ctx_std.std.detach().cpu(),
            strength=strength,
            floor=floor,
        )
    if x_raw.shape != metric_mask.shape:
        raise ValueError("metric_mask must match x_raw")
    delta2 = ((x_raw - target_metrics_raw) / ctx_std.std.detach().cpu()).pow(2)
    observed = metric_mask.to(dtype=delta2.dtype)
    distance = torch.sqrt((delta2 * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1.0))
    weights = torch.exp(-float(strength) * distance)
    return weights.clamp(min=float(floor), max=1.0)


def build_profile_target_weights(
    profiles_raw: torch.Tensor,
    profile_std: ProfileStandardizer,
    strength: float,
    floor: float,
) -> torch.Tensor:
    context_raw = profiles_raw.reshape(profiles_raw.shape[0], -1)
    target_raw = torch.zeros(1, context_raw.shape[1], dtype=torch.float32)
    scale = profile_std.std.reshape(-1)
    return compute_metric_distance_weights(
        metrics=context_raw,
        target_metrics=target_raw,
        scale=scale,
        strength=strength,
        floor=floor,
    )


def resample_marginal_columns(samples: torch.Tensor) -> torch.Tensor:
    """Preserve each sampled 1D marginal while breaking row-wise correlations."""
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D tensor")
    if samples.shape[0] <= 1 or samples.shape[1] <= 1:
        return samples.clone()
    recombined = samples.clone()
    for col in range(samples.shape[1]):
        recombined[:, col] = samples[torch.randperm(samples.shape[0], device=samples.device), col]
    return recombined


def theta_log_bounds(theta_min: float, theta_max: float, theta_internal_min: float) -> Tuple[float, float]:
    lower = max(float(theta_min), float(theta_internal_min))
    upper = float(theta_max)
    if lower <= 0.0:
        lower = float(theta_internal_min)
    if upper <= lower:
        raise ValueError("theta_max must be larger than the positive theta lower bound")
    return math.log10(lower), math.log10(upper)


def transform_theta_log_to_std(theta_log: torch.Tensor, theta_std: Standardizer) -> torch.Tensor:
    mean = theta_std.mean.to(device=theta_log.device, dtype=theta_log.dtype)
    std = theta_std.std.to(device=theta_log.device, dtype=theta_log.dtype)
    return (theta_log - mean) / std


def inverse_transform_std_to_theta_log(samples_std: torch.Tensor, theta_std: Standardizer) -> torch.Tensor:
    mean = theta_std.mean.to(device=samples_std.device, dtype=samples_std.dtype)
    std = theta_std.std.to(device=samples_std.device, dtype=samples_std.dtype)
    return samples_std * std + mean


def clamp_samples_to_theta_bounds(
    samples_std: torch.Tensor,
    theta_std: Standardizer,
    theta_min: float,
    theta_max: float,
    theta_internal_min: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_min, log_max = theta_log_bounds(theta_min, theta_max, theta_internal_min)
    theta_log = inverse_transform_std_to_theta_log(samples_std, theta_std).clamp(min=log_min, max=log_max)
    clamped_samples_std = transform_theta_log_to_std(theta_log, theta_std)
    theta = torch.pow(10.0, theta_log).clamp(min=max(theta_min, theta_internal_min), max=theta_max)
    return theta, theta_log, clamped_samples_std


def mle_upper_bounds(reference: torch.Tensor, log_max: float) -> torch.Tensor:
    """Optional per-coordinate bounds for the bounded MAP optimizer."""
    upper = torch.full_like(reference, log_max)
    overrides = json.loads(os.environ.get("CRQSF_MLE_UPPER_BOUNDS", "{}"))
    for index, value in overrides.items():
        upper[..., int(index)] = min(log_max, math.log10(float(value)))
    return upper


def optimize_mle_starts(
    initial_samples_std: torch.Tensor,
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    theta_std: Standardizer,
    theta_min: float,
    theta_max: float,
    theta_internal_min: float,
    mle_steps: int,
    mle_lr: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if initial_samples_std.ndim != 2:
        raise ValueError("initial_samples_std must be a 2D tensor")
    if initial_samples_std.shape[0] == 0:
        raise ValueError("Need at least one MLE start")
    log_min, log_max = theta_log_bounds(theta_min, theta_max, theta_internal_min)
    mean = theta_std.mean.to(device=initial_samples_std.device, dtype=initial_samples_std.dtype)
    std = theta_std.std.to(device=initial_samples_std.device, dtype=initial_samples_std.dtype)
    upper = mle_upper_bounds(initial_samples_std, log_max)
    start_theta_log = torch.minimum((initial_samples_std * std + mean).clamp(min=log_min), upper)
    width = upper - log_min
    if torch.any(width <= 0):
        raise ValueError("MAP upper bounds must exceed the lower bound")
    start_fraction = ((start_theta_log - log_min) / width).clamp(1.0e-6, 1.0 - 1.0e-6)
    raw = torch.logit(start_fraction).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=float(mle_lr))

    for _ in range(int(mle_steps)):
        optimizer.zero_grad(set_to_none=True)
        theta_log = log_min + width * torch.sigmoid(raw)
        samples_std = (theta_log - mean) / std
        log_probs = log_prob_fn(samples_std)
        loss = -log_probs.mean()
        if not torch.isfinite(loss):
            break
        (grad,) = torch.autograd.grad(loss, raw, retain_graph=False, create_graph=False)
        raw.grad = grad
        optimizer.step()

    theta_log = log_min + width * torch.sigmoid(raw.detach())
    samples_std = (theta_log - mean) / std
    log_probs = log_prob_fn(samples_std).detach()
    return samples_std.detach(), theta_log.detach(), log_probs


def metropolis_sample_pool(
    sample_fn: Callable[[int], torch.Tensor],
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    score_fn: Callable[[torch.Tensor], torch.Tensor],
    theta_std: Standardizer,
    theta_min: float,
    theta_max: float,
    theta_internal_min: float,
    sample_count: int,
    proposal_scale: float,
    proposal_sampling_mode: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    chain_count = min(max(16, min(int(sample_count), 64)), int(sample_count))
    with torch.no_grad():
        states_std = sample_fn(chain_count)
        if proposal_sampling_mode == "marginal":
            states_std = resample_marginal_columns(states_std)
        _, _, states_std = clamp_samples_to_theta_bounds(
            states_std,
            theta_std=theta_std,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_internal_min=theta_internal_min,
        )
        state_log_probs = log_prob_fn(states_std).detach()

    collected: List[torch.Tensor] = []
    collected_log_probs: List[torch.Tensor] = []
    collected_scores: List[torch.Tensor] = []
    accepted = 0
    proposed = 0
    burn_in_steps = max(8, chain_count // 4)
    post_burn_steps = math.ceil(int(sample_count) / chain_count)
    total_steps = burn_in_steps + post_burn_steps
    scale = float(proposal_scale)

    for step in range(total_steps):
        with torch.no_grad():
            proposed_std = states_std + torch.randn_like(states_std) * scale
            _, _, proposed_std = clamp_samples_to_theta_bounds(
                proposed_std,
                theta_std=theta_std,
                theta_min=theta_min,
                theta_max=theta_max,
                theta_internal_min=theta_internal_min,
            )
            proposed_log_probs = log_prob_fn(proposed_std).detach()
            log_accept_ratio = proposed_log_probs - state_log_probs
            accept = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio
            states_std = torch.where(accept.unsqueeze(1), proposed_std, states_std)
            state_log_probs = torch.where(accept, proposed_log_probs, state_log_probs)
            accepted += int(accept.sum().item())
            proposed += int(accept.numel())
            if step >= burn_in_steps:
                collected.append(states_std.detach().cpu())
                collected_log_probs.append(state_log_probs.detach().cpu())
                collected_scores.append(score_fn(states_std).detach().cpu())

    samples_std = torch.cat(collected, dim=0)[: int(sample_count)]
    log_probs = torch.cat(collected_log_probs, dim=0)[: int(sample_count)]
    scores = torch.cat(collected_scores, dim=0)[: int(sample_count)]
    _, theta_log, samples_std = clamp_samples_to_theta_bounds(
        samples_std,
        theta_std=theta_std,
        theta_min=theta_min,
        theta_max=theta_max,
        theta_internal_min=theta_internal_min,
    )
    acceptance_rate = float(accepted / proposed) if proposed else 0.0
    return samples_std.cpu(), theta_log.cpu(), log_probs.cpu(), scores.cpu(), acceptance_rate


def ensemble_policy_scores(member_logs, lambda_t, normalization, reference_stats=None,
                           density_statistic="mean_member_log_probability"):
    """Score an ensemble density plus disagreement over the full sampled pool."""
    if member_logs.ndim != 2 or member_logs.shape[1] == 0 or not torch.isfinite(member_logs).all():
        raise ValueError("member log probabilities must be a finite nonempty matrix")
    if normalization not in {"none", "pool_zscore"} or not math.isfinite(lambda_t):
        raise ValueError("invalid ensemble acquisition configuration")
    if density_statistic not in {"mean_member_log_probability", "mixture_log_probability"}:
        raise ValueError("invalid ensemble density statistic")
    means = (member_logs.mean(0) if density_statistic == "mean_member_log_probability"
             else ensemble_mixture_log_prob(member_logs))
    stds = member_logs.std(0, unbiased=False)
    stats = reference_stats if reference_stats is not None else {
        "ensemble_pool_mean_center": float(means.mean()),
        "ensemble_pool_mean_scale": float(means.std(unbiased=False)),
        "ensemble_pool_std_center": float(stds.mean()),
        "ensemble_pool_std_scale": float(stds.std(unbiased=False)),
    }
    if normalization == "pool_zscore":
        def standardize(values, kind):
            scale = stats[f"ensemble_pool_{kind}_scale"]
            center = stats[f"ensemble_pool_{kind}_center"]
            # Do not amplify float32 cancellation noise in an effectively constant term.
            tolerance = 8 * torch.finfo(values.dtype).eps * max(1.0, abs(center))
            return ((values - center) / scale
                    if scale > tolerance else torch.zeros_like(values))
        means = standardize(means, "mean")
        stds = standardize(stds, "std")
    return means - lambda_t * stds, stats


def sample_topk_candidates(
    sample_fn: Callable[[int], torch.Tensor],
    log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
    historical_theta_log: torch.Tensor,
    historical_context_weights: torch.Tensor,
    theta_std: Standardizer,
    theta_keys: List[str],
    theta_min: float,
    theta_max: float,
    theta_internal_min: float,
    iter_idx: int,
    num_samples: int,
    candidate_pool_size: int,
    sampling_temperature: float,
    weighted_sampling: bool,
    sampling_weight_floor: float,
    sampling_theta_bandwidth: float,
    min_existing_log_distance: float,
    proposal_sampling_mode: str,
    proposal_selection_mode: str = "sample_topk",
    density_statistic: str = "mean_member_log_probability",
    mle_starts: int = 64,
    mle_steps: int = 200,
    mle_lr: float = 0.03,
    mcmc_proposal_scale: float = 0.18,
    uniform_mixture_epsilon: float = 0.0,
    min_existing_normalized_distance: float = 0.0,
    score_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    theta_transform: ThetaCoordinateTransform | None = None,
    policy_member_log_probs_fn=None,
    acquisition_lambda: float = 1.0,
    acquisition_normalization: str = "none",
    allow_short_candidate_list: bool = False,
    independent_prior_fraction: float = 0.0,
    independent_prior_seed: int = 41,
) -> List[Dict[str, float]]:
    if proposal_sampling_mode not in {"joint", "marginal"}:
        raise ValueError("proposal_sampling_mode must be 'joint' or 'marginal'")
    if proposal_selection_mode not in {
        "sample_topk",
        "ensemble_rerank",
        "mle",
        "mcmc",
        "direct_sample",
        "tempered_uniform_mixture",
    }:
        raise ValueError(
            "proposal_selection_mode must be 'sample_topk', 'ensemble_rerank', 'mle', 'mcmc', "
            "'direct_sample', or 'tempered_uniform_mixture'"
        )
    if not 0.0 <= float(uniform_mixture_epsilon) <= 1.0:
        raise ValueError("uniform_mixture_epsilon must be between 0 and 1")
    if min_existing_normalized_distance < 0.0:
        raise ValueError("min_existing_normalized_distance must be >= 0")
    if not math.isfinite(independent_prior_fraction) or not 0<=independent_prior_fraction<=1:raise ValueError('Invalid independent prior fraction')
    if independent_prior_fraction and proposal_selection_mode not in {'sample_topk','ensemble_rerank'}:raise ValueError('Independent candidates require pool ranking')
    prior_generator=torch.Generator(device='cpu').manual_seed(int(independent_prior_seed)+1000003*int(iter_idx))
    pool_size = max(int(candidate_pool_size), int(num_samples))
    if score_fn is None:
        score_fn = log_prob_fn
    hist_theta = (
        theta_transform.inverse_transform(historical_theta_log)
        if theta_transform is not None
        else torch.pow(10.0, historical_theta_log).clamp(min=max(theta_min, theta_internal_min), max=theta_max)
    )
    denom = max(float(theta_max) - float(theta_min), 1.0e-12)

    def normalized_existing_distance(theta_row: torch.Tensor) -> torch.Tensor:
        if theta_transform is not None:
            hist_norm = theta_transform.normalized(hist_theta)
            row_norm = theta_transform.normalized(theta_row.reshape(1, -1))
        else:
            hist_norm = (hist_theta - float(theta_min)) / denom
            row_norm = (theta_row.reshape(1, -1) - float(theta_min)) / denom
        return torch.linalg.norm(hist_norm - row_norm, dim=1).min()

    if proposal_selection_mode in {"direct_sample", "tempered_uniform_mixture"}:
        direct_count = int(num_samples)
        out_rows: List[Dict[str, float]] = []
        rounds_used = 0
        max_rounds = 64 if min_existing_normalized_distance > 0.0 or min_existing_log_distance > 0.0 else 1
        while len(out_rows) < direct_count and rounds_used < max_rounds:
            rounds_used += 1
            with torch.no_grad():
                samples_std = sample_fn(direct_count - len(out_rows))
                mixture_components = ["flow"] * samples_std.shape[0]
                if proposal_selection_mode == "tempered_uniform_mixture":
                    uniform_mask = torch.rand(samples_std.shape[0], device=samples_std.device) < float(uniform_mixture_epsilon)
                    uniform_count = int(uniform_mask.sum().item())
                    if uniform_count:
                        theta_uniform = torch.empty(
                            (uniform_count, len(theta_keys)),
                            dtype=torch.float32,
                            device=samples_std.device,
                        ).uniform_(float(theta_min), float(theta_max))
                        theta_log_uniform = torch.log10(theta_uniform.clamp_min(theta_internal_min)).cpu()
                        samples_std[uniform_mask] = theta_std.transform(theta_log_uniform).to(samples_std.device)
                        for idx in torch.nonzero(uniform_mask, as_tuple=False).flatten().tolist():
                            mixture_components[int(idx)] = "uniform"
                elif proposal_sampling_mode == "marginal":
                    samples_std = resample_marginal_columns(samples_std)

                theta, theta_log, clamped_samples_std = clamp_samples_to_theta_bounds(
                    samples_std,
                    theta_std=theta_std,
                    theta_min=theta_min,
                    theta_max=theta_max,
                    theta_internal_min=theta_internal_min,
                )
                log_probs = log_prob_fn(clamped_samples_std).detach().cpu()
                score_values = score_fn(clamped_samples_std).detach().cpu()
                theta_log = theta_log.cpu()
                theta = theta.cpu()

            for sample_index in range(theta.shape[0]):
                theta_row = theta[sample_index]
                theta_log_row = theta_log[sample_index]
                if min_existing_log_distance > 0.0:
                    min_dist = torch.linalg.norm(historical_theta_log - theta_log_row.reshape(1, -1), dim=1).min()
                    if float(min_dist.item()) < float(min_existing_log_distance):
                        continue
                else:
                    min_dist = torch.tensor(float("nan"))
                if min_existing_normalized_distance > 0.0:
                    normalized_dist = normalized_existing_distance(theta_row)
                    if float(normalized_dist.item()) < float(min_existing_normalized_distance):
                        continue
                else:
                    normalized_dist = torch.tensor(float("nan"))
                row: Dict[str, float] = {}
                for j, key in enumerate(theta_keys):
                    row[key] = float(theta_row[j].item())
                row["log_prob"] = float(log_probs[sample_index].item())
                row["sampling_weight"] = 1.0
                row["final_score"] = float(score_values[sample_index].item())
                row["weighted_sampling"] = bool(weighted_sampling)
                row["proposal_sampling_mode"] = proposal_sampling_mode
                row["proposal_selection_mode"] = proposal_selection_mode
                row["candidate_pool_size"] = direct_count * rounds_used
                row["sampling_temperature"] = float(sampling_temperature)
                row["sampling_theta_bandwidth"] = float(sampling_theta_bandwidth)
                row["min_existing_log_distance"] = None if not torch.isfinite(min_dist) else float(min_dist.item())
                row["min_existing_normalized_distance"] = (
                    None if not torch.isfinite(normalized_dist) else float(normalized_dist.item())
                )
                row["uniform_mixture_epsilon"] = float(uniform_mixture_epsilon)
                row["mixture_component"] = mixture_components[sample_index]
                row["sampling_rounds"] = rounds_used
                row["run_id"] = f"crqsf_iter{iter_idx}_{len(out_rows):03d}"
                row["rank"] = len(out_rows) + 1
                out_rows.append(row)
                if len(out_rows) >= direct_count:
                    break
        if len(out_rows) < direct_count:
            raise RuntimeError(
                "Could not find enough direct proposal candidates after normalized existing-theta distance filtering. "
                "Decrease --min_existing_normalized_distance or rerun with a new seed."
            )
        return out_rows

    if proposal_selection_mode == "mcmc":
        samples_std, theta_log, log_probs, score_values, acceptance_rate = metropolis_sample_pool(
            sample_fn=sample_fn,
            log_prob_fn=log_prob_fn,
            score_fn=score_fn,
            theta_std=theta_std,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_internal_min=theta_internal_min,
            sample_count=pool_size,
            proposal_scale=mcmc_proposal_scale,
            proposal_sampling_mode=proposal_sampling_mode,
        )
        theta = torch.pow(10.0, theta_log).clamp(min=max(theta_min, theta_internal_min), max=theta_max)
        if weighted_sampling:
            neighbor_weights = compute_theta_neighbor_weights(
                candidate_theta_log=theta_log,
                historical_theta_log=historical_theta_log,
                historical_metric_weights=historical_context_weights,
                bandwidth=sampling_theta_bandwidth,
                floor=sampling_weight_floor,
            )
            final_scores = score_values + torch.log(neighbor_weights.clamp_min(1.0e-8))
        else:
            neighbor_weights = torch.ones(pool_size, dtype=torch.float32)
            final_scores = score_values

        sorted_idx = torch.argsort(final_scores, descending=True)
        unique_rows: Dict[tuple[float, ...], Dict[str, float]] = {}
        fallback_rows: List[Dict[str, float]] = []
        for sample_index in sorted_idx:
            theta_row = theta[sample_index]
            theta_log_row = theta_log[sample_index]
            log_prob = log_probs[sample_index]
            acquisition_score = score_values[sample_index]
            neighbor_weight = neighbor_weights[sample_index]
            final_score = final_scores[sample_index]
            if min_existing_log_distance > 0.0:
                min_dist = torch.linalg.norm(historical_theta_log - theta_log_row.reshape(1, -1), dim=1).min()
                if float(min_dist.item()) < float(min_existing_log_distance):
                    continue
            else:
                min_dist = torch.tensor(float("nan"))
            if min_existing_normalized_distance > 0.0:
                normalized_dist = normalized_existing_distance(theta_row)
                if float(normalized_dist.item()) < float(min_existing_normalized_distance):
                    continue
            else:
                normalized_dist = torch.tensor(float("nan"))
            row: Dict[str, float] = {}
            theta_vals = []
            for j, key in enumerate(theta_keys):
                value = float(theta_row[j].item())
                row[key] = value
                theta_vals.append(round(value, 8))
            row["log_prob"] = float(log_prob.item())
            row["acquisition_score"] = float(acquisition_score.item())
            row["sampling_weight"] = float(neighbor_weight.item())
            row["final_score"] = float(final_score.item())
            row["weighted_sampling"] = bool(weighted_sampling)
            row["proposal_sampling_mode"] = proposal_sampling_mode
            row["proposal_selection_mode"] = proposal_selection_mode
            row["candidate_pool_size"] = pool_size
            row["sampling_temperature"] = float(sampling_temperature)
            row["sampling_theta_bandwidth"] = float(sampling_theta_bandwidth)
            row["min_existing_log_distance"] = None if not torch.isfinite(min_dist) else float(min_dist.item())
            row["min_existing_normalized_distance"] = None
            row["mcmc_sample_index"] = int(sample_index.item())
            row["mcmc_sample_count"] = int(pool_size)
            row["mcmc_acceptance_rate"] = float(acceptance_rate)
            row["mcmc_proposal_scale"] = float(mcmc_proposal_scale)
            fallback_rows.append(dict(row))
            theta_signature = tuple(theta_vals)
            prev = unique_rows.get(theta_signature)
            if prev is None or row["final_score"] > prev["final_score"]:
                unique_rows[theta_signature] = row

        out_rows = sorted(unique_rows.values(), key=lambda row: row["final_score"], reverse=True)
        if len(out_rows) < int(num_samples):
            for row in sorted(fallback_rows, key=lambda row: row["final_score"], reverse=True):
                out_rows.append(row)
                if len(out_rows) >= int(num_samples):
                    break
        if len(out_rows) < int(num_samples):
            raise RuntimeError(
                "Could not find enough MCMC proposal candidates after existing-theta distance filtering. "
                "Decrease --min_existing_log_distance or increase --candidate_pool_size."
            )

        final_rows: List[Dict[str, float]] = []
        for rank, row in enumerate(out_rows[: int(num_samples)], start=1):
            row = dict(row)
            row["sampling_rounds"] = 1
            row["run_id"] = f"crqsf_iter{iter_idx}_{rank - 1:03d}"
            row["rank"] = rank
            final_rows.append(row)
        return final_rows

    if proposal_selection_mode == "mle":
        start_pool_size = max(pool_size, int(mle_starts), int(num_samples))
        with torch.no_grad():
            prescan_samples_std = sample_fn(start_pool_size)
            if proposal_sampling_mode == "marginal":
                prescan_samples_std = resample_marginal_columns(prescan_samples_std)
            _, prescan_theta_log, prescan_samples_std = clamp_samples_to_theta_bounds(
                prescan_samples_std,
                theta_std=theta_std,
                theta_min=theta_min,
                theta_max=theta_max,
                theta_internal_min=theta_internal_min,
            )
            prescan_log_probs = log_prob_fn(prescan_samples_std).detach().cpu()
            prescan_scores = score_fn(prescan_samples_std).detach().cpu()
            prescan_theta_log_cpu = prescan_theta_log.detach().cpu()
            if weighted_sampling:
                prescan_neighbor_weights = compute_theta_neighbor_weights(
                    candidate_theta_log=prescan_theta_log_cpu,
                    historical_theta_log=historical_theta_log,
                    historical_metric_weights=historical_context_weights,
                    bandwidth=sampling_theta_bandwidth,
                    floor=sampling_weight_floor,
                )
                prescan_final_scores = prescan_scores + torch.log(prescan_neighbor_weights.clamp_min(1.0e-8))
            else:
                prescan_neighbor_weights = torch.ones(start_pool_size, dtype=torch.float32)
                prescan_final_scores = prescan_scores
            start_count = min(start_pool_size, max(int(mle_starts), int(num_samples)))
            start_idx = torch.argsort(prescan_final_scores, descending=True)[:start_count]
            mle_initial_samples_std = prescan_samples_std[start_idx].detach()
            mle_start_scores = prescan_final_scores[start_idx].detach().cpu()

        mle_samples_std, mle_theta_log, mle_scores = optimize_mle_starts(
            initial_samples_std=mle_initial_samples_std,
            log_prob_fn=score_fn,
            theta_std=theta_std,
            theta_min=theta_min,
            theta_max=theta_max,
            theta_internal_min=theta_internal_min,
            mle_steps=mle_steps,
            mle_lr=mle_lr,
        )
        mle_theta_log_cpu = mle_theta_log.cpu()
        mle_theta = torch.pow(10.0, mle_theta_log_cpu).clamp(min=max(theta_min, theta_internal_min), max=theta_max)
        mle_log_probs_cpu = log_prob_fn(mle_samples_std).detach().cpu()
        mle_scores_cpu = mle_scores.cpu()
        if weighted_sampling:
            mle_neighbor_weights = compute_theta_neighbor_weights(
                candidate_theta_log=mle_theta_log_cpu,
                historical_theta_log=historical_theta_log,
                historical_metric_weights=historical_context_weights,
                bandwidth=sampling_theta_bandwidth,
                floor=sampling_weight_floor,
            )
            mle_final_scores = mle_scores_cpu + torch.log(mle_neighbor_weights.clamp_min(1.0e-8))
        else:
            mle_neighbor_weights = torch.ones(start_count, dtype=torch.float32)
            mle_final_scores = mle_scores_cpu

        sorted_idx = torch.argsort(mle_final_scores, descending=True)
        unique_rows: Dict[tuple[float, ...], Dict[str, float]] = {}
        fallback_rows: List[Dict[str, float]] = []
        for start_pos in sorted_idx:
            theta_row = mle_theta[start_pos]
            theta_log_row = mle_theta_log_cpu[start_pos]
            log_prob = mle_log_probs_cpu[start_pos]
            acquisition_score = mle_scores_cpu[start_pos]
            neighbor_weight = mle_neighbor_weights[start_pos]
            final_score = mle_final_scores[start_pos]
            if min_existing_log_distance > 0.0:
                min_dist = torch.linalg.norm(historical_theta_log - theta_log_row.reshape(1, -1), dim=1).min()
                if float(min_dist.item()) < float(min_existing_log_distance):
                    continue
            else:
                min_dist = torch.tensor(float("nan"))
            if min_existing_normalized_distance > 0.0:
                normalized_dist = normalized_existing_distance(theta_row)
                if float(normalized_dist.item()) < float(min_existing_normalized_distance):
                    continue
            else:
                normalized_dist = torch.tensor(float("nan"))
            row: Dict[str, float] = {}
            theta_vals = []
            for j, key in enumerate(theta_keys):
                value = float(theta_row[j].item())
                row[key] = value
                theta_vals.append(round(value, 8))
            row["log_prob"] = float(log_prob.item())
            row["acquisition_score"] = float(acquisition_score.item())
            row["sampling_weight"] = float(neighbor_weight.item())
            row["final_score"] = float(final_score.item())
            row["weighted_sampling"] = bool(weighted_sampling)
            row["proposal_sampling_mode"] = proposal_sampling_mode
            row["proposal_selection_mode"] = proposal_selection_mode
            row["candidate_pool_size"] = start_pool_size
            row["sampling_temperature"] = float(sampling_temperature)
            row["sampling_theta_bandwidth"] = float(sampling_theta_bandwidth)
            row["min_existing_log_distance"] = None if not torch.isfinite(min_dist) else float(min_dist.item())
            row["mle_start_rank"] = int(start_pos.item()) + 1
            row["mle_start_score"] = float(mle_start_scores[start_pos].item())
            row["mle_start_count"] = int(start_count)
            row["mle_steps"] = int(mle_steps)
            row["mle_lr"] = float(mle_lr)
            fallback_rows.append(dict(row))
            theta_signature = tuple(theta_vals)
            prev = unique_rows.get(theta_signature)
            if prev is None or row["final_score"] > prev["final_score"]:
                unique_rows[theta_signature] = row

        out_rows = sorted(unique_rows.values(), key=lambda row: row["final_score"], reverse=True)
        if len(out_rows) < int(num_samples):
            for row in sorted(fallback_rows, key=lambda row: row["final_score"], reverse=True):
                out_rows.append(row)
                if len(out_rows) >= int(num_samples):
                    break
        if len(out_rows) < int(num_samples):
            raise RuntimeError(
                "Could not find enough MLE proposal candidates after existing-theta distance filtering. "
                "Decrease --min_existing_log_distance, increase --mle_starts, or increase --candidate_pool_size."
            )

        final_rows: List[Dict[str, float]] = []
        for rank, row in enumerate(out_rows[: int(num_samples)], start=1):
            row = dict(row)
            row["sampling_rounds"] = 1
            row["run_id"] = f"crqsf_iter{iter_idx}_{rank - 1:03d}"
            row["rank"] = rank
            final_rows.append(row)
        return final_rows

    policy_pool_logs = []
    max_rounds = 4
    rounds_used = 0
    unique_rows: Dict[tuple[float, ...], Dict[str, float]] = {}
    fallback_rows: List[Dict[str, float]] = []

    for _ in range(max_rounds):
        rounds_used += 1
        with torch.no_grad():
            samples_std = sample_fn(pool_size)
            if proposal_sampling_mode == "marginal":
                samples_std = resample_marginal_columns(samples_std)
            from independent_exploration import replace_with_independent_prior
            samples_std,prior_mask=replace_with_independent_prior(samples_std,theta_std,theta_transform,independent_prior_fraction,prior_generator)
            theta_coordinate = theta_std.inverse_transform(samples_std.cpu())
            if theta_transform is not None:
                theta = theta_transform.inverse_transform(theta_coordinate)
                clamped_theta_log = theta_transform.transform(theta)
            else:
                theta = torch.pow(10.0, theta_coordinate).clamp(max=theta_max)
                if theta_min > 0.0:
                    theta = theta.clamp(min=theta_min)
                else:
                    theta = torch.where(theta < theta_internal_min, torch.zeros_like(theta), theta)
                clamped_theta_log = torch.log10(theta.clamp_min(theta_internal_min))
            clamped_samples_std = theta_std.transform(clamped_theta_log).to(samples_std.device)
            clamped_log_probs = log_prob_fn(clamped_samples_std).detach().cpu()
            clamped_scores = score_fn(clamped_samples_std).detach().cpu()
            if policy_member_log_probs_fn is not None:
                policy_pool_logs.append(policy_member_log_probs_fn(clamped_samples_std).detach().cpu())
        if weighted_sampling:
            neighbor_weights = compute_theta_neighbor_weights(
                candidate_theta_log=clamped_theta_log,
                historical_theta_log=historical_theta_log,
                historical_metric_weights=historical_context_weights,
                bandwidth=sampling_theta_bandwidth,
                floor=sampling_weight_floor,
            )
            final_scores = clamped_scores.cpu() + torch.log(neighbor_weights.clamp_min(1.0e-8))
        else:
            neighbor_weights = torch.ones(pool_size, dtype=torch.float32)
            final_scores = clamped_scores.cpu()

        sorted_idx = torch.argsort(final_scores, descending=True)
        sorted_theta = theta[sorted_idx]
        sorted_log_probs = clamped_log_probs[sorted_idx].cpu()
        sorted_scores = clamped_scores[sorted_idx].cpu()
        sorted_neighbor_weights = neighbor_weights[sorted_idx].cpu()
        sorted_final_scores = final_scores[sorted_idx].cpu()
        sorted_theta_log = clamped_theta_log[sorted_idx].cpu()

        for theta_row, theta_log_row, log_prob, acquisition_score, neighbor_weight, final_score, from_prior in zip(
            sorted_theta,
            sorted_theta_log,
            sorted_log_probs,
            sorted_scores,
            sorted_neighbor_weights,
            sorted_final_scores,
            prior_mask[sorted_idx].cpu(),
        ):
            if min_existing_log_distance > 0.0:
                min_dist = torch.linalg.norm(historical_theta_log - theta_log_row.reshape(1, -1), dim=1).min()
                if float(min_dist.item()) < float(min_existing_log_distance):
                    continue
            else:
                min_dist = torch.tensor(float("nan"))
            if min_existing_normalized_distance > 0.0:
                normalized_dist = normalized_existing_distance(theta_row)
                if float(normalized_dist.item()) < float(min_existing_normalized_distance):
                    continue
            else:
                normalized_dist = torch.tensor(float("nan"))
            row: Dict[str, float] = {}
            theta_vals = []
            for j, key in enumerate(theta_keys):
                value = float(theta_row[j].item())
                row[key] = value
                theta_vals.append(round(value, 8))
            row["candidate_origin"] = "independent_prior" if bool(from_prior) else "flow"
            row["independent_prior_fraction"] = float(independent_prior_fraction)
            row["log_prob"] = float(log_prob.item())
            row["acquisition_score"] = float(acquisition_score.item())
            row["sampling_weight"] = float(neighbor_weight.item())
            row["final_score"] = float(final_score.item())
            row["weighted_sampling"] = bool(weighted_sampling)
            row["proposal_sampling_mode"] = proposal_sampling_mode
            row["proposal_selection_mode"] = proposal_selection_mode
            row["candidate_pool_size"] = pool_size * rounds_used
            row["sampling_temperature"] = float(sampling_temperature)
            row["sampling_theta_bandwidth"] = float(sampling_theta_bandwidth)
            row["min_existing_log_distance"] = None if not torch.isfinite(min_dist) else float(min_dist.item())
            row["min_existing_normalized_distance"] = (
                None if not torch.isfinite(normalized_dist) else float(normalized_dist.item())
            )
            fallback_rows.append(dict(row))
            theta_signature = tuple(theta_vals)
            prev = unique_rows.get(theta_signature)
            if prev is None or row["final_score"] > prev["final_score"]:
                unique_rows[theta_signature] = row

        if len(unique_rows) >= int(num_samples):
            break

    if policy_member_log_probs_fn is not None:
        _, stats = ensemble_policy_scores(torch.cat(policy_pool_logs, dim=1),
                                          acquisition_lambda, acquisition_normalization,
                                          density_statistic=density_statistic)
        # Normalize once over every sampled round, before filtering/truncation.
        rows = list(unique_rows.values()) + fallback_rows
        if rows:
            raw = torch.tensor([[row[key] for key in theta_keys] for row in rows], dtype=torch.float32)
            coordinates = (theta_transform.transform(raw) if theta_transform is not None
                           else torch.log10(raw.clamp_min(theta_internal_min)))
            with torch.no_grad():
                logs = policy_member_log_probs_fn(theta_std.transform(coordinates).to(samples_std.device)).detach().cpu()
            scores, _ = ensemble_policy_scores(logs, acquisition_lambda, acquisition_normalization,
                                               stats, density_statistic=density_statistic)
            for index, row in enumerate(rows):
                row.update(stats)
                row["ensemble_acquisition_lambda"] = acquisition_lambda
                row["ensemble_acquisition_normalization"] = acquisition_normalization
                row["ensemble_acquisition_score"] = float(scores[index])
                row["acquisition_score"] = float(scores[index])
                row["final_score"] = float(scores[index]) + (math.log(max(row["sampling_weight"], 1.e-8)) if weighted_sampling else 0.)

    out_rows = sorted(unique_rows.values(), key=lambda row: row["final_score"], reverse=True)
    if len(out_rows) < int(num_samples):
        for row in sorted(fallback_rows, key=lambda row: row["final_score"], reverse=True):
            out_rows.append(row)
            if len(out_rows) >= int(num_samples):
                break
    if len(out_rows) < int(num_samples) and (not allow_short_candidate_list or not out_rows):
        raise RuntimeError(
            "Could not find enough proposal candidates after existing-theta distance filtering. "
            "Decrease --min_existing_log_distance or increase --candidate_pool_size."
        )
    if len(out_rows) < int(num_samples):
        print(f"Returning {len(out_rows)} eligible proposal rows (requested up to {num_samples}); "
              f"distance threshold unchanged, {pool_size * rounds_used} total sampled candidates.", flush=True)

    final_rows: List[Dict[str, float]] = []
    for row in out_rows[: int(num_samples)]:
        row = dict(row)
        row["candidate_pool_size"] = pool_size * rounds_used
        row["candidate_pool_prior_draws"] = round(pool_size*independent_prior_fraction)*rounds_used
        row["candidate_pool_flow_draws"] = (pool_size-round(pool_size*independent_prior_fraction))*rounds_used
        row["sampling_rounds"] = rounds_used
        row["run_id"] = f"crqsf_iter{iter_idx}_{len(final_rows):03d}"
        row["rank"] = len(final_rows) + 1
        final_rows.append(row)
    return final_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Conditional NSF p(theta | context)")
    parser.add_argument("--metrics_file", type=str, default="metrics.jsonl")
    parser.add_argument("--iter", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--hidden_dims", type=str, default="256,256")
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--tail_bound", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--ensemble_size", "--ensemble-size", type=int, default=1)
    parser.add_argument("--ensemble_seeds", "--ensemble-seeds", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument("--metric_keys", type=str, default=",".join(DEFAULT_METRIC_KEYS))
    parser.add_argument("--theta_keys", type=str, default=",".join(DEFAULT_THETA_KEYS))
    parser.add_argument("--theta-transform-json", type=str, default="")
    parser.add_argument("--theta_min", type=float, default=0.0)
    parser.add_argument("--theta_max", type=float, default=1000.0)
    parser.add_argument("--context_mode", type=str, default="profile", choices=["scalar", "profile"])
    parser.add_argument("--profile_n_bins", type=int, default=768)
    parser.add_argument("--profile_scaling_mode", type=str, default="standardize", choices=["standardize", "raw", "robust_minmax"])
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--encoder_channels", type=int, default=64)
    parser.add_argument("--encoder_layers", type=int, default=4)
    parser.add_argument("--encoder_kernel_size", type=int, default=5)
    parser.add_argument("--weighted_sampling", dest="weighted_sampling", action="store_true")
    parser.add_argument("--no_weighted_sampling", dest="weighted_sampling", action="store_false")
    parser.add_argument("--weighted_nll", dest="weighted_sampling", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no_weighted_nll", dest="weighted_sampling", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--sampling_weight_strength", type=float, default=0.5)
    parser.add_argument("--sampling_weight_floor", type=float, default=0.25)
    parser.add_argument("--sampling_theta_bandwidth", type=float, default=0.35)
    parser.add_argument(
        "--proposal_sampling_mode",
        "--proposal-sampling-mode",
        choices=["joint", "marginal"],
        default="joint",
    )
    parser.add_argument(
        "--proposal_selection_mode",
        "--proposal-selection-mode",
        choices=["sample_topk", "ensemble_rerank", "mle", "mcmc", "direct_sample", "tempered_uniform_mixture"],
        default="sample_topk",
    )
    parser.add_argument("--candidate_pool_size", type=int, default=256)
    parser.add_argument("--independent_prior_fraction",type=float,default=0.)
    parser.add_argument("--independent_prior_seed",type=int,default=41)
    parser.add_argument("--allow_short_candidate_list", action="store_true",
                        help="Return available ranked candidates after the existing retry limit; still fail if none survive.")
    parser.add_argument("--sampling_temperature", type=float, default=0.7)
    parser.add_argument("--mle_starts", "--mle-starts", type=int, default=64)
    parser.add_argument("--mle_steps", "--mle-steps", type=int, default=200)
    parser.add_argument("--mle_lr", "--mle-lr", type=float, default=0.03)
    parser.add_argument("--mcmc_proposal_scale", "--mcmc-proposal-scale", type=float, default=0.18)
    parser.add_argument("--uniform_mixture_epsilon", "--uniform-mixture-epsilon", type=float, default=0.0)
    parser.add_argument("--ensemble_disagreement_penalty", "--ensemble-disagreement-penalty", type=float, default=0.0)
    parser.add_argument("--ensemble_acquisition_policy", choices=["legacy", "current", "posterior", "explore", "explore_exploit", "log_explore_exploit", "mixture_log_explore_exploit"], default="legacy")
    parser.add_argument("--ensemble_density_statistic", choices=["mean_member_log_probability", "mixture_log_probability"], default="mean_member_log_probability")
    parser.add_argument("--ensemble_acquisition_lambda", type=float, default=None)
    parser.add_argument("--ensemble_acquisition_normalization", choices=["none", "pool_zscore"], default="pool_zscore")
    parser.add_argument("--min_existing_log_distance", type=float, default=0.0)
    parser.add_argument("--min_existing_normalized_distance", "--min-existing-normalized-distance", type=float, default=0.0)
    parser.add_argument(
        "--proposal_correction_mode",
        "--proposal-correction-mode",
        choices=["none", "prior_over_proposal"],
        default="none",
    )
    parser.add_argument("--proposal_correction_min_weight", "--proposal-correction-min-weight", type=float, default=0.05)
    parser.add_argument("--proposal_correction_max_weight", "--proposal-correction-max-weight", type=float, default=20.0)
    parser.add_argument("--target_metrics", type=str, default="")
    parser.add_argument("--target_metrics_json", type=str, default="")
    parser.add_argument("--target_run_dir", type=str, default=DEFAULT_TARGET_RUN_DIR)
    parser.add_argument("--source_filter", type=str, default="all")
    parser.add_argument("--history_file", type=str, default="")
    parser.add_argument("--model_out", type=str, default="crqsf_model.pth")
    parser.add_argument("--theta_out", type=str, default="theta_next.json")
    parser.add_argument("--early_stopping", dest="early_stopping", action="store_true")
    parser.add_argument("--no_early_stopping", dest="early_stopping", action="store_false")
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1.0e-4)
    parser.set_defaults(weighted_sampling=True)
    parser.set_defaults(early_stopping=True)
    args = parser.parse_args()

    if args.theta_min < 0.0:
        raise ValueError("--theta_min must be >= 0")
    if args.theta_max < args.theta_min:
        raise ValueError("--theta_max must be >= --theta_min")
    if args.sampling_weight_strength < 0.0:
        raise ValueError("--sampling_weight_strength must be >= 0")
    if not (0.0 < args.sampling_weight_floor <= 1.0):
        raise ValueError("--sampling_weight_floor must be in (0, 1]")
    if args.sampling_theta_bandwidth <= 0.0:
        raise ValueError("--sampling_theta_bandwidth must be > 0")
    if args.candidate_pool_size <= 0:
        raise ValueError("--candidate_pool_size must be > 0")
    if args.sampling_temperature <= 0.0:
        raise ValueError("--sampling_temperature must be > 0")
    if args.mle_starts <= 0:
        raise ValueError("--mle_starts must be > 0")
    if args.mle_steps < 0:
        raise ValueError("--mle_steps must be >= 0")
    if args.mle_lr <= 0.0:
        raise ValueError("--mle_lr must be > 0")
    if args.mcmc_proposal_scale <= 0.0:
        raise ValueError("--mcmc_proposal_scale must be > 0")
    if not 0.0 <= args.uniform_mixture_epsilon <= 1.0:
        raise ValueError("--uniform_mixture_epsilon must be between 0 and 1")
    fixed_lambdas = {"current": 1.0, "posterior": 0.0, "explore": -1.0}
    if args.ensemble_acquisition_policy in fixed_lambdas:
        args.ensemble_acquisition_lambda = fixed_lambdas[args.ensemble_acquisition_policy]
    elif args.ensemble_acquisition_policy == "legacy":
        args.ensemble_acquisition_lambda = 1.0
    elif args.ensemble_acquisition_lambda is None:
        raise ValueError("scheduled policies require the scheduled --ensemble_acquisition_lambda")
    if args.ensemble_acquisition_policy != "legacy":
        if args.proposal_selection_mode != "ensemble_rerank":
            raise ValueError("ensemble acquisition policies require ensemble_rerank")
        if not math.isfinite(args.ensemble_acquisition_lambda):
            raise ValueError("ensemble acquisition lambda must be finite")
    if args.ensemble_disagreement_penalty < 0.0:
        raise ValueError("--ensemble_disagreement_penalty must be >= 0")
    if args.min_existing_log_distance < 0.0:
        raise ValueError("--min_existing_log_distance must be >= 0")
    if args.min_existing_normalized_distance < 0.0:
        raise ValueError("--min_existing_normalized_distance must be >= 0")
    if args.proposal_correction_min_weight <= 0.0:
        raise ValueError("--proposal-correction-min-weight must be > 0")
    if args.proposal_correction_max_weight < args.proposal_correction_min_weight:
        raise ValueError("--proposal-correction-max-weight must be >= --proposal-correction-min-weight")
    if args.embed_dim <= 0:
        raise ValueError("--embed_dim must be > 0")
    if args.early_stopping_patience <= 0:
        raise ValueError("--early_stopping_patience must be > 0")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("--early_stopping_min_delta must be >= 0")
    if args.ensemble_size <= 0:
        raise ValueError("--ensemble_size must be > 0")
    ensemble_seeds = parse_ensemble_seeds(args.ensemble_seeds, base_seed=args.seed, ensemble_size=args.ensemble_size)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_keys = parse_csv_keys(args.metric_keys, DEFAULT_METRIC_KEYS)
    import os
    weighted_condition_config=json.loads(Path(os.environ['SAGE_TARGET_CONDITION_CONFIG']).read_text())
    if weighted_condition_config['metric_keys'] != metric_keys:
        raise ValueError('Target condition metric ordering mismatch')
    theta_keys = parse_csv_keys(args.theta_keys, DEFAULT_THETA_KEYS)
    theta_transform = ThetaCoordinateTransform.from_spec(
        theta_keys,
        json.loads(args.theta_transform_json) if args.theta_transform_json else None,
    )
    metrics_file = resolve_metrics_file(args.metrics_file)
    if metrics_file != args.metrics_file:
        print(f"Resolved metrics file: {metrics_file}")
    records = read_metrics_jsonl(metrics_file, source_filter=args.source_filter)
    _, _, theta_log_raw = build_masked_tensors(
        records, metric_keys=metric_keys, theta_keys=theta_keys, theta_transform=theta_transform
    )
    if theta_log_raw.shape[0] < 2:
        raise ValueError("Need at least 2 samples in metrics.jsonl")

    n = theta_log_raw.shape[0]
    validate_configuration(n, ensemble_seeds, args.seed, args.context_mode, args.test_fraction)
    train_idx, val_idx, test_idx = split_dataset_indices(
        n,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    theta_train_raw = theta_log_raw[train_idx]
    theta_val_raw = theta_log_raw[val_idx]
    theta_test_raw = theta_log_raw[test_idx] if test_idx.numel() > 0 else None
    proposal_sample_weights = None
    if args.proposal_correction_mode == "prior_over_proposal":
        proposal_sample_weights = build_prior_over_proposal_sample_weights(
            records,
            min_weight=float(args.proposal_correction_min_weight),
            max_weight=float(args.proposal_correction_max_weight),
        )
    # Use all observed theta values for label scaling. With two bootstrap samples,
    # the train split can contain a single row; fitting the theta scaler on one row
    # collapses std to eps and makes proposal sampling reproduce that seed theta.
    from prior_scaling import fixed_theta_standardizer
    theta_std = fixed_theta_standardizer(theta_transform)
    y_train = theta_std.transform(theta_train_raw)
    y_val = theta_std.transform(theta_val_raw)
    y_test = theta_std.transform(theta_test_raw) if theta_test_raw is not None else None

    if args.context_mode == "scalar":
        x_raw, x_mask, _ = build_masked_tensors(
            records, metric_keys=metric_keys, theta_keys=theta_keys, theta_transform=theta_transform
        )
        x_train_raw = x_raw[train_idx]
        x_val_raw = x_raw[val_idx]
        x_test_raw = x_raw[test_idx] if test_idx.numel() > 0 else None
        x_train_mask = x_mask[train_idx]
        x_val_mask = x_mask[val_idx]
        x_test_mask = x_mask[test_idx] if test_idx.numel() > 0 else None
        ctx_std = MaskedContextStandardizer.from_data(x_train_raw, x_train_mask)
        x_train = ctx_std.transform(x_train_raw, x_train_mask)
        x_val = ctx_std.transform(x_val_raw, x_val_mask)
        x_test = ctx_std.transform(x_test_raw, x_test_mask) if x_test_raw is not None else None
        target_metrics_raw = parse_target_metrics(args, metric_keys=metric_keys, device=torch.device("cpu"))
        target_context_std = ctx_std.transform_target(target_metrics_raw).to(device)
        historical_context_weights = build_scalar_target_weights(
            x_raw=x_raw,
            target_metrics_raw=target_metrics_raw,
            ctx_std=ctx_std,
            strength=args.sampling_weight_strength,
            floor=args.sampling_weight_floor,
            metric_mask=x_mask,
        )
        train_weights = proposal_sample_weights[train_idx] if proposal_sample_weights is not None else None
        val_weights = proposal_sample_weights[val_idx] if proposal_sample_weights is not None else None
        test_weights = (
            proposal_sample_weights[test_idx]
            if proposal_sample_weights is not None and test_idx.numel() > 0
            else None
        )
        train_loader = make_loader(x_train, y_train, batch_size=args.batch_size, sample_weights=train_weights, shuffle=True)
        val_loader = make_loader(x_val, y_val, batch_size=args.batch_size, sample_weights=val_weights, shuffle=False)
        test_loader = (
            make_loader(x_test, y_test, batch_size=args.batch_size, sample_weights=test_weights, shuffle=False)
            if x_test is not None and y_test is not None
            else None
        )
        flow_cfg = FlowConfig(
            features=len(theta_keys),
            context_features=2 * len(metric_keys),
            num_layers=args.num_layers,
            num_bins=args.num_bins,
            hidden_dims=parse_hidden_dims(args.hidden_dims),
            tail_bound=args.tail_bound,
            dropout=args.dropout,
        )
        def make_member_model():
            model = ConditionalRQSplineFlow(flow_cfg).to(device)

            def sample_fn_local(n: int, member_model=model) -> torch.Tensor:
                return member_model.sample(n, target_context_std, temperature=args.sampling_temperature)

            def log_prob_fn_local(samples_std: torch.Tensor, member_model=model) -> torch.Tensor:
                return member_model.log_prob(samples_std, target_context_std.repeat(samples_std.shape[0], 1))

            return model, sample_fn_local, log_prob_fn_local

        checkpoint_extra = {
            "context_mode": "scalar",
            "context_standardizer": ctx_std.state_dict(),
            "metric_keys": metric_keys,
            "partial_metric_context": "masked",
        }
    else:
        profiles_raw, _ = build_profile_tensors(records, field_names=PROFILE_FIELD_NAMES, summary_key=None)
        if profiles_raw.shape[2] != int(args.profile_n_bins):
            raise ValueError(f"Expected profile context with {int(args.profile_n_bins)} bins, got {profiles_raw.shape[2]}")
        profiles_train_raw = profiles_raw[train_idx]
        profiles_val_raw = profiles_raw[val_idx]
        profiles_test_raw = profiles_raw[test_idx] if test_idx.numel() > 0 else None
        profile_std = ProfileStandardizer.from_data(profiles_train_raw)
        profile_minmax = ProfileMinMaxScaler.from_data(profiles_train_raw)
        profiles_train = prepare_profile_context(
            profiles_train_raw,
            profile_std=profile_std,
            profile_minmax=profile_minmax,
            scaling_mode=args.profile_scaling_mode,
            clip_value=10.0,
        )
        profiles_val = prepare_profile_context(
            profiles_val_raw,
            profile_std=profile_std,
            profile_minmax=profile_minmax,
            scaling_mode=args.profile_scaling_mode,
            clip_value=10.0,
        )
        profiles_test = (
            prepare_profile_context(
                profiles_test_raw,
                profile_std=profile_std,
                profile_minmax=profile_minmax,
                scaling_mode=args.profile_scaling_mode,
                clip_value=10.0,
            )
            if profiles_test_raw is not None
            else None
        )
        target_profiles_std = prepare_profile_context(
            torch.zeros(1, len(PROFILE_FIELD_NAMES), args.profile_n_bins, dtype=torch.float32),
            profile_std=profile_std,
            profile_minmax=profile_minmax,
            scaling_mode=args.profile_scaling_mode,
            clip_value=10.0,
        ).to(device)
        historical_context_weights = build_profile_target_weights(
            profiles_raw=profiles_raw,
            profile_std=profile_std,
            strength=args.sampling_weight_strength,
            floor=args.sampling_weight_floor,
        )
        train_weights = proposal_sample_weights[train_idx] if proposal_sample_weights is not None else None
        val_weights = proposal_sample_weights[val_idx] if proposal_sample_weights is not None else None
        test_weights = (
            proposal_sample_weights[test_idx]
            if proposal_sample_weights is not None and test_idx.numel() > 0
            else None
        )
        train_loader = make_loader(profiles_train, y_train, batch_size=args.batch_size, sample_weights=train_weights, shuffle=True)
        val_loader = make_loader(profiles_val, y_val, batch_size=args.batch_size, sample_weights=val_weights, shuffle=False)
        test_loader = (
            make_loader(profiles_test, y_test, batch_size=args.batch_size, sample_weights=test_weights, shuffle=False)
            if profiles_test is not None and y_test is not None
            else None
        )
        encoder_cfg = ProfileEncoderConfig(
            n_bins=args.profile_n_bins,
            n_channels=len(PROFILE_FIELD_NAMES),
            embed_dim=args.embed_dim,
            conv_channels=args.encoder_channels,
            conv_layers=args.encoder_layers,
            kernel_size=args.encoder_kernel_size,
        )
        profile_encoder = ProfileEncoder(encoder_cfg)
        flow_cfg = FlowConfig(
            features=len(theta_keys),
            context_features=args.embed_dim,
            num_layers=args.num_layers,
            num_bins=args.num_bins,
            hidden_dims=parse_hidden_dims(args.hidden_dims),
            tail_bound=args.tail_bound,
            dropout=args.dropout,
        )
        def make_member_model():
            member_encoder = ProfileEncoder(encoder_cfg)
            model = ProfileConditionedFlow(member_encoder, ConditionalRQSplineFlow(flow_cfg)).to(device)

            def sample_fn_local(n: int, member_model=model) -> torch.Tensor:
                return member_model.sample(n, target_profiles_std, temperature=args.sampling_temperature)

            def log_prob_fn_local(samples_std: torch.Tensor, member_model=model) -> torch.Tensor:
                return member_model.log_prob(samples_std, target_profiles_std.repeat(samples_std.shape[0], 1, 1))

            return model, sample_fn_local, log_prob_fn_local

        checkpoint_extra = {
            "context_mode": "profile",
            "profile_field_names": PROFILE_FIELD_NAMES,
            "profile_n_bins": int(args.profile_n_bins),
            "profile_standardizer": profile_std.state_dict() if args.profile_scaling_mode == "standardize" else None,
            "profile_minmax": profile_minmax.state_dict() if args.profile_scaling_mode == "robust_minmax" else None,
            "profile_scaling_mode": str(args.profile_scaling_mode),
            "profile_context_uses_summary": False,
            "profile_context_inputs": "profiles_diff_only",
            "profile_context_clip_value": 10.0,
            "profile_encoder_config": {
                "n_bins": int(args.profile_n_bins),
                "n_channels": len(PROFILE_FIELD_NAMES),
                "embed_dim": int(args.embed_dim),
                "conv_channels": int(args.encoder_channels),
                "conv_layers": int(args.encoder_layers),
                "kernel_size": int(args.encoder_kernel_size),
            },
        }

    print(
        "Dataset split sizes: "
        f"train={int(train_idx.numel())}, val={int(val_idx.numel())}, test={int(test_idx.numel())}"
    )

    member_models = []
    member_sample_fns = []
    member_log_prob_fns = []
    member_histories = []
    member_test_nlls = []
    folds = member_splits(n, len(ensemble_seeds), 5, args.seed)
    fold_metadata = []
    x_all = ctx_std.transform(x_raw, x_mask)
    y_all = theta_std.transform(theta_log_raw)
    import os
    parallel_cpu_results = None
    if device.type == 'cpu' and int(os.environ.get('SAGE_CPU_ENSEMBLE_WORKERS','1')) > 1:
        from cpu_ensemble import train_members
        parallel_cpu_results = train_members(args, flow_cfg, x_all, y_all, folds, ensemble_seeds, proposal_sample_weights, device)
    for member_index, member_seed in enumerate(ensemble_seeds):
        torch.manual_seed(int(member_seed))
        model, sample_fn_member, log_prob_fn_member = make_member_model()
        train_idx, val_idx = folds[member_index]
        train_loader = make_loader(x_all[train_idx], y_all[train_idx], batch_size=args.batch_size,
                                   sample_weights=proposal_sample_weights[train_idx] if proposal_sample_weights is not None else None,
                                   shuffle=True)
        val_loader = make_loader(x_all[val_idx], y_all[val_idx], batch_size=args.batch_size,
                                 sample_weights=proposal_sample_weights[val_idx] if proposal_sample_weights is not None else None,
                                 shuffle=False)
        fold_metadata.append(dict(member_index=member_index, seed=int(member_seed),
                                  fold=member_index % 5, train_indices=train_idx.tolist(),
                                  validation_indices=val_idx.tolist(), train_count=int(train_idx.numel()),
                                  validation_count=int(val_idx.numel())))
        Path(args.model_out).with_suffix('.folds.json').write_text(json.dumps(dict(
            strategy='balanced_5fold', split_seed=args.seed, record_ids=[r['run_id'] for r in records],
            n_folds=5, members=fold_metadata), indent=2))
        member_label = f"[ensemble {member_index + 1}/{len(ensemble_seeds)} seed={member_seed}]"
        if parallel_cpu_results is None:
            history, test_nll = train_model_to_best(
                model,
                train_loader,
                val_loader,
                test_loader,
                lr=args.lr,
                weight_decay=args.weight_decay,
                grad_clip=args.grad_clip,
                epochs=args.epochs,
                early_stopping=args.early_stopping,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                device=device,
                member_label=member_label if len(ensemble_seeds) > 1 else "",
            )
        else:
            fitted = parallel_cpu_results[member_index]
            model.load_state_dict({key: torch.as_tensor(value) for key,value in fitted['state'].items()})
            model.train(fitted['training'])
            history, test_nll = fitted['history'], fitted['test_nll']
            torch.set_rng_state(torch.as_tensor(fitted['rng_state'],dtype=torch.uint8))
        if history:
            history[-1]["split_sizes"] = {
                "train": int(train_idx.numel()),
                "val": int(val_idx.numel()),
                "test": int(test_idx.numel()),
            }
            history[-1]["ensemble_member_index"] = int(member_index)
            history[-1]["ensemble_member_seed"] = int(member_seed)
        member_models.append(model)
        member_sample_fns.append(sample_fn_member)
        member_log_prob_fns.append(log_prob_fn_member)
        member_histories.append(
            {
                "member_index": int(member_index),
                "cross_fitting_split": fold_metadata[-1],
                "seed": int(member_seed),
                "history": history,
                "test_nll": None if test_nll is None else float(test_nll),
            }
        )
        member_test_nlls.append(test_nll)

    model = member_models[0]
    history = member_histories[0]["history"]
    test_nll = member_test_nlls[0]
    torch.manual_seed(args.seed)

    def policy_member_log_probs_fn(samples_std):
        return torch.stack([fn(samples_std) for fn in member_log_prob_fns], dim=0)

    if len(member_log_prob_fns) == 1:
        sample_fn = member_sample_fns[0]
        log_prob_fn = member_log_prob_fns[0]
        score_fn = log_prob_fn
    else:
        sample_fn = make_round_robin_ensemble_sample_fn(member_sample_fns)

        def member_log_probs_fn(samples_std: torch.Tensor) -> torch.Tensor:
            return torch.stack([fn(samples_std) for fn in member_log_prob_fns], dim=0)

        def log_prob_fn(samples_std: torch.Tensor) -> torch.Tensor:
            return ensemble_mixture_log_prob(member_log_probs_fn(samples_std))

        def score_fn(samples_std: torch.Tensor) -> torch.Tensor:
            member_log_probs = member_log_probs_fn(samples_std)
            mixture_log_prob = ensemble_mixture_log_prob(member_log_probs)
            disagreement = torch.std(member_log_probs, dim=0, unbiased=False)
            return mixture_log_prob - float(args.ensemble_disagreement_penalty) * disagreement

    if history:
        last_history = history[-1]
        best_epoch = int(last_history.get("best_epoch", 0))
        best_val = float(last_history.get("best_val_nll", float("inf")))
        stopped_early = bool(last_history.get("stopped_early", False))
    else:
        best_epoch = 0
        best_val = float("inf")
        stopped_early = False

    split_sizes = {
        "train": int(train_idx.numel()),
        "val": int(val_idx.numel()),
        "test": int(test_idx.numel()),
    }

    checkpoint = {
        "cross_fitting": {"strategy": "balanced_5fold", "split_seed": args.seed,
                          "record_ids": [r['run_id'] for r in records], "members": fold_metadata,
                          "n_folds": 5, "theta_scaling": "fixed transformed-prior bounds; independent of simulation data",
                          "context_scaling": "original fixed observation scaling"},
        "model_state_dict": model.state_dict(),
        "ensemble_member_state_dicts": [member.state_dict() for member in member_models],
        "flow_config": {
            "features": flow_cfg.features,
            "context_features": flow_cfg.context_features,
            "num_layers": flow_cfg.num_layers,
            "num_bins": flow_cfg.num_bins,
            "hidden_dims": list(flow_cfg.hidden_dims),
            "tail_bound": flow_cfg.tail_bound,
            "dropout": flow_cfg.dropout,
        },
        "theta_keys": theta_keys,
        "theta_coordinate_transform": theta_transform.state_dict(),
        "theta_sample_bounds": {"min": args.theta_min, "max": args.theta_max},
        "theta_internal_min": float(THETA_INTERNAL_MIN),
        "training_objective": (
            "prior_over_proposal_weighted_nll"
            if args.proposal_correction_mode == "prior_over_proposal"
            else "standard_nll"
        ),
        "proposal_correction": {
            "mode": str(args.proposal_correction_mode),
            "min_weight": float(args.proposal_correction_min_weight),
            "max_weight": float(args.proposal_correction_max_weight),
            "normalized_mean_one": True,
            "weight_summary": (
                {
                    "min": float(proposal_sample_weights.min().item()),
                    "max": float(proposal_sample_weights.max().item()),
                    "mean": float(proposal_sample_weights.mean().item()),
                    "count": int(proposal_sample_weights.numel()),
                }
                if proposal_sample_weights is not None
                else None
            ),
        },
        "early_stopping": {
            "enabled": bool(args.early_stopping),
            "patience": int(args.early_stopping_patience),
            "min_delta": float(args.early_stopping_min_delta),
            "stopped_early": bool(stopped_early),
            "best_epoch": int(best_epoch),
            "best_val_nll": float(best_val),
        },
        "ensemble": {
            "size": int(len(ensemble_seeds)),
            "seeds": [int(seed) for seed in ensemble_seeds],
            "mixture_log_prob": bool(len(ensemble_seeds) > 1),
            "sample_mixture": "round_robin_uniform",
            "member_test_nlls": [None if value is None else float(value) for value in member_test_nlls],
        },
        "inference_defaults": {
            "strategy": (
                "ensemble_flow_mle"
                if len(ensemble_seeds) > 1 and args.proposal_selection_mode == "mle"
                else "flow_mle"
                if args.proposal_selection_mode == "mle"
                else "ensemble_mcmc_sample_pool_score_topk"
                if len(ensemble_seeds) > 1 and args.proposal_selection_mode == "mcmc"
                else "mcmc_sample_pool_score_topk"
                if args.proposal_selection_mode == "mcmc"
                else "ensemble_uncertainty_rerank"
                if len(ensemble_seeds) > 1 and args.proposal_selection_mode == "ensemble_rerank"
                else "uncertainty_rerank"
                if args.proposal_selection_mode == "ensemble_rerank"
                else "ensemble_sample_pool_score_topk"
                if len(ensemble_seeds) > 1
                else "sample_pool_score_topk"
            ),
            "proposal_sampling_mode": args.proposal_sampling_mode,
            "proposal_selection_mode": args.proposal_selection_mode,
            "candidate_pool_size": int(max(args.candidate_pool_size, args.num_samples)),
            "sampling_temperature": float(args.sampling_temperature),
            "mle": {
                "starts": int(args.mle_starts),
                "steps": int(args.mle_steps),
                "lr": float(args.mle_lr),
            },
            "mcmc": {
                "proposal_scale": float(args.mcmc_proposal_scale),
            },
            "independent_candidate_prior": {"fraction": args.independent_prior_fraction,"seed": args.independent_prior_seed},
            "uniform_mixture": {
                "epsilon": float(args.uniform_mixture_epsilon),
            },
            "ensemble_acquisition_policy": args.ensemble_acquisition_policy,
            "ensemble_acquisition_lambda": args.ensemble_acquisition_lambda,
            "ensemble_acquisition_normalization": args.ensemble_acquisition_normalization,
            "ensemble_disagreement_penalty": float(args.ensemble_disagreement_penalty),
            "weighted_sampling": {
                "enabled": bool(args.weighted_sampling),
                "strength": float(args.sampling_weight_strength),
                "floor": float(args.sampling_weight_floor),
                "theta_bandwidth": float(args.sampling_theta_bandwidth),
                "min_existing_log_distance": float(args.min_existing_log_distance),
                "min_existing_normalized_distance": float(args.min_existing_normalized_distance),
                "proxy": "theta_neighbor_target_closeness",
            },
        },
        "theta_standardizer": theta_std.state_dict(),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(args.test_fraction),
        "split_sizes": split_sizes,
        "test_nll": None if test_nll is None else float(test_nll),
        **checkpoint_extra,
    }
    save_checkpoint(checkpoint, args.model_out)
    print(f"Saved model checkpoint to {args.model_out}")
    if test_nll is not None:
        print(f"Best-checkpoint test NLL: {float(test_nll):.6f}")

    history_file = args.history_file or f"history_iter_{args.iter}.json"
    history_payload = (
        history
        if len(ensemble_seeds) == 1
        else {
            "ensemble_size": int(len(ensemble_seeds)),
            "ensemble_seeds": [int(seed) for seed in ensemble_seeds],
            "split_sizes": split_sizes,
            "members": member_histories,
        }
    )
    Path(history_file).write_text(json.dumps(history_payload, indent=2))
    print(f"Saved training history to {history_file}")

    out_rows = sample_topk_candidates(
        sample_fn=sample_fn,
        log_prob_fn=log_prob_fn,
        historical_theta_log=theta_log_raw,
        historical_context_weights=historical_context_weights,
        theta_std=theta_std,
        theta_keys=theta_keys,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_internal_min=THETA_INTERNAL_MIN,
        iter_idx=args.iter,
        allow_short_candidate_list=args.allow_short_candidate_list,
        independent_prior_fraction=args.independent_prior_fraction,
        independent_prior_seed=args.independent_prior_seed,
        num_samples=args.num_samples,
        candidate_pool_size=args.candidate_pool_size,
        sampling_temperature=args.sampling_temperature,
        weighted_sampling=args.weighted_sampling,
        sampling_weight_floor=args.sampling_weight_floor,
        sampling_theta_bandwidth=args.sampling_theta_bandwidth,
        min_existing_log_distance=args.min_existing_log_distance,
        proposal_sampling_mode=args.proposal_sampling_mode,
        proposal_selection_mode=args.proposal_selection_mode,
        mle_starts=args.mle_starts,
        mle_steps=args.mle_steps,
        mle_lr=args.mle_lr,
        mcmc_proposal_scale=args.mcmc_proposal_scale,
        uniform_mixture_epsilon=args.uniform_mixture_epsilon,
        min_existing_normalized_distance=args.min_existing_normalized_distance,
        score_fn=score_fn,
        theta_transform=theta_transform,
        policy_member_log_probs_fn=(policy_member_log_probs_fn if args.ensemble_acquisition_policy != "legacy" else None),
        acquisition_lambda=args.ensemble_acquisition_lambda,
        acquisition_normalization=args.ensemble_acquisition_normalization,
        density_statistic=("mixture_log_probability" if args.ensemble_acquisition_policy == "mixture_log_explore_exploit"
                           else args.ensemble_density_statistic),
    )
    for row in out_rows:
        row["ensemble_acquisition_policy"] = args.ensemble_acquisition_policy
    if len(member_log_prob_fns) > 1:
        for row in out_rows:
            theta_raw_row = torch.tensor([[float(row[key]) for key in theta_keys]], dtype=torch.float32)
            theta_log_row = theta_transform.transform(theta_transform.clamp(theta_raw_row))
            samples_std = theta_std.transform(theta_log_row).to(device)
            with torch.no_grad():
                member_log_probs = member_log_probs_fn(samples_std).detach().cpu()
            flat_member_log_probs = member_log_probs[:, 0]
            best_member_index = int(torch.argmax(flat_member_log_probs).item())
            ensemble_std = float(torch.std(flat_member_log_probs, unbiased=False).item())
            ensemble_log_prob = float(ensemble_mixture_log_prob(member_log_probs)[0].item())
            ensemble_score = ensemble_log_prob - float(args.ensemble_disagreement_penalty) * ensemble_std
            row["ensemble_size"] = int(len(member_log_prob_fns))
            row["ensemble_seeds"] = ",".join(str(seed) for seed in ensemble_seeds)
            row["ensemble_disagreement_penalty"] = float(args.ensemble_disagreement_penalty)
            row["ensemble_log_prob_mean"] = float(flat_member_log_probs.mean().item())
            row["ensemble_log_prob_std"] = ensemble_std
            row["ensemble_log_prob_min"] = float(flat_member_log_probs.min().item())
            row["ensemble_log_prob_max"] = float(flat_member_log_probs.max().item())
            if args.ensemble_acquisition_policy == "legacy":
                row["ensemble_acquisition_score"] = ensemble_score
            row["best_member_seed"] = int(ensemble_seeds[best_member_index])

    Path(args.theta_out).write_text(json.dumps(out_rows, indent=2))
    print(f"Saved {len(out_rows)} top-ranked theta combinations to {args.theta_out}")


if __name__ == "__main__":
    main()
