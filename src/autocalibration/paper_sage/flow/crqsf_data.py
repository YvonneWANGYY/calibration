#!/usr/bin/env python3
"""Data IO and preprocessing helpers for conditional RQ-spline flow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union
import math

import torch
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_METRIC_KEYS = [
    "profile_point_count",
    "profile_channel_count",
]

DEFAULT_THETA_KEYS = [
    "minslope",
    "minpressurejump",
    "minenergyratio",
]

THETA_ALIASES = {
    "minslope": ["minslope", "min_slope"],
    "minpressurejump": ["minpressurejump", "min_pres", "min_pressure_jump"],
    "minenergyratio": ["minenergyratio", "energy_ratio", "min_energy_ratio"],
}


@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def from_data(cls, x: torch.Tensor, eps: float = 1e-6) -> "Standardizer":
        mean = x.mean(dim=0)
        std = x.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor]) -> "Standardizer":
        return cls(mean=state["mean"], std=state["std"])


@dataclass(frozen=True)
class ThetaCoordinateTransform:
    """Explicit physical-to-flow transform for bounded calibration parameters."""

    keys: Tuple[str, ...]
    kinds: Tuple[str, ...]
    lower: torch.Tensor
    upper: torch.Tensor

    @classmethod
    def from_spec(cls, keys: Sequence[str], spec: Dict[str, Dict[str, Any]] | None = None) -> "ThetaCoordinateTransform":
        spec = spec or {}
        kinds: List[str] = []
        lower: List[float] = []
        upper: List[float] = []
        for key in keys:
            item = spec.get(key, {"kind": "log10", "low": 1.0e-8, "high": 1000.0})
            kind = str(item.get("kind", "log10"))
            lo, hi = float(item["low"]), float(item["high"])
            if kind not in {"linear", "log10"} or not lo < hi or (kind == "log10" and lo <= 0.0):
                raise ValueError(f"invalid theta transform for {key!r}")
            kinds.append(kind)
            lower.append(lo)
            upper.append(hi)
        return cls(tuple(keys), tuple(kinds), torch.tensor(lower), torch.tensor(upper))

    def transform(self, theta: torch.Tensor) -> torch.Tensor:
        if theta.shape[-1] != len(self.keys):
            raise ValueError("theta width does not match coordinate transform")
        out = theta.clone()
        for index, kind in enumerate(self.kinds):
            if kind == "log10":
                out[..., index] = torch.log10(out[..., index].clamp_min(float(self.lower[index])))
        return out

    def inverse_transform(self, coordinate: torch.Tensor) -> torch.Tensor:
        if coordinate.shape[-1] != len(self.keys):
            raise ValueError("coordinate width does not match coordinate transform")
        out = coordinate.clone()
        for index, kind in enumerate(self.kinds):
            if kind == "log10":
                out[..., index] = torch.pow(10.0, out[..., index])
        return self.clamp(out)

    def clamp(self, theta: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(device=theta.device, dtype=theta.dtype)
        upper = self.upper.to(device=theta.device, dtype=theta.dtype)
        return torch.maximum(torch.minimum(theta, upper), lower)

    def normalized(self, theta: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(device=theta.device, dtype=theta.dtype)
        upper = self.upper.to(device=theta.device, dtype=theta.dtype)
        return (self.clamp(theta) - lower) / (upper - lower)

    def state_dict(self) -> Dict[str, Any]:
        return {"keys": list(self.keys), "kinds": list(self.kinds), "lower": self.lower, "upper": self.upper}


@dataclass
class MaskedContextStandardizer(Standardizer):
    """Standardize observed scalar metrics and append their observation mask.

    Missing bins are filled with the standardized mean (zero) only after the
    binary mask has been appended, so zero is never interpreted as a measured
    observable value.
    """

    @classmethod
    def from_data(cls, x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> "MaskedContextStandardizer":
        if x.shape != mask.shape or x.ndim != 2:
            raise ValueError("masked scalar context and mask must both be [batch, metrics]")
        mask = mask.to(dtype=x.dtype)
        counts = mask.sum(dim=0)
        if torch.any(counts <= 0):
            missing = torch.nonzero(counts <= 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"no observed training values for metric columns {missing}")
        mean = (x * mask).sum(dim=0) / counts
        variance = (((x - mean) ** 2) * mask).sum(dim=0) / counts
        std = torch.sqrt(variance).clamp_min(eps)
        return cls(mean=mean, std=std)

    def transform(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.shape != mask.shape:
            raise ValueError("masked scalar context and mask must have identical shapes")
        normalized = (x - self.mean) / self.std
        observed = mask.to(dtype=normalized.dtype)
        return torch.cat((normalized * observed, observed), dim=-1)

    def transform_target(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x).all():
            raise ValueError("target scalar metrics must be finite")
        normalized = super().transform(x)
        return torch.cat((normalized, torch.ones_like(normalized)), dim=-1)

    def state_dict(self) -> Dict[str, torch.Tensor | str]:
        state = super().state_dict()
        state["mask_encoding"] = "standardized_observed_value_plus_binary_mask"
        return state


class ProfileStandardizer(Standardizer):
    """Per-channel, per-bin standardizer for profile tensors."""
    pass


@dataclass
class ProfileMinMaxScaler:
    lower: torch.Tensor
    upper: torch.Tensor
    feature_range: Tuple[float, float] = (-1.0, 1.0)

    @classmethod
    def from_data(
        cls,
        x: torch.Tensor,
        *,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        feature_range: Tuple[float, float] = (-1.0, 1.0),
        eps: float = 1e-6,
    ) -> "ProfileMinMaxScaler":
        if x.ndim != 3:
            raise ValueError("ProfileMinMaxScaler expects x with shape [batch, channels, bins]")
        if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
            raise ValueError("Require 0 <= lower_quantile < upper_quantile <= 1")
        lower = torch.quantile(x, lower_quantile, dim=0)
        upper = torch.quantile(x, upper_quantile, dim=0)
        upper = torch.maximum(upper, lower + eps)
        return cls(lower=lower, upper=upper, feature_range=feature_range)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.feature_range
        denom = (self.upper - self.lower).clamp_min(1e-6)
        x_clipped = torch.clamp(x, min=self.lower, max=self.upper)
        x_unit = (x_clipped - self.lower) / denom
        return x_unit * (hi - lo) + lo

    def state_dict(self) -> Dict[str, torch.Tensor | Tuple[float, float]]:
        return {"lower": self.lower, "upper": self.upper, "feature_range": self.feature_range}

    @classmethod
    def from_state_dict(cls, state: Dict[str, torch.Tensor | Tuple[float, float]]) -> "ProfileMinMaxScaler":
        feature_range = state.get("feature_range", (-1.0, 1.0))
        return cls(lower=state["lower"], upper=state["upper"], feature_range=feature_range)


def sanitize_profile_context(
    x: torch.Tensor,
    *,
    clip_value: float = 10.0,
) -> torch.Tensor:
    if clip_value <= 0.0:
        raise ValueError("clip_value must be > 0")
    x = torch.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    return x.clamp(min=-clip_value, max=clip_value)


def prepare_profile_context(
    profiles_raw: torch.Tensor,
    *,
    profile_std: ProfileStandardizer | None = None,
    profile_minmax: ProfileMinMaxScaler | None = None,
    scaling_mode: str = "standardize",
    clip_value: float = 10.0,
) -> torch.Tensor:
    if scaling_mode == "standardize":
        if profile_std is None:
            raise ValueError("profile_std is required when scaling_mode='standardize'")
        x = profile_std.transform(profiles_raw)
    elif scaling_mode == "robust_minmax":
        if profile_minmax is None:
            raise ValueError("profile_minmax is required when scaling_mode='robust_minmax'")
        x = profile_minmax.transform(profiles_raw)
    elif scaling_mode == "raw":
        x = profiles_raw.clone()
    else:
        raise ValueError(f"Unsupported scaling_mode: {scaling_mode}")
    return sanitize_profile_context(x, clip_value=clip_value)


def _extract_theta_value(theta_map: Dict[str, float], canonical_key: str) -> float:
    aliases = THETA_ALIASES.get(canonical_key, [canonical_key])
    for k in aliases:
        if k in theta_map:
            return float(theta_map[k])
    raise KeyError(f"Missing theta key '{canonical_key}' (aliases: {aliases})")


def read_metrics_jsonl(path: Union[str, Path], source_filter: str | None = None) -> List[dict]:
    path_obj = Path(path)
    if isinstance(source_filter, str) and source_filter.strip().lower() in {"", "all", "any", "none"}:
        source_filter = None
    if not path_obj.exists():
        # ... (keep existing candidates logic)
        cwd = Path.cwd()
        candidates = []
        try:
            candidates.extend(str(p) for p in sorted(cwd.glob("**/metrics.jsonl"))[:8])
        except Exception:
            pass
        if not candidates:
            common_roots = [Path.cwd()]
            for root in common_roots:
                if root.exists():
                    try:
                        found = sorted(root.glob("**/metrics.jsonl"))
                        candidates.extend(str(p) for p in found[:8])
                    except Exception:
                        continue
                if candidates:
                    break
        hint_lines = [
            f"metrics file not found: {path_obj}",
            f"current working directory: {cwd}",
            "pass an explicit file, e.g.:",
            "  python train_crqsf.py --metrics_file /path/to/metrics.jsonl",
        ]
        if candidates:
            hint_lines.append("detected candidates:")
            hint_lines.extend(f"  - {c}" for c in candidates)
        raise FileNotFoundError("\n".join(hint_lines))

    records: List[dict] = []
    with path_obj.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}: {exc}") from exc
            if "metrics" not in rec or "theta" not in rec:
                raise ValueError(f"Record at line {line_no} missing 'metrics' or 'theta'")

            if source_filter and rec.get("source") != source_filter:
                continue

            records.append(rec)
    if not records:
        raise ValueError(f"No valid records found in {path} (filter={source_filter})")
    return records


def build_tensors(
    records: Sequence[dict],
    metric_keys: Sequence[str],
    theta_keys: Sequence[str],
    theta_min_clip: float = 1e-8,
    theta_transform: ThetaCoordinateTransform | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_rows: List[List[float]] = []
    theta_rows: List[List[float]] = []

    for i, rec in enumerate(records):
        metrics = rec.get("metrics", {})
        theta_map = rec.get("theta", {})
        if not isinstance(metrics, dict) or not isinstance(theta_map, dict):
            raise ValueError(f"Record index {i} has non-dict metrics/theta")
        x_rows.append([float(metrics.get(k, 0.0)) for k in metric_keys])
        theta_rows.append([_extract_theta_value(theta_map, k) for k in theta_keys])

    x = torch.tensor(x_rows, dtype=torch.float32)
    theta = torch.tensor(theta_rows, dtype=torch.float32)
    if theta_transform is None:
        theta_coordinate = torch.log10(theta.clamp_min(theta_min_clip))
    else:
        theta_coordinate = theta_transform.transform(theta_transform.clamp(theta))
    return x, theta_coordinate


def build_masked_tensors(
    records: Sequence[dict],
    metric_keys: Sequence[str],
    theta_keys: Sequence[str],
    theta_min_clip: float = 1e-8,
    theta_transform: ThetaCoordinateTransform | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build scalar contexts without assigning numeric values to absent bins."""
    x_rows: List[List[float]] = []
    mask_rows: List[List[float]] = []
    theta_rows: List[List[float]] = []
    for i, rec in enumerate(records):
        metrics = rec.get("metrics", {})
        theta_map = rec.get("theta", {})
        metric_mask = rec.get("metric_mask", {})
        if not isinstance(metrics, dict) or not isinstance(theta_map, dict):
            raise ValueError(f"Record index {i} has non-dict metrics/theta")
        if metric_mask is not None and not isinstance(metric_mask, dict):
            raise ValueError(f"Record index {i} has non-dict metric_mask")
        values: List[float] = []
        observed: List[float] = []
        for key in metric_keys:
            value = _finite_float(metrics.get(key))
            declared = metric_mask.get(key) if isinstance(metric_mask, dict) else None
            is_observed = value is not None if declared is None else bool(declared)
            if is_observed and value is None:
                raise ValueError(f"Record index {i} marks metric {key!r} observed without a finite value")
            values.append(float(value) if value is not None else 0.0)
            observed.append(float(is_observed))
        x_rows.append(values)
        mask_rows.append(observed)
        theta_rows.append([_extract_theta_value(theta_map, k) for k in theta_keys])
    x = torch.tensor(x_rows, dtype=torch.float32)
    mask = torch.tensor(mask_rows, dtype=torch.float32)
    theta = torch.tensor(theta_rows, dtype=torch.float32)
    if theta_transform is None:
        theta_coordinate = torch.log10(theta.clamp_min(theta_min_clip))
    else:
        theta_coordinate = theta_transform.transform(theta_transform.clamp(theta))
    return x, mask, theta_coordinate


def _finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def build_prior_over_proposal_sample_weights(
    records: Sequence[dict],
    *,
    proposal_log_prob_keys: Sequence[str] = ("proposal_log_prob", "log_prob", "final_score"),
    min_weight: float = 0.05,
    max_weight: float = 20.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Approximate prior/proposal correction weights from recorded proposal metadata.

    Historical LHS or records without proposal-density metadata are assigned unit
    weight. Active-learning records with a finite proposal log probability receive
    a weight proportional to exp(-log q_proposal(theta)).
    """
    if min_weight <= 0.0:
        raise ValueError("min_weight must be positive")
    if max_weight < min_weight:
        raise ValueError("max_weight must be >= min_weight")

    raw_weights: List[float] = []
    for rec in records:
        log_prob = None
        for key in proposal_log_prob_keys:
            log_prob = _finite_float(rec.get(key))
            if log_prob is not None:
                break
        if log_prob is None:
            raw_weights.append(1.0)
            continue
        raw_weights.append(float(math.exp(max(min(-log_prob, 60.0), -60.0))))

    weights = torch.tensor(raw_weights, dtype=torch.float32).clamp(min=float(min_weight), max=float(max_weight))
    if normalize and weights.numel() > 0:
        weights = weights / weights.mean().clamp_min(1.0e-8)
    return weights.to(dtype=torch.float32)


def build_profile_tensors(
    records: Sequence[dict],
    *,
    field_names: Sequence[str],
    summary_key: str | None = None,
    profile_key: str = "profiles_diff",
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    profile_rows: List[List[List[float]]] = []
    summary_rows: List[List[float]] = []
    expected_bins: int | None = None

    for i, rec in enumerate(records):
        profile_payload = rec.get(profile_key)
        if not isinstance(profile_payload, dict):
            raise ValueError(f"Record index {i} missing '{profile_key}' payload; rebuild metrics.jsonl with profile data")
        n_bins = int(profile_payload.get("n_bins", 0))
        if n_bins <= 0:
            raise ValueError(f"Record index {i} has invalid n_bins in '{profile_key}'")
        if expected_bins is None:
            expected_bins = n_bins
        elif n_bins != expected_bins:
            raise ValueError(f"Record index {i} has n_bins={n_bins}, expected {expected_bins}")

        channels: List[List[float]] = []
        for field in field_names:
            values = profile_payload.get(field)
            if not isinstance(values, list):
                raise ValueError(f"Record index {i} missing profile field '{field}' in '{profile_key}'")
            if len(values) != n_bins:
                raise ValueError(f"Record index {i} field '{field}' length {len(values)} != n_bins {n_bins}")
            channels.append([float(v) for v in values])
        profile_rows.append(channels)

        if summary_key is not None:
            summary_payload = rec.get("summary_features", {})
            if not isinstance(summary_payload, dict):
                raise ValueError(f"Record index {i} has non-dict 'summary_features'")
            summary_rows.append([float(summary_payload.get(summary_key, 0.0))])

    profiles = torch.tensor(profile_rows, dtype=torch.float32)
    summaries = torch.tensor(summary_rows, dtype=torch.float32) if summary_key is not None else None
    return profiles, summaries


def split_dataset_indices(
    n_samples: int,
    *,
    val_fraction: float,
    test_fraction: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if n_samples < 2:
        raise ValueError("Need at least 2 samples to create a train/validation split")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")
    if not (0.0 <= test_fraction < 1.0):
        raise ValueError("test_fraction must be in [0, 1)")
    if (val_fraction + test_fraction) >= 1.0:
        raise ValueError("val_fraction + test_fraction must be < 1")

    need_val = val_fraction > 0.0
    need_test = test_fraction > 0.0
    min_required = 1 + int(need_val) + int(need_test)
    if n_samples < min_required:
        raise ValueError(
            f"Need at least {min_required} samples for train/val/test with "
            f"val_fraction={val_fraction} and test_fraction={test_fraction}, got {n_samples}"
        )

    n_val = int(round(n_samples * val_fraction)) if need_val else 0
    n_test = int(round(n_samples * test_fraction)) if need_test else 0
    if need_val:
        n_val = max(n_val, 1)
    if need_test:
        n_test = max(n_test, 1)

    max_holdout = n_samples - 1
    while (n_val + n_test) > max_holdout:
        if n_test > int(need_test) and n_test >= n_val:
            n_test -= 1
        elif n_val > int(need_val):
            n_val -= 1
        elif n_test > int(need_test):
            n_test -= 1
        else:
            raise ValueError(
                f"Unable to create split with n={n_samples}, "
                f"val_fraction={val_fraction}, test_fraction={test_fraction}"
            )

    perm = torch.randperm(n_samples)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    if train_idx.numel() == 0:
        raise ValueError("Split left no training samples")
    return train_idx, val_idx, test_idx


def make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    sample_weights: torch.Tensor | None = None,
    summary_features: torch.Tensor | None = None,
    shuffle: bool = True,
) -> DataLoader:
    if x.shape[0] == 0:
        raise ValueError("Empty dataset")
    bs = max(1, min(int(batch_size), x.shape[0]))
    tensors = [x]
    if summary_features is not None:
        if summary_features.shape[0] != x.shape[0]:
            raise ValueError("summary_features must have the same length as x and y")
        tensors.append(summary_features)
    tensors.append(y)
    if sample_weights is not None:
        if sample_weights.shape[0] != x.shape[0]:
            raise ValueError("sample_weights must have the same length as x and y")
        tensors.append(sample_weights)
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=bs, shuffle=shuffle)


def compute_metric_distance_weights(
    metrics: torch.Tensor,
    target_metrics: torch.Tensor,
    scale: torch.Tensor,
    strength: float = 0.5,
    floor: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    if target_metrics.ndim == 1:
        target_metrics = target_metrics.unsqueeze(0)
    if target_metrics.shape[0] != 1:
        raise ValueError("target_metrics must be [features] or [1, features]")
    if metrics.shape[1] != target_metrics.shape[1]:
        raise ValueError("metrics and target_metrics feature counts do not match")
    if scale.ndim != 1 or scale.shape[0] != metrics.shape[1]:
        raise ValueError("scale must be a 1D tensor with one value per metric feature")
    if strength < 0.0:
        raise ValueError("strength must be >= 0")
    if not (0.0 < floor <= 1.0):
        raise ValueError("floor must be in (0, 1]")

    scale = scale.clamp_min(eps).reshape(1, -1)
    delta = (metrics - target_metrics.to(metrics)) / scale
    dist2 = delta.pow(2).mean(dim=1)
    weights = floor + (1.0 - floor) * torch.exp(-strength * dist2)
    return weights.to(dtype=torch.float32)


def compute_theta_neighbor_weights(
    candidate_theta_log: torch.Tensor,
    historical_theta_log: torch.Tensor,
    historical_metric_weights: torch.Tensor,
    bandwidth: float = 0.35,
    floor: float = 0.25,
    eps: float = 1e-6,
) -> torch.Tensor:
    if candidate_theta_log.ndim != 2:
        raise ValueError("candidate_theta_log must be 2D")
    if historical_theta_log.ndim != 2:
        raise ValueError("historical_theta_log must be 2D")
    if candidate_theta_log.shape[1] != historical_theta_log.shape[1]:
        raise ValueError("candidate and historical theta feature counts do not match")
    if historical_metric_weights.ndim != 1 or historical_metric_weights.shape[0] != historical_theta_log.shape[0]:
        raise ValueError("historical_metric_weights must have one value per historical sample")
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be > 0")
    if not (0.0 < floor <= 1.0):
        raise ValueError("floor must be in (0, 1]")

    scaled_delta = (candidate_theta_log[:, None, :] - historical_theta_log[None, :, :]) / float(bandwidth)
    dist2 = scaled_delta.pow(2).mean(dim=2)
    kernel = torch.exp(-0.5 * dist2)
    weighted_kernel = kernel * historical_metric_weights.to(candidate_theta_log).reshape(1, -1)
    neighbor_score = weighted_kernel.sum(dim=1) / kernel.sum(dim=1).clamp_min(eps)
    return neighbor_score.clamp(min=floor, max=1.0).to(dtype=torch.float32)


def parse_csv_keys(value: str, fallback: Sequence[str]) -> List[str]:
    value = value.strip()
    if not value:
        return list(fallback)
    return [v.strip() for v in value.split(",") if v.strip()]

# Job-local observation weighting, applied AFTER standardization.
@dataclass
class WeightedObservationStandardizer(MaskedContextStandardizer):
    condition_factors: torch.Tensor | None = None

    def __post_init__(self):
        if self.condition_factors is None:
            self.condition_factors = torch.ones_like(self.mean)
        if self.condition_factors.shape != self.mean.shape or not torch.isfinite(self.condition_factors).all() or (self.condition_factors <= 0).any():
            raise ValueError('Condition factors must be positive, finite and match metric dimensions')

    @classmethod
    def from_data(cls, x, mask, eps=1e-6):
        import os,json
        from pathlib import Path
        config=json.loads(Path(os.environ['SAGE_TARGET_CONDITION_CONFIG']).read_text())
        factors=torch.tensor(config['condition_factors'],dtype=x.dtype,device=x.device)
        mean=torch.tensor(config['observation_means'],dtype=x.dtype,device=x.device)
        std=torch.tensor(config['observation_stds'],dtype=x.dtype,device=x.device)
        if x.shape[-1] != mean.numel() or not torch.isfinite(std).all() or (std<=0).any():
            raise ValueError('Invalid fixed observation normalization')
        return cls(mean,std,factors)

    def transform(self,x,mask):
        encoded=super().transform(x,mask)
        m=self.mean.numel()
        return torch.cat((encoded[...,:m]*self.condition_factors,encoded[...,m:]),dim=-1)

    def transform_target(self,x):
        encoded=super().transform_target(x)
        m=self.mean.numel()
        return torch.cat((encoded[...,:m]*self.condition_factors,encoded[...,m:]),dim=-1)

    def state_dict(self):
        state=super().state_dict()
        state['condition_factors']=self.condition_factors
        state['conditioning']='fixed_observation_family_normalization_then_weighting'
        return state

    @classmethod
    def from_state_dict(cls,state):
        if 'condition_factors' not in state:
            raise ValueError('Missing weighted-condition checkpoint factors')
        return cls(state['mean'],state['std'],state['condition_factors'])

MaskedContextStandardizer = WeightedObservationStandardizer
