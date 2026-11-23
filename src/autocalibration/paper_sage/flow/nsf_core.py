#!/usr/bin/env python3
"""Conditional neural spline flow (NSF) with rational-quadratic spline coupling layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_signed_clamp(x: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.where(x >= 0.0, x.clamp_min(eps), x.clamp_max(-eps))


def _make_mlp(in_dim: int, out_dim: int, hidden_dims: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def rational_quadratic_spline(
    x: torch.Tensor,
    unnorm_widths: torch.Tensor,
    unnorm_heights: torch.Tensor,
    unnorm_derivatives: torch.Tensor,
    inverse: bool = False,
    tail_bound: float = 3.0,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Elementwise RQ spline with linear tails."""
    if x.ndim != 2:
        raise ValueError(f"x must be [batch, dims], got {tuple(x.shape)}")

    batch_size, dims = x.shape
    num_bins = unnorm_widths.shape[-1]
    if unnorm_widths.shape != (batch_size, dims, num_bins):
        raise ValueError("unnorm_widths shape mismatch")
    if unnorm_heights.shape != (batch_size, dims, num_bins):
        raise ValueError("unnorm_heights shape mismatch")
    if unnorm_derivatives.shape != (batch_size, dims, num_bins + 1):
        raise ValueError("unnorm_derivatives shape mismatch")

    widths = F.softmax(unnorm_widths, dim=-1)
    widths = min_bin_width + (2.0 * tail_bound - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), mode="constant", value=0.0)
    cumwidths = cumwidths + (-tail_bound)
    cumwidths[..., -1] = tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = F.softmax(unnorm_heights, dim=-1)
    heights = min_bin_height + (2.0 * tail_bound - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), mode="constant", value=0.0)
    cumheights = cumheights + (-tail_bound)
    cumheights[..., -1] = tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = min_derivative + F.softplus(unnorm_derivatives)

    inside = (x > -tail_bound) & (x < tail_bound)
    y = x.clone()
    logabsdet = torch.zeros_like(x)
    if not inside.any():
        return y, logabsdet

    x_or_y_cum = cumheights if inverse else cumwidths
    search_values = torch.where(inside, x, torch.zeros_like(x))
    bin_idx = torch.searchsorted(x_or_y_cum, search_values.unsqueeze(-1), right=True).squeeze(-1) - 1
    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    gather_idx = bin_idx.unsqueeze(-1)
    x_k = torch.gather(cumwidths, -1, gather_idx).squeeze(-1)
    y_k = torch.gather(cumheights, -1, gather_idx).squeeze(-1)
    w_k = torch.gather(widths, -1, gather_idx).squeeze(-1)
    h_k = torch.gather(heights, -1, gather_idx).squeeze(-1)
    d_k = torch.gather(derivatives, -1, gather_idx).squeeze(-1)
    d_kp1 = torch.gather(derivatives[..., 1:], -1, gather_idx).squeeze(-1)
    stability_eps = 1e-8
    s_k = h_k / w_k.clamp_min(stability_eps)

    if not inverse:
        theta = (x - x_k) / w_k
        theta = torch.clamp(theta, 0.0, 1.0)
        theta_one_minus = theta * (1.0 - theta)
        numerator = h_k * (s_k * theta.pow(2) + d_k * theta_one_minus)
        denominator = s_k + (d_k + d_kp1 - 2.0 * s_k) * theta_one_minus
        denominator = denominator.clamp_min(stability_eps)
        inside_y = y_k + numerator / denominator

        derivative_num = s_k.pow(2) * (
            d_kp1 * theta.pow(2) + 2.0 * s_k * theta_one_minus + d_k * (1.0 - theta).pow(2)
        )
        derivative_num = derivative_num.clamp_min(stability_eps)
        inside_logabsdet = torch.log(derivative_num) - 2.0 * torch.log(denominator)
    else:
        y_minus_yk = x - y_k
        a = y_minus_yk * (d_k + d_kp1 - 2.0 * s_k) + h_k * (s_k - d_k)
        b = h_k * d_k - y_minus_yk * (d_k + d_kp1 - 2.0 * s_k)
        c = -s_k * y_minus_yk

        discriminant = torch.clamp(b.pow(2) - 4.0 * a * c, min=stability_eps)
        root_denom = _safe_signed_clamp(-b - torch.sqrt(discriminant), stability_eps)
        root = (2.0 * c) / root_denom
        theta = torch.clamp(root, 0.0, 1.0)
        theta_one_minus = theta * (1.0 - theta)
        inside_y = x_k + theta * w_k

        denominator = s_k + (d_k + d_kp1 - 2.0 * s_k) * theta_one_minus
        denominator = denominator.clamp_min(stability_eps)
        derivative_num = s_k.pow(2) * (
            d_kp1 * theta.pow(2) + 2.0 * s_k * theta_one_minus + d_k * (1.0 - theta).pow(2)
        )
        derivative_num = derivative_num.clamp_min(stability_eps)
        inside_logabsdet = -(torch.log(derivative_num) - 2.0 * torch.log(denominator))

    y = torch.where(inside, inside_y, y)
    logabsdet = torch.where(inside, inside_logabsdet, logabsdet)
    return y, logabsdet


class ConditionalSplineCoupling(nn.Module):
    def __init__(
        self,
        features: int,
        context_features: int,
        mask: torch.Tensor,
        num_bins: int = 8,
        hidden_dims: Sequence[int] = (128, 128),
        tail_bound: float = 3.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mask.ndim != 1 or mask.numel() != features:
            raise ValueError("mask must be [features]")
        if torch.all(mask == 1) or (features > 1 and torch.all(mask == 0)):
            raise ValueError("mask must include both transformed and identity dimensions")

        self.features = features
        self.context_features = context_features
        self.num_bins = num_bins
        self.tail_bound = tail_bound

        mask = mask.bool()
        self.register_buffer("identity_mask", mask)
        self.register_buffer("transform_mask", ~mask)

        self.identity_idx = torch.where(mask)[0]
        self.transform_idx = torch.where(~mask)[0]
        in_dim = int(self.identity_idx.numel()) + context_features
        out_dim = int(self.transform_idx.numel()) * (3 * num_bins + 1)
        self.net = _make_mlp(in_dim, out_dim, hidden_dims, dropout=dropout)

    def _conditioner(self, x_identity: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = torch.cat([x_identity, context], dim=-1)
        raw = self.net(h)
        num_t = self.transform_idx.numel()
        raw = raw.view(raw.shape[0], num_t, 3 * self.num_bins + 1)
        widths = raw[..., : self.num_bins]
        heights = raw[..., self.num_bins : 2 * self.num_bins]
        derivatives = raw[..., 2 * self.num_bins :]
        return widths, heights, derivatives

    def forward(self, x: torch.Tensor, context: torch.Tensor, inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x_id = x[:, self.identity_idx]
        x_tr = x[:, self.transform_idx]
        widths, heights, derivatives = self._conditioner(x_id, context)

        y_tr, logabsdet_tr = rational_quadratic_spline(
            x_tr,
            widths,
            heights,
            derivatives,
            inverse=inverse,
            tail_bound=self.tail_bound,
        )
        y = x.clone()
        y[:, self.identity_idx] = x_id
        y[:, self.transform_idx] = y_tr
        return y, logabsdet_tr.sum(dim=-1)


@dataclass
class FlowConfig:
    features: int
    context_features: int
    num_layers: int = 6
    num_bins: int = 8
    hidden_dims: Tuple[int, ...] = (128, 128)
    tail_bound: float = 3.0
    dropout: float = 0.0


class ConditionalNeuralSplineFlow(nn.Module):
    """Conditional normalizing flow p(x | context)."""

    def __init__(self, config: FlowConfig):
        super().__init__()
        self.config = config
        transforms: List[ConditionalSplineCoupling] = []
        for i in range(config.num_layers):
            mask = torch.tensor([config.features > 1 and (j + i) % 2 == 0 for j in range(config.features)], dtype=torch.bool)
            transforms.append(
                ConditionalSplineCoupling(
                    features=config.features,
                    context_features=config.context_features,
                    mask=mask,
                    num_bins=config.num_bins,
                    hidden_dims=config.hidden_dims,
                    tail_bound=config.tail_bound,
                    dropout=config.dropout,
                )
            )
        self.transforms = nn.ModuleList(transforms)

    def _forward_to_base(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        logabsdet = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for t in self.transforms:
            z, lad = t(z, context, inverse=False)
            logabsdet = logabsdet + lad
        return z, logabsdet

    def _inverse_from_base(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        logabsdet = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        for t in reversed(self.transforms):
            x, lad = t(x, context, inverse=True)
            logabsdet = logabsdet + lad
        return x, logabsdet

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x).all():
            raise FloatingPointError("Non-finite theta input reached flow.log_prob")
        if not torch.isfinite(context).all():
            raise FloatingPointError("Non-finite context input reached flow.log_prob")
        z, logabsdet = self._forward_to_base(x, context)
        if not torch.isfinite(z).all():
            raise FloatingPointError("Non-finite latent state produced inside flow._forward_to_base")
        if not torch.isfinite(logabsdet).all():
            raise FloatingPointError("Non-finite logabsdet produced inside flow._forward_to_base")
        log_base = -0.5 * (z.pow(2) + torch.log(torch.tensor(2.0 * torch.pi, device=z.device, dtype=z.dtype))).sum(dim=-1)
        out = log_base + logabsdet
        if not torch.isfinite(out).all():
            raise FloatingPointError("Non-finite log_prob produced by ConditionalNeuralSplineFlow")
        return out

    @torch.no_grad()
    def sample(self, num_samples: int, context: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError("context must be [batch, context_features]")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if context.shape[0] == 1:
            context = context.repeat(num_samples, 1)
        elif context.shape[0] != num_samples:
            raise ValueError("context batch size must be 1 or num_samples")
        z = torch.randn(num_samples, self.config.features, device=context.device, dtype=context.dtype) * temperature
        x, _ = self._inverse_from_base(z, context)
        return x


def nll_loss(
    model: ConditionalNeuralSplineFlow,
    x: torch.Tensor,
    context: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample_nll = -model.log_prob(x, context)
    if sample_weights is None:
        return per_sample_nll.mean()
    weights = sample_weights.reshape(-1).to(device=per_sample_nll.device, dtype=per_sample_nll.dtype)
    weight_sum = weights.sum().clamp_min(1e-12)
    return (per_sample_nll * weights).sum() / weight_sum


def parse_hidden_dims(value: str | Iterable[int]) -> Tuple[int, ...]:
    if isinstance(value, str):
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)
