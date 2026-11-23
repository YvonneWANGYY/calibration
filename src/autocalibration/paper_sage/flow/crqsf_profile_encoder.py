#!/usr/bin/env python3
"""Profile encoder and profile-conditioned flow wrapper for CRQSF."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from crqsf_core import ConditionalRQSplineFlow


@dataclass
class ProfileEncoderConfig:
    n_bins: int = 768
    n_channels: int = 4
    embed_dim: int = 32
    conv_channels: int = 64
    conv_layers: int = 4
    kernel_size: int = 5


def _group_norm_groups(num_channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, num_channels), 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1


class ProfileEncoder(nn.Module):
    """Compress radial profile channels into a fixed-length embedding."""

    def __init__(self, config: ProfileEncoderConfig):
        super().__init__()
        self.config = config
        if config.conv_channels <= 0:
            raise ValueError("conv_channels must be > 0")
        if config.conv_layers <= 0:
            raise ValueError("conv_layers must be > 0")
        if config.kernel_size <= 0 or config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        layers = []
        in_channels = config.n_channels
        padding = config.kernel_size // 2
        for _ in range(config.conv_layers):
            layers.extend(
                [
                    nn.Conv1d(in_channels, config.conv_channels, kernel_size=config.kernel_size, padding=padding),
                    nn.GroupNorm(_group_norm_groups(config.conv_channels), config.conv_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = config.conv_channels
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.backbone = nn.Sequential(*layers)
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(config.conv_channels, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
        )

    def forward(self, profiles: torch.Tensor) -> torch.Tensor:
        if profiles.ndim != 3:
            raise ValueError("profiles must be [batch, channels, n_bins]")
        if profiles.shape[1] != self.config.n_channels:
            raise ValueError(f"expected {self.config.n_channels} profile channels, got {profiles.shape[1]}")
        if profiles.shape[2] != self.config.n_bins:
            raise ValueError(f"expected {self.config.n_bins} profile bins, got {profiles.shape[2]}")
        if not torch.isfinite(profiles).all():
            raise FloatingPointError("Non-finite raw profile input reached ProfileEncoder")
        backbone_out = self.backbone(profiles)
        if not torch.isfinite(backbone_out).all():
            raise FloatingPointError("Non-finite activation produced by ProfileEncoder backbone")
        proj_out = self.proj(backbone_out)
        if not torch.isfinite(proj_out).all():
            raise FloatingPointError("Non-finite context embedding produced by ProfileEncoder projection")
        return proj_out


class ProfileConditionedFlow(nn.Module):
    def __init__(self, profile_encoder: ProfileEncoder, flow: ConditionalRQSplineFlow):
        super().__init__()
        self.profile_encoder = profile_encoder
        self.flow = flow

    def encode_context(self, profiles: torch.Tensor, summary_features: torch.Tensor | None = None) -> torch.Tensor:
        if summary_features is not None:
            raise ValueError("summary_features are disabled in this branch; context must be profile points only")
        if not torch.isfinite(profiles).all():
            raise FloatingPointError("Non-finite profile tensor reached ProfileConditionedFlow.encode_context")
        ctx = self.profile_encoder(profiles)
        if not torch.isfinite(ctx).all():
            raise FloatingPointError("Non-finite encoded context produced by ProfileConditionedFlow.encode_context")
        return ctx

    def log_prob(
        self,
        theta: torch.Tensor,
        profiles: torch.Tensor,
        summary_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = self.encode_context(profiles, summary_features)
        return self.flow.log_prob(theta, ctx)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        profiles: torch.Tensor,
        summary_features: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        ctx = self.encode_context(profiles, summary_features)
        return self.flow.sample(num_samples, ctx, temperature=temperature)
