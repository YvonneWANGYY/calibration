"""Shared data structures for adaptive calibration sessions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("parameter name is required")
        if self.upper <= self.lower:
            raise ValueError(f"{self.name}: upper bound must be greater than lower bound")
        if self.scale not in {"linear", "log"}:
            raise ValueError(f"{self.name}: scale must be 'linear' or 'log'")
        if self.scale == "log" and self.lower <= 0:
            raise ValueError(f"{self.name}: log-scale lower bound must be positive")


@dataclass(frozen=True)
class Observation:
    parameters: dict[str, float]
    metrics: dict[str, float]
