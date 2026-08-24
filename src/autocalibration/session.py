"""User-facing adaptive calibration session API."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .io import load_mapping, load_observations
from .recommender import recommend_from_observations
from .session_types import Observation, ParameterSpec


class AdaptiveCalibrationSession:
    """A lightweight recommendation loop around user-supplied simulation metrics."""

    def __init__(
        self,
        parameters: Sequence[ParameterSpec],
        target_profile: Mapping[str, float],
        observations: Sequence[Observation] | None = None,
    ) -> None:
        if not parameters:
            raise ValueError("at least one parameter is required")
        self.parameters = list(parameters)
        self.target_profile = {str(key): float(value) for key, value in target_profile.items()}
        if not self.target_profile:
            raise ValueError("target_profile must contain at least one metric")
        self.observations = list(observations or [])
        for observation in self.observations:
            self._validate_observation(observation)

    @classmethod
    def from_inputs(
        cls,
        parameters: Sequence[ParameterSpec],
        target_profile: Mapping[str, float],
        initial_observations: Sequence[Mapping[str, Any]] | None = None,
    ) -> "AdaptiveCalibrationSession":
        observations = [_coerce_observation(item) for item in initial_observations or []]
        return cls(parameters=parameters, target_profile=target_profile, observations=observations)

    @classmethod
    def from_files(
        cls,
        parameters: Sequence[ParameterSpec],
        target_profile: str | Path | Mapping[str, float],
        metrics_file: str | Path | None = None,
        metric_columns: Sequence[str] | None = None,
        parameter_paths: Mapping[str, str] | None = None,
        metric_paths: Mapping[str, str] | None = None,
    ) -> "AdaptiveCalibrationSession":
        target = load_mapping(target_profile)
        session = cls(parameters=parameters, target_profile=target)
        if metrics_file is not None:
            session.add_observations_from_file(
                metrics_file,
                metric_columns=metric_columns,
                parameter_paths=parameter_paths,
                metric_paths=metric_paths,
            )
        return session

    def add_observation(self, parameters: Mapping[str, float], metrics: Mapping[str, float]) -> None:
        observation = Observation(
            parameters={str(key): float(value) for key, value in parameters.items()},
            metrics={str(key): float(value) for key, value in metrics.items()},
        )
        self._validate_observation(observation)
        self.observations.append(observation)

    def add_observations_from_file(
        self,
        metrics_file: str | Path,
        metric_columns: Sequence[str] | None = None,
        parameter_paths: Mapping[str, str] | None = None,
        metric_paths: Mapping[str, str] | None = None,
    ) -> None:
        observations = load_observations(
            metrics_file,
            parameter_names=[parameter.name for parameter in self.parameters],
            metric_columns=metric_columns,
            parameter_paths=parameter_paths,
            metric_paths=metric_paths,
        )
        for observation in observations:
            self.add_observation(observation["parameters"], observation["metrics"])

    def recommend(
        self,
        n: int,
        candidate_pool_size: int = 256,
        seed: int | None = None,
    ) -> list[dict[str, object]]:
        return recommend_from_observations(
            parameters=self.parameters,
            observations=self.observations,
            losses=self.losses(),
            count=n,
            candidate_pool_size=candidate_pool_size,
            seed=seed,
        )

    def losses(self) -> list[float]:
        return [self._loss(observation.metrics) for observation in self.observations]

    def _loss(self, metrics: Mapping[str, float]) -> float:
        loss = 0.0
        for name, target_value in self.target_profile.items():
            if name not in metrics:
                raise ValueError(f"observation is missing target metric {name!r}")
            loss += (float(metrics[name]) - target_value) ** 2
        return loss

    def _validate_observation(self, observation: Observation) -> None:
        for parameter in self.parameters:
            if parameter.name not in observation.parameters:
                raise ValueError(f"observation is missing parameter {parameter.name!r}")
            value = observation.parameters[parameter.name]
            if value < parameter.lower or value > parameter.upper:
                raise ValueError(f"{parameter.name}={value} is outside [{parameter.lower}, {parameter.upper}]")
        self._loss(observation.metrics)


def _coerce_observation(item: Mapping[str, Any]) -> Observation:
    parameters = item.get("parameters")
    metrics = item.get("metrics")
    if not isinstance(parameters, Mapping) or not isinstance(metrics, Mapping):
        raise ValueError("each observation must contain parameters and metrics objects")
    return Observation(
        parameters={str(key): float(value) for key, value in parameters.items()},
        metrics={str(key): float(value) for key, value in metrics.items()},
    )
