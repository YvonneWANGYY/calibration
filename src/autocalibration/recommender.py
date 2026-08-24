"""Recommendation scoring for lightweight adaptive calibration."""

import math
import random
from collections.abc import Sequence

from .lhs import latin_hypercube, to_unit
from .session_types import Observation, ParameterSpec


def recommend_from_observations(
    parameters: Sequence[ParameterSpec],
    observations: Sequence[Observation],
    losses: Sequence[float],
    count: int,
    candidate_pool_size: int = 256,
    seed: int | None = None,
) -> list[dict[str, object]]:
    if count <= 0:
        raise ValueError("number of recommendations must be positive")
    if candidate_pool_size < count:
        raise ValueError("candidate_pool_size must be >= number of recommendations")
    if not observations:
        return [
            {"rank": index + 1, "kind": "initial_lhs", "parameters": point}
            for index, point in enumerate(latin_hypercube(parameters, count, seed=seed))
        ]

    rng = random.Random(seed)
    candidates = latin_hypercube(parameters, candidate_pool_size, seed=rng.randrange(2**31))
    scored = []
    selected_units: list[list[float]] = []
    observation_units = [_unit_vector(parameters, observation.parameters) for observation in observations]

    for candidate in candidates:
        candidate_unit = _unit_vector(parameters, candidate)
        nearest_loss, nearest_distance = _nearest_loss(candidate_unit, observation_units, losses)
        exploration_bonus = 0.05 * nearest_distance
        score = nearest_loss - exploration_bonus
        scored.append(
            {
                "kind": "adaptive",
                "parameters": candidate,
                "score": score,
                "nearest_loss": nearest_loss,
                "nearest_distance": nearest_distance,
            }
        )

    recommendations = []
    for item in sorted(scored, key=lambda row: (row["score"], -row["nearest_distance"])):
        candidate_unit = _unit_vector(parameters, item["parameters"])
        if selected_units and min(_distance(candidate_unit, other) for other in selected_units) < 1e-9:
            continue
        selected_units.append(candidate_unit)
        item["rank"] = len(recommendations) + 1
        recommendations.append(item)
        if len(recommendations) == count:
            break
    return recommendations


def _nearest_loss(candidate: list[float], observations: list[list[float]], losses: Sequence[float]) -> tuple[float, float]:
    best_index = 0
    best_distance = float("inf")
    for index, observation in enumerate(observations):
        distance = _distance(candidate, observation)
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return float(losses[best_index]), best_distance


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _unit_vector(parameters: Sequence[ParameterSpec], values: dict[str, float]) -> list[float]:
    return [to_unit(parameter, values[parameter.name]) for parameter in parameters]
