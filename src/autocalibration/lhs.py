"""Latin-hypercube initial-design generation."""

import math
import random
from collections.abc import Sequence

from .session_types import ParameterSpec


def latin_hypercube(parameters: Sequence[ParameterSpec], count: int, seed: int | None = None) -> list[dict[str, float]]:
    if count <= 0:
        raise ValueError("number of recommendations must be positive")

    rng = random.Random(seed)
    columns: dict[str, list[float]] = {}
    for parameter in parameters:
        values = []
        for index in range(count):
            unit_value = (index + rng.random()) / count
            values.append(_from_unit(parameter, unit_value))
        rng.shuffle(values)
        columns[parameter.name] = values

    return [
        {parameter.name: columns[parameter.name][row] for parameter in parameters}
        for row in range(count)
    ]


def to_unit(parameter: ParameterSpec, value: float) -> float:
    if parameter.scale == "linear":
        return (value - parameter.lower) / (parameter.upper - parameter.lower)
    lower = math.log(parameter.lower)
    upper = math.log(parameter.upper)
    return (math.log(value) - lower) / (upper - lower)


def _from_unit(parameter: ParameterSpec, unit_value: float) -> float:
    if parameter.scale == "linear":
        return parameter.lower + unit_value * (parameter.upper - parameter.lower)
    lower = math.log(parameter.lower)
    upper = math.log(parameter.upper)
    return math.exp(lower + unit_value * (upper - lower))
