"""One-dimensional heat diffusion in a 10 cm rod, sampled after 60 seconds."""
import json
import math
import sys


def integration_steps(diffusivity_cm2_s, cells=100):
    if not math.isfinite(diffusivity_cm2_s) or diffusivity_cm2_s <= 0:
        raise ValueError("diffusivity must be positive and finite")
    dx = 10.0 / cells
    return math.ceil(60 * diffusivity_cm2_s / (0.45 * dx**2))


def temperature_profile(diffusivity_cm2_s, heater_c, cells=100):
    """Advance the heat equation on a uniform grid with fixed end temperatures."""
    dx = 10.0 / cells
    steps = integration_steps(diffusivity_cm2_s, cells)
    coefficient = diffusivity_cm2_s * (60 / steps) / dx**2
    temperature = [heater_c] + [20.0] * cells
    for _ in range(steps):
        temperature = [heater_c] + [
            temperature[i] + coefficient * (temperature[i-1] - 2*temperature[i] + temperature[i+1])
            for i in range(1, cells)
        ] + [20.0]
    return temperature


if __name__ == "__main__":
    parameters = json.load(sys.stdin)
    profile = temperature_profile(**parameters)
    outputs = {f"temp_{x}cm": profile[x * 10] for x in (1, 3, 5)}
    if "--scalar-cost" in sys.argv[1:]:
        outputs["cost"] = integration_steps(parameters["diffusivity_cm2_s"])
    elif "--report-cost" in sys.argv[1:]:
        outputs["solver_steps"] = integration_steps(parameters["diffusivity_cm2_s"])
    print(json.dumps(outputs))
