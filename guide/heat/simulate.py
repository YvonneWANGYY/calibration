"""One-dimensional heat diffusion in a 10 cm rod, sampled after 60 seconds."""
import json
import math
import sys


def temperature_profile(diffusivity_cm2_s, heater_c, cells=100):
    """Advance the heat equation on a uniform grid with fixed end temperatures."""
    if not math.isfinite(diffusivity_cm2_s) or diffusivity_cm2_s <= 0:
        raise ValueError("diffusivity must be positive and finite")
    dx = 10.0 / cells
    steps = math.ceil(60 * diffusivity_cm2_s / (0.45 * dx**2))
    coefficient = diffusivity_cm2_s * (60 / steps) / dx**2
    temperature = [heater_c] + [20.0] * cells
    for _ in range(steps):
        temperature = [heater_c] + [
            temperature[i] + coefficient * (temperature[i-1] - 2*temperature[i] + temperature[i+1])
            for i in range(1, cells)
        ] + [20.0]
    return temperature


if __name__ == "__main__":
    profile = temperature_profile(**json.load(sys.stdin))
    print(json.dumps({f"temp_{x}cm": profile[x * 10] for x in (1, 3, 5)}))
