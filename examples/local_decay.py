"""Calibrate an exponential decay curve at two observation times."""

import argparse
import json
import math
from dataclasses import asdict

from autocalibration import AdaptiveCalibrationSession, ParameterSpec


def simulate(amplitude: float, rate: float) -> dict[str, float]:
    return {
        "signal_t1": amplitude * math.exp(-rate),
        "signal_t2": amplitude * math.exp(-2 * rate),
    }


def run(seed: int) -> dict:
    parameters = [ParameterSpec("amplitude", 0.5, 2.0), ParameterSpec("rate", 0.1, 1.0)]
    target = {"signal_t1": 0.8, "signal_t2": 0.5}
    session = AdaptiveCalibrationSession(parameters, target)
    for point in session.recommend(n=8, seed=seed):
        theta = point["parameters"]
        session.add_observation(theta, simulate(**theta))

    return {
        "seed": seed,
        "parameters": [asdict(parameter) for parameter in parameters],
        "target": target,
        "initial_evaluations": [
            {**asdict(observation), "loss": loss}
            for observation, loss in zip(session.observations, session.losses())
        ],
        "next_recommendation": session.recommend(n=1, seed=seed + 1)[0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(run(parser.parse_args().seed), indent=2))
