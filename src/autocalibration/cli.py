"""Create and run local calibration projects."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from . import AdaptiveCalibrationSession, ParameterSpec


def numeric_map(values, label):
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{label} must be a nonempty JSON object")
    for name, value in values.items():
        if not name or type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"{label} field {name!r} must have a name and finite numeric value")
    return values


def read_problem(path):
    problem = json.loads(path.read_text())
    if not isinstance(problem, dict):
        raise ValueError("problem must be a JSON object")
    bounds = problem["parameters"]
    if not isinstance(bounds, list):
        raise ValueError("parameters must be a list")
    parameters = []
    for item in bounds:
        spec = ParameterSpec(**item)
        if not isinstance(spec.name, str):
            raise ValueError("parameter name must be a string")
        numeric_map({"lower": spec.lower, "upper": spec.upper}, spec.name)
        parameters.append(spec)
    if len({spec.name for spec in parameters}) != len(parameters):
        raise ValueError("parameter names must be unique")

    session = AdaptiveCalibrationSession(parameters, numeric_map(problem["target"], "target"))
    command = problem["simulator"]
    if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
        raise ValueError("simulator must be a nonempty list of command arguments")
    for key, minimum in (("initial_samples", 1), ("adaptive_steps", 0), ("candidate_pool_size", 1)):
        if type(problem[key]) is not int or problem[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if type(problem["seed"]) is not int:
        raise ValueError("seed must be an integer")
    return problem, session


def proposals(session, problem):
    initial = problem["initial_samples"]
    pool = problem["candidate_pool_size"]
    seed = problem["seed"]
    yield from session.recommend(initial, candidate_pool_size=max(pool, initial), seed=seed)
    for step in range(problem["adaptive_steps"]):
        yield from session.recommend(1, candidate_pool_size=pool, seed=seed + step + 1)


def run(problem_path, output):
    problem, session = read_problem(problem_path)
    command = [sys.executable if arg == "{python}" else arg for arg in problem["simulator"]]
    output.mkdir(parents=True, exist_ok=False)
    (output / "problem.json").write_text(json.dumps(problem, indent=2) + "\n")
    best = None
    with (output / "evaluations.jsonl").open("w") as history:
        for index, point in enumerate(proposals(session, problem), start=1):
            theta = point["parameters"]
            try:
                completed = subprocess.run(
                    command, cwd=problem_path.resolve().parent, input=json.dumps(theta),
                    stdout=subprocess.PIPE, text=True, check=True,
                )
                metrics = numeric_map(json.loads(completed.stdout), "simulator metrics")
                session.add_observation(theta, metrics)
                loss = session.losses()[-1]
                row = dict(point, evaluation=index, metrics=metrics, loss=loss)
                history.write(json.dumps(row, allow_nan=False) + "\n")
                history.flush()
            except (ValueError, OverflowError, OSError, subprocess.SubprocessError) as exc:
                raise ValueError(f"evaluation {index} failed for {theta}: {exc}") from exc
            if best is None or loss < best["loss"]:
                best = row
    result = {"evaluations": len(session.observations), "best_observed": best}
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    init = commands.add_parser("init", help="create an editable example project")
    init.add_argument("directory", type=Path)
    execute = commands.add_parser("run", help="evaluate a calibration problem locally")
    execute.add_argument("problem", type=Path)
    execute.add_argument("--output", type=Path, required=True, help="new run directory")
    reproduce = commands.add_parser("reproduce-sage", help="reselect and score the manuscript SAGE models")
    reproduce.add_argument("directory", nargs="?", type=Path, default=Path(__file__).with_name("paper_sage"),
                        help="archived study directory (defaults to the bundled study)")
    proposal = commands.add_parser("propose-sage", help="train and rank the first SMF50 fixed-baryon proposal pool")
    proposal.add_argument("--output", required=True, type=Path, help="new directory")
    proposal.add_argument("--smoke", action="store_true", help="one epoch and 64 draws")
    args = parser.parse_args()
    try:
        if args.action == "init":
            args.directory.mkdir(parents=True, exist_ok=False)
            for name in ("problem.json", "simulate.py"):
                shutil.copyfile(Path(__file__).with_name("starter") / name, args.directory / name)
            print(f"Created {args.directory / 'problem.json'}")
        elif args.action == "propose-sage":
            from .paper_sage.propose import propose
            propose(args.output, args.smoke)
        elif args.action == "reproduce-sage":
            from .sage import replay
            print(json.dumps(list(replay(args.directory)), indent=2, allow_nan=False))
        else:
            result = run(args.problem, args.output)
            print(f"Completed {result['evaluations']} evaluations. Results in {args.output}")
    except (KeyError, TypeError, ValueError, OSError, OverflowError) as exc:
        parser.exit(1, f"autocalibration: {exc}\n")


if __name__ == "__main__":
    main()
