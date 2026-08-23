#!/usr/bin/env python3
"""Materialize generic adaptive-calibration recipes into command artifacts."""

import argparse
import json
import re
import shlex
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = "scripts/<your_adaptive_calibration_engine>.py"
REQUIRED_TOP_LEVEL = (
    "science_target",
    "parameter_space",
    "initial_design",
    "simulator",
    "proposal_model",
    "run_control",
    "outputs",
)


@dataclass(frozen=True)
class LaunchPlan:
    recipe: Mapping[str, Any]
    command: list[str]
    manifest_path: Path
    generated_at: str

    @property
    def command_text(self) -> str:
        return " ".join(shlex.quote(str(part)) for part in self.command)


def load_recipe(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"recipe must be a JSON object: {path}")
    return payload


def build_launch_plan(
    recipe: Mapping[str, Any],
    recipe_path: Path | None = None,
    project_root: Path = ROOT,
    now: Callable[[], str] | None = None,
) -> LaunchPlan:
    project_root = Path(project_root).resolve()
    name = _required_string(recipe, "name", "name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("recipe name may only contain letters, numbers, '.', '-', and '_'")

    for key in REQUIRED_TOP_LEVEL:
        if key not in recipe:
            raise ValueError(f"recipe field {key!r} is required")

    science_target = _required_mapping(recipe, "science_target")
    initial_design = _required_mapping(recipe, "initial_design")
    simulator = _required_mapping(recipe, "simulator")
    proposal_model = _required_mapping(recipe, "proposal_model")
    run_control = _required_mapping(recipe, "run_control")
    outputs = _required_mapping(recipe, "outputs")
    parameter_space = _parameter_space(recipe["parameter_space"])

    max_iters = _positive_int(run_control.get("max_iters"), "run_control.max_iters")
    start_iter = _nonnegative_int(run_control.get("start_iter", 0), "run_control.start_iter")
    if start_iter >= max_iters:
        raise ValueError("run_control.start_iter must be smaller than run_control.max_iters")
    num_samples = _positive_int(
        run_control.get("num_samples_per_iter"),
        "run_control.num_samples_per_iter",
    )
    random_seed = _nonnegative_int(run_control.get("random_seed", 0), "run_control.random_seed")

    output_root = _required_string(outputs, "root", "outputs.root")
    manifest_name = str(outputs.get("manifest") or "launch_manifest.json")
    if Path(manifest_name).is_absolute() or ".." in Path(manifest_name).parts:
        raise ValueError("outputs.manifest must be a relative filename")
    manifest_path = _resolve_path(output_root, project_root) / manifest_name

    python_exec = str(recipe.get("python_exec") or "python")
    engine = str(recipe.get("engine") or DEFAULT_ENGINE)

    command = [python_exec, engine]
    if recipe_path is not None:
        _add_arg(command, "--recipe", _display_path(str(recipe_path)))
    _add_arg(command, "--target", _required_string(science_target, "target_file", "science_target.target_file"))
    _add_arg(
        command,
        "--initial-summary",
        _required_string(initial_design, "summary", "initial_design.summary"),
    )
    _add_arg(
        command,
        "--simulator-wrapper",
        _required_string(simulator, "wrapper", "simulator.wrapper"),
    )
    _add_arg(
        command,
        "--proposal-model-root",
        _required_string(proposal_model, "model_root", "proposal_model.model_root"),
    )
    _add_arg(command, "--output-root", output_root)
    _add_arg(command, "--run-label", _required_string(outputs, "label", "outputs.label"))
    _add_arg(command, "--start-iter", start_iter)
    _add_arg(command, "--max-iters", max_iters)
    _add_arg(command, "--num-samples-per-iter", num_samples)
    _add_arg(command, "--random-seed", random_seed)
    _add_arg(command, "--parameter-space-json", json.dumps(parameter_space, separators=(",", ":")))

    return LaunchPlan(
        recipe=dict(recipe),
        command=command,
        manifest_path=manifest_path,
        generated_at=now() if now is not None else datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(plan: LaunchPlan, force: bool = False) -> None:
    _ensure_writable(plan.manifest_path, force)
    plan.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    plan.manifest_path.write_text(json.dumps(_manifest_payload(plan), indent=2, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recipe = load_recipe(args.recipe)
    plan = build_launch_plan(recipe, recipe_path=args.recipe, project_root=args.project_root)
    if args.write_manifest:
        write_manifest(plan, force=args.force)
        print(f"Manifest path: {plan.manifest_path}")
    else:
        print("DRY RUN")
        print(f"Manifest path: {plan.manifest_path}")
    print("Command for scheduler wrapper:")
    print(plan.command_text)
    return 0


def _manifest_payload(plan: LaunchPlan) -> dict[str, Any]:
    return {
        "recipe": plan.recipe,
        "command": plan.command,
        "command_text": plan.command_text,
        "manifest_path": str(plan.manifest_path),
        "generated_at": plan.generated_at,
    }


def _parameter_space(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("parameter_space must be a non-empty list")

    normalized = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"parameter_space[{index}] must be an object")
        name = _required_string(item, "name", f"parameter_space[{index}].name")
        lower = float(item.get("min"))
        upper = float(item.get("max"))
        if upper <= lower:
            raise ValueError(f"parameter_space[{index}].max must be greater than min")
        scale = str(item.get("scale") or "linear")
        if scale not in {"linear", "log"}:
            raise ValueError(f"parameter_space[{index}].scale must be 'linear' or 'log'")
        normalized.append({"name": name, "min": lower, "max": upper, "scale": scale})
    return normalized


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"recipe field {key!r} must be an object")
    return value


def _required_string(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"{label} is required")
    return str(value)


def _positive_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _nonnegative_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{label} must be non-negative")
    return parsed


def _add_arg(command: list[str], flag: str, value: Any) -> None:
    command.extend([flag, str(value)])


def _display_path(value: str) -> str:
    return str(Path(value).expanduser())


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _ensure_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass --force to overwrite")


if __name__ == "__main__":
    raise SystemExit(main())
