#!/usr/bin/env python3
"""Materialize declarative SAGE active-calibration recipes into launch artifacts."""

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[1]
ENGINE = "scripts/run_sage_crqsf_active_learning.py"

# USER EDIT REQUIRED:
# DEFAULT_PBS is a cluster-specific fallback used only when recipe["pbs"] omits
# a field. Edit these values, or override them in each recipe, before running on
# another scheduler. On this cluster, nodetype A/B/C maps to A=node01-node06,
# B=node07-node13, and C=node14 GPU.
DEFAULT_PBS = {
    "walltime": "72:00:00",
    "ncpus": 12,
    "mpiprocs": 1,
    "nodetype": "C",
    "host": "node14",
    "ngpus": 1,
}

NODETYPE_HOSTS = {
    "A": {f"node{idx:02d}" for idx in range(1, 7)},
    "B": {f"node{idx:02d}" for idx in range(7, 14)},
    "C": {"node14"},
}

STRATEGY_PRESETS: dict[str, dict[str, Any]] = {
    "crqsf_rank": {
        "acquisition_mode": "crqsf",
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "sample_topk",
        "sampling_weight_strength": 0.0,
    },
    "direct_sample": {
        "acquisition_mode": "crqsf",
        "proposal_candidates": 1,
        "candidate_pool_size": 1,
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "direct_sample",
        "sampling_weight_strength": 0.0,
    },
    "tempered_uniform_mixture": {
        "acquisition_mode": "crqsf",
        "proposal_candidates": 1,
        "candidate_pool_size": 1,
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "tempered_uniform_mixture",
        "sampling_weight_strength": 0.0,
        "sampling_temperature_schedule": "2.0,1.8,1.6,1.5,1.4,1.3,1.2,1.1,1.05,1.0",
        "uniform_mixture_epsilon_schedule": "0.30,0.25,0.20,0.20,0.15,0.10,0.10,0.05,0.05,0.0",
    },
    "simple_mixture_mindist": {
        "acquisition_mode": "crqsf",
        "proposal_candidates": 1,
        "candidate_pool_size": 1,
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "tempered_uniform_mixture",
        "sampling_temperature": 1.0,
        "sampling_weight_strength": 0.0,
        "uniform_mixture_epsilon_schedule": "0.30,0.25,0.20,0.20,0.15,0.10,0.10,0.05,0.05,0.0",
        "min_existing_normalized_distance": 0.05,
    },
    "mle": {
        "acquisition_mode": "crqsf",
        "proposal_candidates": 64,
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "mle",
        "mle_starts": 64,
        "mle_steps": 200,
        "mle_lr": 0.03,
        "sampling_weight_strength": 0.0,
    },
    "ensemble_map": {
        "acquisition_mode": "crqsf",
        "proposal_candidates": 64,
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "mle",
        "sampling_weight_strength": 0.0,
    },
    "ensemble_direct_sample": {
        "acquisition_mode": "crqsf",
        "proposal_candidates": 1,
        "candidate_pool_size": 1,
        "proposal_sampling_mode": "joint",
        "proposal_selection_mode": "direct_sample",
        "sampling_weight_strength": 0.0,
    },
}

OPTIONAL_FLAG_FIELDS = [
    ("metric_keys", "--metric-keys"),
    ("surrogate_k", "--surrogate-k"),
    ("n_trials", "--n-trials"),
    ("train_epochs", "--train-epochs"),
    ("candidate_pool_size", "--candidate-pool-size"),
    ("val_fraction", "--val-fraction"),
    ("test_fraction", "--test-fraction"),
    ("sampling_temperature", "--sampling-temperature"),
    ("sampling_weight_strength", "--sampling-weight-strength"),
    ("sampling_weight_floor", "--sampling-weight-floor"),
    ("sampling_theta_bandwidth", "--sampling-theta-bandwidth"),
    ("proposal_sampling_mode", "--proposal-sampling-mode"),
    ("proposal_selection_mode", "--proposal-selection-mode"),
    ("mle_starts", "--mle-starts"),
    ("mle_steps", "--mle-steps"),
    ("mle_lr", "--mle-lr"),
    ("mcmc_proposal_scale", "--mcmc-proposal-scale"),
    ("uniform_mixture_epsilon", "--uniform-mixture-epsilon"),
    ("sampling_temperature_schedule", "--sampling-temperature-schedule"),
    ("uniform_mixture_epsilon_schedule", "--uniform-mixture-epsilon-schedule"),
    ("min_existing_log_distance", "--min-existing-log-distance"),
    ("min_existing_normalized_distance", "--min-existing-normalized-distance"),
    ("proposal_correction_mode", "--proposal-correction-mode"),
    ("proposal_correction_min_weight", "--proposal-correction-min-weight"),
    ("proposal_correction_max_weight", "--proposal-correction-max-weight"),
    ("early_stopping_patience", "--early-stopping-patience"),
    ("early_stopping_min_delta", "--early-stopping-min-delta"),
    ("python_exec", "--python-exec"),
]


@dataclass(frozen=True)
class LaunchPlan:
    recipe: Mapping[str, Any]
    resolved_paths: Mapping[str, str]
    command: list[str]
    pbs: Mapping[str, Any]
    pbs_path: Path
    manifest_path: Path
    generated_at: str
    submitted_job_id: str | None = None

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
    project_root: Path = ROOT,
    now: Callable[[], str] | None = None,
) -> LaunchPlan:
    project_root = Path(project_root).resolve()
    name = _required_string(recipe, "name")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("recipe name may only contain letters, numbers, '.', '-', and '_'")

    strategy = _required_string(recipe, "strategy")
    if strategy not in STRATEGY_PRESETS:
        known = ", ".join(sorted(STRATEGY_PRESETS))
        raise ValueError(f"unsupported strategy {strategy!r}; expected one of: {known}")
    preset = STRATEGY_PRESETS[strategy]

    resolved_paths: dict[str, str] = {"project_root": str(project_root)}
    display_paths: dict[str, str] = {}
    for key in ("target", "initial_summary", "template", "crqsf_root", "out_root"):
        value = _required_string(recipe, key)
        display_paths[key] = _display_path(value)
        resolved_paths[key] = str(_resolve_path(value, project_root))

    _validate_required_paths(resolved_paths)

    pbs = _pbs_config(recipe, project_root)
    resolved_paths["pbs_log"] = str(_resolve_path(str(pbs["log_path"]), project_root))
    pbs_path = _resolve_path(str(pbs.get("path", f"pbs_scripts/auto_{name}.pbs")), project_root)
    manifest_path = Path(resolved_paths["out_root"]) / "launch_manifest.json"

    start_iter = _int_value(recipe, preset, "start_iter", 0)
    max_iters = _int_value(recipe, preset, "max_iters", 1)
    num_samples = _int_value(recipe, preset, "num_samples", 1)
    proposal_candidates = _int_value(recipe, preset, "proposal_candidates", 32)
    theta_min = _float_value(recipe, preset, "theta_min", 0.3)
    theta_max = _float_value(recipe, preset, "theta_max", 5.0)
    crqsf_seed = _int_value(recipe, preset, "crqsf_seed", 36)
    beta = _float_value(recipe, preset, "beta", 0.5)
    ensemble_seeds = _seed_list(_value(recipe, preset, "crqsf_ensemble_seeds"))
    ensemble_size = _int_value(recipe, preset, "crqsf_ensemble_size", len(ensemble_seeds) if ensemble_seeds else 1)

    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if start_iter < 0:
        raise ValueError("start_iter must be non-negative")
    if start_iter >= max_iters:
        raise ValueError("start_iter must be smaller than max_iters")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if proposal_candidates < num_samples:
        raise ValueError("proposal_candidates must be >= num_samples")
    if theta_max <= theta_min:
        raise ValueError("theta_max must be greater than theta_min")
    if ensemble_seeds and len(ensemble_seeds) != ensemble_size:
        raise ValueError("crqsf_ensemble_seeds length must match crqsf_ensemble_size")
    if strategy.startswith("ensemble_") and not ensemble_seeds:
        raise ValueError(f"{strategy} requires crqsf_ensemble_seeds")

    command = ["python", ENGINE]
    _add_arg(command, "--parameter-set", _required_string(recipe, "parameter_set"))
    _add_arg(command, "--initial-summary", display_paths["initial_summary"])
    _add_arg(command, "--seed-label", _required_string(recipe, "seed_label"))
    _add_arg(command, "--out-root", display_paths["out_root"])
    _add_arg(command, "--crqsf-root", display_paths["crqsf_root"])
    _add_arg(command, "--target", display_paths["target"])
    _add_arg(command, "--loss-kind", _required_string(recipe, "loss_kind"))
    _add_arg(command, "--template", display_paths["template"])
    _add_arg(command, "--expected-mode", _required_string(recipe, "expected_mode"))
    _add_arg(command, "--start-iter", start_iter)
    _add_arg(command, "--max-iters", max_iters)
    _add_arg(command, "--num-samples", num_samples)
    _add_arg(command, "--proposal-candidates", proposal_candidates)
    _add_arg(command, "--crqsf-seed", crqsf_seed)
    if ensemble_size > 1:
        _add_arg(command, "--crqsf-ensemble-size", ensemble_size)
    if ensemble_seeds:
        _add_arg(command, "--crqsf-ensemble-seeds", ",".join(str(seed) for seed in ensemble_seeds))
    _add_arg(command, "--acquisition-mode", _value(recipe, preset, "acquisition_mode", "crqsf"))
    _add_arg(command, "--theta-min", theta_min)
    _add_arg(command, "--theta-max", theta_max)

    for key, flag in OPTIONAL_FLAG_FIELDS:
        _add_arg(command, flag, _value(recipe, preset, key))

    _add_arg(command, "--tune-preset", _required_string(recipe, "tune_preset"))
    _add_arg(command, "--beta", beta)
    if _value(recipe, preset, "require_initial_lhs", True) is False:
        command.append("--allow-non-lhs-initial")
    tune_once = _value(recipe, preset, "tune_once")
    if tune_once is True:
        command.append("--tune-once")
    elif tune_once is False:
        command.append("--retune-each-iter")

    return LaunchPlan(
        recipe=dict(recipe),
        resolved_paths=resolved_paths,
        command=command,
        pbs=pbs,
        pbs_path=pbs_path,
        manifest_path=manifest_path,
        generated_at=now() if now is not None else datetime.now(timezone.utc).isoformat(),
    )


def render_pbs(plan: LaunchPlan) -> str:
    pbs = plan.pbs
    log_path = plan.resolved_paths["pbs_log"]
    log_parent = _display_path(str(Path(str(pbs["log_path"])).parent))
    lines = [
        "#!/bin/bash",
        f"#PBS -N {pbs['job_name']}",
        f"#PBS -l {_select_resource(pbs)}",
    ]
    if pbs.get("place"):
        lines.append(f"#PBS -l place={pbs['place']}")
    lines.extend(
        [
            f"#PBS -l walltime={pbs['walltime']}",
            "#PBS -j oe",
            f"#PBS -o {log_path}",
            "",
            "set -euo pipefail",
            "",
            f"cd {plan.resolved_paths['project_root']}",
            f"mkdir -p {shlex.quote(log_parent)}",
            "",
            'echo "PBS_JOBID=${PBS_JOBID:-unknown}"',
            'echo "PBS_O_WORKDIR=${PBS_O_WORKDIR:-unknown}"',
            'echo "HOST=$(hostname)"',
            'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"',
            "",
            _render_multiline_command(plan.command),
            "",
        ]
    )
    return "\n".join(lines)


def execute_plan(
    plan: LaunchPlan,
    write_pbs: bool = False,
    submit: bool = False,
    force: bool = False,
    submit_runner: Callable[[Path], str] | None = None,
) -> LaunchPlan:
    if not write_pbs and not submit:
        return plan

    _ensure_writable(plan.pbs_path, force)
    _ensure_writable(plan.manifest_path, force)
    plan.pbs_path.parent.mkdir(parents=True, exist_ok=True)
    plan.pbs_path.write_text(render_pbs(plan))

    submitted_job_id = None
    if submit:
        runner = submit_runner or submit_pbs
        submitted_job_id = runner(plan.pbs_path)

    updated = replace(plan, submitted_job_id=submitted_job_id)
    updated.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    updated.manifest_path.write_text(json.dumps(_manifest_payload(updated), indent=2, sort_keys=True) + "\n")
    return updated


def submit_pbs(pbs_path: Path) -> str:
    result = subprocess.run(["qsub", str(pbs_path)], check=True, capture_output=True, text=True)
    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("qsub succeeded but did not return a job ID")
    return stdout.split()[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--write-pbs", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    recipe = load_recipe(args.recipe)
    plan = build_launch_plan(recipe, project_root=args.project_root)
    if not args.write_pbs and not args.submit:
        print("DRY RUN")
        print(f"PBS path: {plan.pbs_path}")
        print(f"Manifest path: {plan.manifest_path}")
        print(plan.command_text)
        return 0

    updated = execute_plan(plan, write_pbs=args.write_pbs, submit=args.submit, force=args.force)
    print(f"PBS path: {updated.pbs_path}")
    print(f"Manifest path: {updated.manifest_path}")
    if updated.submitted_job_id:
        print(f"Submitted job: {updated.submitted_job_id}")
    return 0


def _manifest_payload(plan: LaunchPlan) -> dict[str, Any]:
    return {
        "recipe": plan.recipe,
        "resolved_paths": dict(plan.resolved_paths),
        "command": plan.command,
        "command_text": plan.command_text,
        "pbs_path": str(plan.pbs_path),
        "manifest_path": str(plan.manifest_path),
        "generated_at": plan.generated_at,
        "submitted_job_id": plan.submitted_job_id,
    }


def _pbs_config(recipe: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    name = _required_string(recipe, "name")
    raw = dict(DEFAULT_PBS)
    raw.update(dict(recipe.get("pbs") or {}))
    raw.setdefault("job_name", name[:15])
    raw.setdefault("log_path", f"pbs_logs/{name}.$PBS_JOBID.log")
    raw.setdefault("select", 1)

    for key in ("select", "ncpus", "mpiprocs"):
        raw[key] = _positive_int(raw[key], f"pbs.{key}")
    if raw.get("ngpus") is not None:
        raw["ngpus"] = _positive_int(raw["ngpus"], "pbs.ngpus")
    nodetype = str(raw["nodetype"])
    if nodetype not in NODETYPE_HOSTS:
        known = ", ".join(sorted(NODETYPE_HOSTS))
        raise ValueError(f"pbs.nodetype must be one of: {known}")
    host = raw.get("host")
    if host:
        host = str(host)
        raw["host"] = host
        expected_hosts = NODETYPE_HOSTS[nodetype]
        if host not in expected_hosts:
            raise ValueError(f"pbs.host {host!r} is inconsistent with nodetype {nodetype}")
    if nodetype == "C":
        raw.setdefault("host", "node14")
        raw.setdefault("ngpus", 1)
    if not str(raw["job_name"]).strip():
        raise ValueError("pbs.job_name must be non-empty")
    if not str(raw["walltime"]).strip():
        raise ValueError("pbs.walltime must be non-empty")
    if not str(raw["log_path"]).strip():
        raise ValueError("pbs.log_path must be non-empty")
    raw["project_root"] = str(project_root)
    return raw


def _select_resource(pbs: Mapping[str, Any]) -> str:
    parts = [
        f"select={pbs['select']}",
        f"ncpus={pbs['ncpus']}",
        f"mpiprocs={pbs['mpiprocs']}",
        f"nodetype={pbs['nodetype']}",
    ]
    if pbs.get("host"):
        parts.append(f"host={pbs['host']}")
    if pbs.get("ngpus") is not None:
        parts.append(f"ngpus={pbs['ngpus']}")
    return ":".join(parts)


def _render_multiline_command(command: list[str]) -> str:
    if len(command) <= 2:
        return " ".join(shlex.quote(str(part)) for part in command)
    head = f"{shlex.quote(command[0])} {shlex.quote(command[1])}"
    segments: list[str] = []
    idx = 2
    while idx < len(command):
        current = command[idx]
        if current.startswith("--") and idx + 1 < len(command) and not command[idx + 1].startswith("--"):
            segments.append(f"{shlex.quote(current)} {shlex.quote(command[idx + 1])}")
            idx += 2
        else:
            segments.append(shlex.quote(current))
            idx += 1
    lines = [f"{head} \\"]
    for index, segment in enumerate(segments):
        suffix = " \\" if index < len(segments) - 1 else ""
        lines.append(f"  {segment}{suffix}")
    return "\n".join(lines)


def _validate_required_paths(resolved_paths: Mapping[str, str]) -> None:
    labels = {
        "target": "target",
        "initial_summary": "initial summary",
        "template": "template",
        "crqsf_root": "CRQSF root",
    }
    for key, label in labels.items():
        path = Path(resolved_paths[key])
        if not path.exists():
            raise ValueError(f"missing {label}: {path}")


def _value(recipe: Mapping[str, Any], preset: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in recipe:
        return recipe[key]
    controls = recipe.get("controls") or {}
    if key in controls:
        return controls[key]
    return preset.get(key, default)


def _required_string(recipe: Mapping[str, Any], key: str) -> str:
    value = recipe.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"recipe field {key!r} is required")
    return str(value)


def _int_value(recipe: Mapping[str, Any], preset: Mapping[str, Any], key: str, default: int) -> int:
    return int(_value(recipe, preset, key, default))


def _float_value(recipe: Mapping[str, Any], preset: Mapping[str, Any], key: str, default: float) -> float:
    return float(_value(recipe, preset, key, default))


def _positive_int(value: Any, label: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _seed_list(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value)
    return [int(part) for part in parts]


def _add_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def _display_path(value: str) -> str:
    path = Path(value).expanduser()
    return str(path)


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
