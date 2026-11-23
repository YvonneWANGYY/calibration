"""Train the fixed-baryon SMF50 ensemble on its initial design and rank the first pool."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def propose(output, smoke=False):
    if importlib.util.find_spec("torch") is None:
        raise ValueError("propose-sage requires PyTorch. Install the package with the flow extra.")
    root = Path(__file__).resolve().parent
    case = root / "smf050-fixed"
    options = json.loads((case / "first_round.json").read_text())
    target = json.loads((case / "target.json").read_text())["target_observational_metrics"]
    records = [json.loads(row) for row in (case / "evaluations.jsonl").read_text().splitlines()]
    initial = [r for r in records if r["stage"] == "initial"]
    if len(initial) != 25:
        raise ValueError("expected the 25 original initial evaluations")
    output.mkdir(parents=True)
    (output / options["--metrics_file"]).write_text("".join(json.dumps(r) + "\n" for r in initial))
    values = {r["metric_id"]: r["value"] for r in target}
    (output / options["--target_metrics_json"]).write_text(json.dumps(values))
    if smoke:
        options.update({"--epochs": "1", "--candidate_pool_size": "64", "--num_samples": "1"})
    command = [sys.executable, str(root / "flow/train_crqsf.py")]
    for key, value in options.items():
        command.append(key)
        if value is not True:
            command.append(value)
    env = dict(os.environ, SAGE_TARGET_CONDITION_CONFIG=str(case / "condition.json"))
    env.setdefault("OMP_NUM_THREADS", "1")
    subprocess.run(command, cwd=output, env=env, check=True)
