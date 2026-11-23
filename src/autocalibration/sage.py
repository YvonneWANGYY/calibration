"""Replay SAGE model selection using the paper's complete-bin loss."""

import json
import math
from collections import defaultdict
from statistics import fmean


def read_json(path):
    return json.loads(path.read_text())


class ObservableLoss:
    """Mean of within-observable MSEs, using fixed target ranges."""

    def __init__(self, target, families=None):
        groups = defaultdict(dict)
        seen = set()
        for row in target["target_observational_metrics"]:
            family = row["observable"]
            if families is not None and family not in families:
                continue
            key, value = row["metric_id"], float(row["value"])
            if key in seen or not math.isfinite(value):
                raise ValueError(f"invalid or duplicate target bin {key}")
            seen.add(key)
            groups[family][key] = value
        if not groups or (families is not None and set(groups) != set(families)):
            raise ValueError("requested observable missing from target")
        self.groups = []
        for family, values in groups.items():
            span = max(values.values()) - min(values.values())
            if span <= 0:
                raise ValueError(f"constant target range for {family}")
            self.groups.append((values, span))

    def __call__(self, metrics):
        losses = []
        for values, span in self.groups:
            errors = []
            for key, observed in values.items():
                predicted = metrics.get(key)
                if predicted is None or not math.isfinite(predicted):
                    return math.inf
                errors.append((2 * (predicted - observed) / span) ** 2)
            losses.append(fmean(errors))
        return fmean(losses)


def replay(directory):
    study = read_json(directory / "study.json")
    profiles = read_json(directory / "display_metrics.json")
    target = read_json(directory / "comparison_target.json")
    five = ObservableLoss(target, study["five_families"])
    eight = ObservableLoss(target)

    def result(label, run_id, metrics):
        return {"model": label, "run_id": run_id, "L5": five(metrics), "L8": eight(metrics)}

    yield result("Fiducial model", "fiducial", profiles["Fiducial model"])
    for case in study["cases"]:
        folder = directory / case["directory"]
        score = ObservableLoss(read_json(folder / "target.json"))
        records = [json.loads(line) for line in (folder / "evaluations.jsonl").read_text().splitlines()]
        adaptive = [r for r in records if r["stage"].startswith("iter_")]
        final = [r for r in records if r["stage"] == "final"]
        if len(adaptive) != 25 or len(final) != 1 or len(records) != 51:
            raise ValueError(f"incomplete evaluation history in {folder}")
        candidates = {"adaptive": adaptive, "final": final, "adaptive_and_final": adaptive + final}
        winner = min(candidates[case["rule"]], key=lambda r: score(r["metrics"]))
        if not math.isfinite(score(winner["metrics"])) or winner["run_id"] != case["expected_model"]:
            raise ValueError(f"model selection disagrees with the manuscript for {folder}")
        profile = profiles[case["series"]]
        if not math.isclose(score(winner["metrics"]), score(profile), abs_tol=1e-9):
            raise ValueError(f"selected simulation and plotted profile disagree for {folder}")
        yield result(case["series"], winner["run_id"], profile)
