import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(all(importlib.util.find_spec(k) for k in ("torch", "sklearn", "optuna")),
                     "requires the tuning extra")
class CostSelectionTests(unittest.TestCase):
    def test_cost_ranking_and_imports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads((ROOT / "guide/heat/cost-aware.json").read_text())
            config["simulator"][1] = str(ROOT / "guide/heat/simulate.py")
            config.update(initial_samples=4, adaptive_steps=2, candidate_pool_size=64)
            config["tuning"].update(trials=1, folds=2, epochs=1, members=2)

            def run(name):
                path = root / "problem.json"
                path.write_text(json.dumps(config))
                out = root / name
                subprocess.run([sys.executable, "-m", "autocalibration.cli", "run", str(path),
                                "--output", str(out)], check=True, capture_output=True,
                               env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1"))
                def rows(filename):
                    return [json.loads(s) for s in (out / filename).read_text().splitlines()]
                return rows("evaluations.jsonl"), rows("flow_history.jsonl")

            config["simulator"][-1] = "--report-cost"
            config["cost"] = {"metric": "solver_steps", "beta": 0}
            zero, _ = run("zero")
            cost = config.pop("cost")
            ordinary, _ = run("ordinary")
            self.assertEqual(zero, ordinary)
            config["cost"] = dict(cost, beta=100)
            expensive, fits = run("penalty")
            for i, fit in enumerate(fits[:-1], 4):
                current = min(expensive[:i], key=lambda row: row["loss"])
                acq = fit["acquisition"]
                self.assertEqual(acq["reference_cost"], current["metrics"]["solver_steps"])
                self.assertAlmostEqual(acq["selection_score"],
                                       acq["base_score"] - 100 * acq["predicted_log_cost_ratio"])
            self.assertNotIn("acquisition", fits[-1])
            for row in expensive:
                expected = sum((row["metrics"][k] - target)**2 for k,target in config["target"].items())
                self.assertAlmostEqual(row["loss"], expected)
            # Reuse the same design with user-assigned scores, including zero
            # and negatives. Newly evaluated rows carry the same field.
            del config["cost"]
            config["simulator"][-1] = "--scalar-cost"
            scalar, scalar_fits = run("scalar")
            self.assertIn("predicted_scaled_cost", scalar_fits[0]["acquisition"])
            import csv
            keys = list(expensive[0]["parameters"]) + list(config["target"]) + ["solver_steps"]
            with (root / "initial.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, keys)
                writer.writeheader()
                for row in expensive[:4]:
                    writer.writerow(row["parameters"] | row["metrics"])
            del config["initial_samples"]
            config["initial_data"] = "initial.csv"
            config["cost"] = dict(cost, beta=100)
            config["simulator"][-1] = "--report-cost"
            imported, _ = run("imported")
            self.assertEqual([r["parameters"] for r in expensive], [r["parameters"] for r in imported])

            del config["cost"]
            config["simulator"][-1] = "--scalar-cost"
            with (root / "initial.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, keys[:-1] + ["cost"])
                writer.writeheader()
                for i, row in enumerate(scalar[:4]):
                    writer.writerow(row["parameters"] | row["metrics"] | {"cost": i - 2})
            imported_scalar, audit = run("imported-scalar")
            self.assertEqual([r["metrics"]["cost"] for r in imported_scalar[:4]], [-2,-1,0,1])
            self.assertIn("predicted_scaled_cost", audit[0]["acquisition"])

    def test_cost_changes_winner_of_a_fixed_pool(self):
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        from autocalibration import AdaptiveCalibrationSession, ParameterSpec
        from autocalibration.flow_search import select

        class TwoCandidates:
            def sample(self, count, context):
                return torch.tensor([[-2.0], [2.0]])

            def log_prob(self, points, context):
                return points[:, 0]

        session = AdaptiveCalibrationSession([ParameterSpec("x", 0, 1)], {"response": 1})
        for i in range(20):
            x = i / 20
            session.add_observation({"x": x}, {"response": x, "work": 1 if x < 0.5 else 100})
        models = [(TwoCandidates(), torch.tensor([0.]), torch.tensor([1.]))]
        args = (models, torch.tensor([[1.]]), session, 2, 0)
        high, _ = select(*args, False, None, 42)
        low, audit = select(*args, False, {"metric": "work", "beta": 1}, 42)
        self.assertGreater(high["x"], low["x"])
        self.assertEqual(audit["reference_cost"], 100)
        final, _ = select(*args, True, {"metric": "work", "beta": 100}, 42)
        self.assertEqual(final, high)
        scalar = {"metric": "score", "beta": 10, "scale": "range"}
        for obs in session.observations:
            obs.metrics["score"] = -3 if obs.parameters["x"] < 0.5 else 0
        cheaper, audit = select(*args, False, scalar, 42)
        self.assertEqual(cheaper, low)
        for obs in session.observations:
            obs.metrics["score"] = -3
        tied, audit = select(*args, False, scalar, 42)
        self.assertEqual(tied, high)
        self.assertEqual(audit["predicted_scaled_cost"], 0)
