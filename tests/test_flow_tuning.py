import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ("optuna", "torch")),
                     "install the tuning extra to test the flow runner")
class FlowTuningTests(unittest.TestCase):
    def test_schedules_with_generated_and_existing_heat_simulations(self):
        for schedule in ("initial", "every_iteration"):
            with self.subTest(schedule=schedule), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                shutil.copytree(ROOT / "guide/heat", root / "heat")
                spec = root / "heat/problem.json"
                problem = json.loads(spec.read_text())
                problem.update(method="neural_spline_flow", adaptive_steps=2, candidate_pool_size=16,
                               tuning=dict(schedule=schedule, epochs=1, members=2, folds=2, trials=2))
                if schedule == "every_iteration":
                    del problem["initial_samples"]
                    shutil.copyfile(ROOT / "guide/initial.csv", spec.parent / "initial.csv")
                    problem["initial_data"] = "initial.csv"
                else:
                    problem["initial_samples"] = 4
                spec.write_text(json.dumps(problem))
                out = root / "run"
                env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1")
                subprocess.run([sys.executable, "-m", "autocalibration.cli", "run", str(spec),
                                "--output", str(out)], env=env, check=True, capture_output=True)
                def rows(file):
                    return [json.loads(v) for v in (out / file).read_text().splitlines()]
                fits = rows("flow_history.jsonl")
                measured = rows("evaluations.jsonl")
                self.assertEqual([len(f["trials"]) for f in fits],
                                 [2, 0, 0] if schedule == "initial" else [2, 2, 0])
                for i, f in enumerate(fits):
                    self.assertEqual(f["observations"], len(measured) - 3 + i)
                    if f["trials"]:
                        best = min(f["trials"], key=lambda t: t["validation_nll"])
                        self.assertEqual(f["parameters"], best["parameters"])
                    else:
                        self.assertEqual(f["parameters"], fits[i-1]["parameters"])
                self.assertEqual([r["kind"] for r in measured[-3:]],
                                 ["adaptive", "adaptive", "final_recommendation"])
                result = json.loads((out / "summary.json").read_text())
                self.assertEqual(result["final_recommendation"], measured[-1])
                self.assertEqual(result["best_observed"]["loss"], min(r["loss"] for r in measured))
                final = measured[-1]
                check = subprocess.check_output([sys.executable, str(spec.parent / "simulate.py")],
                                                input=json.dumps(final["parameters"]), text=True)
                self.assertEqual(final["metrics"], json.loads(check))

    def test_single_log_parameter_and_invalid_fold_count(self):
        sys.path.insert(0, str(ROOT / "src"))
        from autocalibration.cli import run
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = root / "problem.json"
            problem = dict(parameters=[dict(name="mass", lower=0.1, upper=10, scale="log")],
                           target={"response": 1.0}, method="neural_spline_flow", adaptive_steps=1,
                           initial_samples=3, candidate_pool_size=8, seed=0,
                           simulator=["{python}", "-c", "import json,sys; print(json.dumps("
                                      "{'response':json.load(sys.stdin)['mass']}))"],
                           tuning=dict(trials=1, folds=4, epochs=1, members=1))
            spec.write_text(json.dumps(problem))
            with self.assertRaisesRegex(ValueError, "folds"):
                run(spec, root / "invalid")
            self.assertFalse((root / "invalid").exists())
            problem["tuning"]["folds"] = 2
            spec.write_text(json.dumps(problem))
            result = run(spec, root / "valid")
            self.assertEqual(result["evaluations"], 5)
            self.assertEqual(result["final_recommendation"]["parameters"]["mass"],
                             result["final_recommendation"]["metrics"]["response"])
