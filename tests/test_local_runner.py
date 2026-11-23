import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from autocalibration.cli import run


class LocalRunnerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "problem.json"
        self.wrapper = self.root / "model.py"
        self.wrapper.write_text(
            "import json, sys\nx = json.load(sys.stdin)['x']\n"
            "print(json.dumps({'response': 2 * x}))\n"
        )
        self.path.write_text(json.dumps({
            "parameters": [{"name": "x", "lower": 1, "upper": 10, "scale": "log"}],
            "target": {"response": 5}, "simulator": ["{python}", "model.py"],
            "initial_samples": 3, "adaptive_steps": 2, "candidate_pool_size": 9, "seed": 3,
        }))

    def test_custom_problem_replays_and_reports_best_measured_loss(self):
        first = self.root / "first"
        result = run(self.path, first)
        second = self.root / "second"
        run(self.path, second)
        self.assertEqual((first / "evaluations.jsonl").read_bytes(),
                         (second / "evaluations.jsonl").read_bytes())
        rows = [json.loads(line) for line in (first / "evaluations.jsonl").read_text().splitlines()]
        self.assertEqual([row["kind"] for row in rows], ["initial_lhs"] * 3 + ["adaptive"] * 2)
        for row in rows:
            self.assertAlmostEqual(row["loss"], (2 * row["parameters"]["x"] - 5) ** 2)
        self.assertEqual(result["best_observed"], min(rows, key=lambda row: row["loss"]))
        with self.assertRaises(FileExistsError):
            run(self.path, first)

    def test_failed_simulator_preserves_completed_rows(self):
        self.wrapper.write_text(
            "from pathlib import Path\n"
            "marker = Path('started')\n"
            "if marker.exists(): raise SystemExit(3)\n"
            "marker.touch()\nimport json\nprint(json.dumps({'response': 4}))\n"
        )
        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "evaluation 2 failed"):
            run(self.path, output)
        self.assertEqual(len((output / "evaluations.jsonl").read_text().splitlines()), 1)
        self.assertFalse((output / "summary.json").exists())
