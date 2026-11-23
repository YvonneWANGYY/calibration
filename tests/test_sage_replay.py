import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocalibration.sage import ObservableLoss, replay


class SageReplayTests(unittest.TestCase):
    def test_current_paper_table(self):
        rows = list(replay(ROOT / "src/autocalibration/paper_sage"))
        expected = [(0.090493, 0.128536), (0.058346, 0.157553), (0.055785, 0.134744),
                    (0.047770, 0.144398), (0.043655, 0.134188), (0.082809, 0.104723),
                    (0.094088, 0.100687)]
        self.assertEqual([(round(r["L5"], 6), round(r["L8"], 6)) for r in rows], expected)

    def test_full_coverage_required(self):
        target = {"target_observational_metrics": [
            {"metric_id": key, "observable": "a", "value": value}
            for key, value in [("x", 1), ("y", 3)]
        ]}
        loss = ObservableLoss(target)
        self.assertEqual(loss({"x": 1, "y": 5}), 2)
        self.assertTrue(math.isinf(loss({"x": 1})))
