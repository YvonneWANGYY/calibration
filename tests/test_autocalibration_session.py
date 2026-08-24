import builtins
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class AdaptiveCalibrationSessionTests(unittest.TestCase):
    def test_recommend_without_metrics_returns_lhs_initial_runs(self):
        from autocalibration import AdaptiveCalibrationSession, ParameterSpec

        session = AdaptiveCalibrationSession.from_inputs(
            parameters=[
                ParameterSpec("alpha", 0.0, 1.0),
                ParameterSpec("beta", 10.0, 20.0),
            ],
            target_profile={"stellar_mass": 1.0},
        )

        recommendations = session.recommend(n=5, seed=11)

        self.assertEqual(len(recommendations), 5)
        self.assertTrue(all(item["kind"] == "initial_lhs" for item in recommendations))
        for item in recommendations:
            self.assertGreaterEqual(item["parameters"]["alpha"], 0.0)
            self.assertLessEqual(item["parameters"]["alpha"], 1.0)
            self.assertGreaterEqual(item["parameters"]["beta"], 10.0)
            self.assertLessEqual(item["parameters"]["beta"], 20.0)

        alpha_strata = sorted(int(item["parameters"]["alpha"] * 5) for item in recommendations)
        beta_strata = sorted(int(((item["parameters"]["beta"] - 10.0) / 10.0) * 5) for item in recommendations)
        self.assertEqual(alpha_strata, [0, 1, 2, 3, 4])
        self.assertEqual(beta_strata, [0, 1, 2, 3, 4])

    def test_loads_target_and_initial_metrics_from_csv_files(self):
        from autocalibration import AdaptiveCalibrationSession, ParameterSpec

        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            target_path = workdir / "target_profile.json"
            metrics_path = workdir / "initial_metrics.csv"
            target_path.write_text(json.dumps({"stellar_mass": 1.0, "gas_fraction": 0.3}))
            metrics_path.write_text(
                "alpha,beta,stellar_mass,gas_fraction\n"
                "0.2,1.0,0.8,0.25\n"
                "0.6,2.5,1.2,0.35\n"
                "0.9,4.0,1.7,0.80\n"
            )

            session = AdaptiveCalibrationSession.from_files(
                parameters=[
                    ParameterSpec("alpha", 0.0, 1.0),
                    ParameterSpec("beta", 0.5, 5.0),
                ],
                target_profile=target_path,
                metrics_file=metrics_path,
                metric_columns=["stellar_mass", "gas_fraction"],
            )

            self.assertEqual(len(session.observations), 3)
            self.assertAlmostEqual(session.losses()[0], 0.0425)

            recommendations = session.recommend(n=2, seed=5)

        self.assertEqual(len(recommendations), 2)
        self.assertTrue(all(item["kind"] == "adaptive" for item in recommendations))
        for item in recommendations:
            self.assertIn("score", item)
            self.assertIn("nearest_loss", item)
            self.assertGreaterEqual(item["parameters"]["alpha"], 0.0)
            self.assertLessEqual(item["parameters"]["alpha"], 1.0)
            self.assertGreaterEqual(item["parameters"]["beta"], 0.5)
            self.assertLessEqual(item["parameters"]["beta"], 5.0)

    def test_can_continue_session_with_json_observation_file(self):
        from autocalibration import AdaptiveCalibrationSession, ParameterSpec

        with tempfile.TemporaryDirectory() as tmpdir:
            observations_path = Path(tmpdir) / "new_metrics.json"
            observations_path.write_text(
                json.dumps(
                    [
                        {
                            "parameters": {"alpha": 0.3, "beta": 1.5},
                            "metrics": {"stellar_mass": 0.9},
                        },
                        {
                            "parameters": {"alpha": 0.8, "beta": 3.5},
                            "metrics": {"stellar_mass": 1.4},
                        },
                    ]
                )
            )
            session = AdaptiveCalibrationSession.from_inputs(
                parameters=[
                    ParameterSpec("alpha", 0.0, 1.0),
                    ParameterSpec("beta", 0.5, 5.0),
                ],
                target_profile={"stellar_mass": 1.0},
            )

            session.add_observations_from_file(observations_path)
            recommendations = session.recommend(n=1, seed=13)

        self.assertEqual(len(session.observations), 2)
        self.assertEqual(recommendations[0]["kind"], "adaptive")

    def test_hdf5_loader_reports_optional_dependency(self):
        from autocalibration import AdaptiveCalibrationSession, ParameterSpec

        session = AdaptiveCalibrationSession.from_inputs(
            parameters=[ParameterSpec("alpha", 0.0, 1.0)],
            target_profile={"stellar_mass": 1.0},
        )
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "h5py":
                raise ModuleNotFoundError("No module named 'h5py'")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            hdf5_path = Path(tmpdir) / "metrics.h5"
            hdf5_path.write_bytes(b"")
            builtins.__import__ = fake_import
            try:
                with self.assertRaisesRegex(RuntimeError, "pip install h5py"):
                    session.add_observations_from_file(
                        hdf5_path,
                        parameter_paths={"alpha": "/parameters/alpha"},
                        metric_paths={"stellar_mass": "/metrics/stellar_mass"},
                    )
            finally:
                builtins.__import__ = real_import


if __name__ == "__main__":
    unittest.main()
