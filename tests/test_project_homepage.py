import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectHomepageTests(unittest.TestCase):
    def setUp(self):
        self.index_path = ROOT / "index.html"
        self.readme_path = ROOT / "README.md"
        self.profile_path = ROOT / "profile" / "README-project-card.md"

    def test_reader_guide_uses_the_released_flow_workflow(self):
        text = self.index_path.read_text()
        sections = re.findall(r'<section id="([^"]+)"', text)
        self.assertEqual(sections, ["introduction", "reproduce", "your-problem", "pipeline"])
        self.assertIn("@v0.3.0", text)
        self.assertNotIn("@v0.2.0", text)
        for term in ("autocalibration reproduce-sage", "best_observed", "final_recommendation",
                     "published manual calibration", "Croton et al.", "sage-home/sage-model",
                     "0.104723", "0.128536", "18.5%", "every_iteration"):
            self.assertIn(term, text)

    def test_heat_download_matches_the_commands_and_released_simulator(self):
        import json
        from zipfile import ZipFile
        text = self.index_path.read_text()
        with ZipFile(ROOT / "guide/heat-example.zip") as archive:
            def read(name):
                return archive.read("heat-example/" + name)
            generated = json.loads(read("new-design.json"))
            imported = json.loads(read("existing-runs.json"))
            self.assertEqual(generated, json.loads((ROOT / "guide/heat/flow-problem.json").read_text()))
            self.assertEqual(read("simulate.py"), (ROOT / "guide/heat/simulate.py").read_bytes())
            self.assertEqual(read(imported.pop("initial_data")), (ROOT / "guide/initial.csv").read_bytes())
            self.assertEqual(generated.pop("initial_samples"), 8)
            self.assertEqual(imported, generated)
            for file in ("new-design.json", "existing-runs.json"):
                self.assertIn("autocalibration run " + file, text)

    def test_guide_links_and_figures_resolve(self):
        text = self.index_path.read_text()
        ids = re.findall(r' id="([^"]+)"', text)
        self.assertEqual(len(ids), len(set(ids)))
        for fragment in re.findall(r'href="#([^"]+)"', text):
            self.assertIn(fragment, ids)
        for path in re.findall(r'(?:href|src)="((?:guide|assets)/[^"?]+)', text):
            self.assertTrue((ROOT / path).is_file(), path)
        self.assertIn("assets/calibration-workflow.png", text)

    def test_profile_card_is_short_and_links_to_full_project(self):
        text = self.profile_path.read_text()

        self.assertIn("Fully Autonomous Calibration for Astrophysics", text)
        self.assertIn("https://YvonneWANGYY.github.io/calibration/", text)
        self.assertIn("https://github.com/YvonneWANGYY/calibration", text)
        self.assertIn("Paper in prep", text)
        self.assertNotIn("new_structured_calibration.pdf", text)
        self.assertIn("Generic interface", text)
        self.assertIn("SAGE example", text)
        self.assertLess(len(text.splitlines()), 45)

    def test_readme_is_a_public_project_landing_page(self):
        text = self.readme_path.read_text()
        normalized_text = re.sub(r"\s+", " ", text)

        self.assertIn("Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration", text)
        self.assertIn("https://YvonneWANGYY.github.io/calibration/", text)
        self.assertIn("Paper in prep", text)
        self.assertNotIn("assets/new_structured_calibration.pdf", text)
        self.assertIn("supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py", text)
        self.assertIn("supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py", text)
        self.assertIn("src/autocalibration", text)
        self.assertIn("notebooks/sage_like_recommendation_loop.ipynb", text)
        self.assertIn(
            "The generic recipe and launcher are the primary public interface; SAGE is the included worked example.",
            normalized_text,
        )
        self.assertIn("Autonomy boundary", text)
        self.assertNotIn("fully-autonomous-calibration", text)

    def test_library_and_notebook_examples_are_publicly_linked(self):
        readme_text = self.readme_path.read_text()

        self.assertIn("examples/sage_like_initial_metrics.csv", readme_text)
        self.assertIn("examples/sage_like_target_profile.json", readme_text)

    def test_scheduler_wrapping_is_left_to_users(self):
        launcher = ROOT / "supplement" / "sage_recipe_launcher" / "scripts" / "launch_sage_calibration.py"
        recipe = ROOT / "supplement" / "sage_recipe_launcher" / "recipes" / "sage8_ensemble_map_20iter.json"
        readme_text = self.readme_path.read_text()

        launcher_text = launcher.read_text()
        recipe_text = recipe.read_text()

        self.assertNotIn("DEFAULT_PBS", launcher_text)
        self.assertNotIn("NODETYPE_HOSTS", launcher_text)
        self.assertNotIn("--write-pbs", launcher_text)
        self.assertNotIn("--submit", launcher_text)
        self.assertNotIn("qsub", launcher_text)
        self.assertIn('"user_edit_notes"', recipe_text)
        self.assertIn('"edit_before_running"', recipe_text)
        self.assertIn("Keep scheduler submission outside this JSON", recipe_text)
        self.assertNotIn('"pbs"', recipe_text)
        self.assertNotIn('"nodetype"', recipe_text)
        self.assertNotIn('"ngpus"', recipe_text)
        self.assertIn("Edit before running", readme_text)
        self.assertIn("write your own scheduler wrapper", readme_text)

    def test_public_sage_recipe_uses_path_placeholders(self):
        recipe = ROOT / "supplement" / "sage_recipe_launcher" / "recipes" / "sage8_ensemble_map_20iter.json"
        recipe_text = recipe.read_text()

        private_root_pattern = re.compile(r"/(?:data|home|Users|scratch|tmp)/[A-Za-z0-9_.-]+")
        self.assertIsNone(private_root_pattern.search(recipe_text))
        self.assertNotIn("active_learning_runs" + "_", recipe_text)
        self.assertNotIn("lhs_" + "sagepaper8", recipe_text)
        self.assertNotRegex(recipe_text, r"seed20\d{6}")
        self.assertIn("<your_target>.json", recipe_text)
        self.assertIn("<your_initial_lhs_summary>.csv", recipe_text)
        self.assertIn("<your_crqsf_root>", recipe_text)
        self.assertIn("<your_output_root>", recipe_text)
        self.assertIn("<your_run_label>", recipe_text)

    def test_sage_launcher_supplement_is_included(self):
        launcher = ROOT / "supplement" / "sage_recipe_launcher" / "scripts" / "launch_sage_calibration.py"
        recipe = ROOT / "supplement" / "sage_recipe_launcher" / "recipes" / "sage8_ensemble_map_20iter.json"

        self.assertTrue(launcher.exists(), f"missing SAGE launcher supplement: {launcher}")
        self.assertTrue(recipe.exists(), f"missing SAGE recipe supplement: {recipe}")
        self.assertIn("STRATEGY_PRESETS", launcher.read_text())
        self.assertIn('"strategy": "ensemble_map"', recipe.read_text())

    def test_generic_recipe_launcher_supplement_is_included(self):
        launcher = ROOT / "supplement" / "generic_recipe_launcher" / "scripts" / "launch_adaptive_calibration.py"
        recipe = ROOT / "supplement" / "generic_recipe_launcher" / "recipes" / "adaptive_calibration_template.json"

        self.assertTrue(launcher.exists(), f"missing generic launcher supplement: {launcher}")
        self.assertTrue(recipe.exists(), f"missing generic recipe supplement: {recipe}")

        launcher_text = launcher.read_text()
        recipe_text = recipe.read_text()
        private_root_pattern = re.compile(r"/(?:data|home|Users|scratch|tmp)/[A-Za-z0-9_.-]+")

        self.assertIn("def build_launch_plan", launcher_text)
        self.assertIn("--write-manifest", launcher_text)
        self.assertNotIn("qsub", launcher_text)
        self.assertNotIn("DEFAULT_PBS", launcher_text)
        self.assertNotIn("SAGE", launcher_text)
        self.assertIsNone(private_root_pattern.search(launcher_text))

        self.assertIn('"parameter_space"', recipe_text)
        self.assertIn('"simulator"', recipe_text)
        self.assertIn('"proposal_model"', recipe_text)
        self.assertIn('"scheduler"', recipe_text)
        self.assertIn("<your_adaptive_calibration_engine>.py", recipe_text)
        self.assertIn("<parameter_1>", recipe_text)
        self.assertNotIn("SAGE", recipe_text)
        self.assertNotIn('"pbs"', recipe_text)
        self.assertNotIn('"nodetype"', recipe_text)
        self.assertNotIn('"ngpus"', recipe_text)
        self.assertIsNone(private_root_pattern.search(recipe_text))


if __name__ == "__main__":
    unittest.main()
