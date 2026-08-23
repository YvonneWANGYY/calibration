import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectHomepageTests(unittest.TestCase):
    def setUp(self):
        self.index_path = ROOT / "index.html"
        self.readme_path = ROOT / "README.md"
        self.profile_path = ROOT / "profile" / "README-project-card.md"

    def test_project_page_has_required_research_claims(self):
        text = self.index_path.read_text()

        self.assertIn(
            "Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration",
            text,
        )
        self.assertIn("SAGE recipe-driven launcher", text)
        self.assertIn("researcher-defined inputs", text)
        self.assertIn("Sedov--Taylor", text)
        self.assertIn("Rotating Cylinder", text)
        self.assertIn("local SAGE", text)

    def test_project_page_links_to_reproducibility_artifacts(self):
        text = self.index_path.read_text()

        self.assertIn("new_structured_calibration.pdf", text)
        self.assertIn("https://github.com/YvonneWANGYY/calibration", text)
        self.assertIn("supplement/sage_recipe_launcher/recipes/sage8_ensemble_map_20iter.json", text)
        self.assertIn("supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py", text)
        self.assertIn("python scripts/launch_sage_calibration.py --recipe recipes/sage8_ensemble_map_20iter.json", text)
        self.assertIn("supplement/generic_recipe_launcher/recipes/adaptive_calibration_template.json", text)
        self.assertIn("supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py", text)
        self.assertIn("python scripts/launch_adaptive_calibration.py --recipe recipes/adaptive_calibration_template.json", text)
        self.assertNotIn("--write-pbs", text)
        self.assertNotIn("fully-autonomous-calibration", text)

    def test_project_page_presents_generic_launcher_as_primary_interface(self):
        text = self.index_path.read_text()

        self.assertIn("Generic recipe launcher", text)
        self.assertIn("Worked example", text)
        self.assertIn(
            "SAGE is the included example implementation of the generic recipe interface",
            text,
        )
        self.assertLess(
            text.index("Generic recipe launcher"),
            text.index("SAGE recipe-driven launcher"),
        )
        self.assertNotIn("SAGE automation highlight", text)

    def test_reproducibility_links_use_short_resource_cards(self):
        text = self.index_path.read_text()

        self.assertIn('class="resource-grid"', text)
        self.assertIn('class="resource-card"', text)
        self.assertIn('<span class="resource-title">Paper PDF</span>', text)
        self.assertIn('<span class="resource-title">SAGE recipe</span>', text)
        self.assertIn('<span class="resource-title">SAGE launcher</span>', text)
        self.assertIn('<span class="resource-title">Generic recipe</span>', text)
        self.assertIn('<span class="resource-title">Generic launcher</span>', text)
        self.assertNotIn('>supplement/sage_recipe_launcher', text)
        self.assertNotIn('>supplement/generic_recipe_launcher', text)
        self.assertNotIn('>Download new_structured_calibration.pdf<', text)

    def test_project_page_uses_existing_visual_asset(self):
        text = self.index_path.read_text()
        asset = ROOT / "assets" / "sage8-direct-map-trajectory.png"

        self.assertTrue(asset.exists(), f"missing homepage visual asset: {asset}")
        self.assertIn("assets/sage8-direct-map-trajectory.png", text)
        self.assertRegex(text, r"<img[^>]+alt=\"SAGE active-learning trajectory")

    def test_project_page_keeps_autonomy_boundary_honest(self):
        text = re.sub(r"\s+", " ", self.index_path.read_text())

        self.assertIn(
            "Autonomous means the closed-loop run proceeds after the recipe is specified",
            text,
        )
        self.assertIn(
            "It does not mean the system invents the science target, parameter domain, simulator wrapper, PBS resource policy, or initial design",
            text,
        )

    def test_profile_card_is_short_and_links_to_full_project(self):
        text = self.profile_path.read_text()

        self.assertIn("Fully Autonomous Calibration for Astrophysics", text)
        self.assertIn("https://YvonneWANGYY.github.io/calibration/", text)
        self.assertIn("https://github.com/YvonneWANGYY/calibration", text)
        self.assertIn("new_structured_calibration.pdf", text)
        self.assertIn("Generic interface", text)
        self.assertIn("SAGE example", text)
        self.assertLess(len(text.splitlines()), 45)

    def test_readme_is_a_public_project_landing_page(self):
        text = self.readme_path.read_text()
        normalized_text = re.sub(r"\s+", " ", text)

        self.assertIn("Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration", text)
        self.assertIn("https://YvonneWANGYY.github.io/calibration/", text)
        self.assertIn("assets/new_structured_calibration.pdf", text)
        self.assertIn("supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py", text)
        self.assertIn("supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py", text)
        self.assertIn(
            "The generic recipe and launcher are the primary public interface; SAGE is the included worked example.",
            normalized_text,
        )
        self.assertIn("Autonomy boundary", text)
        self.assertNotIn("fully-autonomous-calibration", text)

    def test_scheduler_wrapping_is_left_to_users(self):
        launcher = ROOT / "supplement" / "sage_recipe_launcher" / "scripts" / "launch_sage_calibration.py"
        recipe = ROOT / "supplement" / "sage_recipe_launcher" / "recipes" / "sage8_ensemble_map_20iter.json"
        index_text = self.index_path.read_text()
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
        self.assertIn("Edit before running", index_text)
        self.assertIn("write your own scheduler wrapper", index_text)
        self.assertNotIn("JSON-to-PBS", index_text)
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
