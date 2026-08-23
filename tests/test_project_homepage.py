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
        self.assertNotIn("--write-pbs", text)
        self.assertNotIn("fully-autonomous-calibration", text)

    def test_reproducibility_links_use_short_resource_cards(self):
        text = self.index_path.read_text()

        self.assertIn('class="resource-grid"', text)
        self.assertIn('class="resource-card"', text)
        self.assertIn('<span class="resource-title">Paper PDF</span>', text)
        self.assertIn('<span class="resource-title">SAGE recipe</span>', text)
        self.assertIn('<span class="resource-title">SAGE launcher</span>', text)
        self.assertNotIn('>supplement/sage_recipe_launcher', text)
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
        self.assertLess(len(text.splitlines()), 45)

    def test_readme_is_a_public_project_landing_page(self):
        text = self.readme_path.read_text()

        self.assertIn("Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration", text)
        self.assertIn("https://YvonneWANGYY.github.io/calibration/", text)
        self.assertIn("assets/new_structured_calibration.pdf", text)
        self.assertIn("supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py", text)
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
        self.assertNotIn('"pbs"', recipe_text)
        self.assertNotIn('"nodetype"', recipe_text)
        self.assertNotIn('"ngpus"', recipe_text)
        self.assertIn("Edit before running", index_text)
        self.assertIn("write your own scheduler wrapper", index_text)
        self.assertNotIn("JSON-to-PBS", index_text)
        self.assertIn("Edit before running", readme_text)
        self.assertIn("write your own scheduler wrapper", readme_text)

    def test_sage_launcher_supplement_is_included(self):
        launcher = ROOT / "supplement" / "sage_recipe_launcher" / "scripts" / "launch_sage_calibration.py"
        recipe = ROOT / "supplement" / "sage_recipe_launcher" / "recipes" / "sage8_ensemble_map_20iter.json"

        self.assertTrue(launcher.exists(), f"missing SAGE launcher supplement: {launcher}")
        self.assertTrue(recipe.exists(), f"missing SAGE recipe supplement: {recipe}")
        self.assertIn("STRATEGY_PRESETS", launcher.read_text())
        self.assertIn('"strategy": "ensemble_map"', recipe.read_text())


if __name__ == "__main__":
    unittest.main()
