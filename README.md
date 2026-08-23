# Fully Autonomous Calibration for Astrophysics

Standalone project homepage and reproducibility bundle for:

**Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration**

This folder is intentionally separate from the manuscript/documentation repository.
It contains the public-facing project page, selected visual assets, the paper PDF,
and the SAGE recipe-driven launcher supplement.

## Contents

- `index.html` - static project homepage, ready for GitHub Pages from repository root.
- `assets/` - homepage figures and PDF.
- `supplement/sage_recipe_launcher/` - SAGE recipe launcher script and example recipe.
- `profile/README-project-card.md` - short snippet for a GitHub profile README.
- `tests/test_project_homepage.py` - structural checks for the homepage bundle.

## Local Check

```bash
python -m unittest discover -s tests
python -m py_compile supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py
```

## GitHub Pages

The intended repository name is `fully-autonomous-calibration`. If GitHub Pages
is enabled from the `main` branch and repository root, the project page should be:

`https://YvonneWANGYY.github.io/fully-autonomous-calibration/`
