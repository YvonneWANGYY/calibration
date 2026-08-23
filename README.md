# Fully Autonomous Calibration for Astrophysics

**Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration**

A recipe-driven active-learning calibration project for astrophysical simulations
and semi-analytic galaxy-formation models. The workflow learns proposal
distributions from completed evaluations, selects new parameter vectors, runs the
case simulator through a wrapper, and records the adaptive loop as reproducible
artifacts.

## Quick Links

- **Project page:** https://YvonneWANGYY.github.io/calibration/
- **Repository:** https://github.com/YvonneWANGYY/calibration
- **Paper PDF:** [assets/new_structured_calibration.pdf](assets/new_structured_calibration.pdf)
- **SAGE launcher:** [supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py](supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py)
- **SAGE recipe:** [supplement/sage_recipe_launcher/recipes/sage8_ensemble_map_20iter.json](supplement/sage_recipe_launcher/recipes/sage8_ensemble_map_20iter.json)

## What This Project Shows

- A reusable calibration engine spanning Sedov--Taylor, Rotating Cylinder, and
  local SAGE case studies.
- CRQSF-guided adaptive parameter-space exploration after researcher-defined
  targets, bounds, simulator wrappers, resources, and initial designs are set.
- A SAGE eight-observable active-learning comparison that improves the shared
  best initial design from `R2 = 0.435944` to a best active result of
  `R2 = 0.734845` in the stored 20-iteration summaries.
- A recipe-based launcher that moves brittle SAGE run settings out of
  hard-coded scripts and into a preflighted JSON configuration.

## Autonomy boundary

Autonomous means the closed-loop run proceeds after the recipe is specified. It
does not mean the system invents the science target, parameter domain, simulator
wrapper, PBS resource policy, or initial design. Those remain researcher-defined
inputs; the automated contribution is the subsequent proposal fitting,
acquisition, simulator dispatch, and run bookkeeping.

## Repository Layout

- `index.html` - static project homepage for GitHub Pages from repository root.
- `assets/` - homepage figures and the current project PDF.
- `supplement/sage_recipe_launcher/` - SAGE launcher script and example recipe.
- `profile/README-project-card.md` - short card for a GitHub profile README.
- `tests/test_project_homepage.py` - structural checks for the public bundle.

## Local Check

```bash
python -m unittest discover -s tests
python -m py_compile supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py
```

## GitHub Pages

Enable GitHub Pages from the `main` branch and repository root. The public page
will be:

`https://YvonneWANGYY.github.io/calibration/`
