# Fully Autonomous Calibration for Astrophysics

**Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration**

A recipe-driven active-learning calibration project for astrophysical simulations
and semi-analytic galaxy-formation models. The generic recipe and launcher are
the primary public interface; SAGE is the included worked example. The workflow
learns proposal distributions from completed evaluations, selects new parameter
vectors, runs the case simulator through a wrapper, and records the adaptive
loop as reproducible artifacts.

## Quick Links

- **Project page:** https://YvonneWANGYY.github.io/calibration/
- **Repository:** https://github.com/YvonneWANGYY/calibration
- **Local recommendation library:** [`src/autocalibration`](src/autocalibration)
- **SAGE-like notebook:** [notebooks/sage_like_recommendation_loop.ipynb](notebooks/sage_like_recommendation_loop.ipynb)
- **Generic launcher:** [supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py](supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py)
- **Generic recipe:** [supplement/generic_recipe_launcher/recipes/adaptive_calibration_template.json](supplement/generic_recipe_launcher/recipes/adaptive_calibration_template.json)
- **Paper status:** Paper in prep
- **SAGE launcher:** [supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py](supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py)
- **SAGE recipe:** [supplement/sage_recipe_launcher/recipes/sage8_ensemble_map_20iter.json](supplement/sage_recipe_launcher/recipes/sage8_ensemble_map_20iter.json)

## What This Project Shows

- A reusable calibration engine spanning Sedov--Taylor, Rotating Cylinder, and
  local SAGE case studies.
- CRQSF-guided adaptive parameter-space exploration after researcher-defined
  targets, bounds, simulator wrappers, resources, and initial designs are set.
- A generic recipe and launcher template that documents the reusable
  calibration interface separately from any one case study.
- A local recommendation loop that can return Cold-start LHS designs when no
  initial metrics exist, then continue from user-provided CSV, JSON, or optional
  HDF5 metrics files.
- A SAGE eight-observable active-learning comparison that improves the shared
  best initial design from `R2 = 0.435944` to a best active result of
  `R2 = 0.734845` in the stored 20-iteration summaries.
- A SAGE example recipe that moves brittle run settings out of hard-coded
  scripts and into a preflighted JSON configuration.

## Autonomy boundary

Autonomous means the closed-loop run proceeds after the recipe is specified. It
does not mean the system invents the science target, parameter domain, simulator
wrapper, PBS resource policy, or initial design. Those remain researcher-defined
inputs; the automated contribution is the subsequent proposal fitting,
acquisition, simulator dispatch, and run bookkeeping.

## Edit before running

Start from the generic recipe for a new calibration problem. Use the SAGE recipe
as a concrete example of how a case study fills the same interface without
exposing local workspace paths. Before reusing them, update the
researcher-defined inputs:

- Science files: `target`, `initial_summary`, `template`, and `crqsf_root`.
- Output identity: `out_root` and `seed_label`.
- Parameter controls: `parameter_set`, `theta_min`, `theta_max`, and strategy
  settings when changing the calibration problem.
- Scheduler submission: write your own scheduler wrapper around the printed
  command for PBS, SLURM, or local execution.

## Local Recommendation Loop

The lightweight library in `src/autocalibration` supports the core manual loop:

1. Define parameters and ranges.
2. Load a target profile from JSON.
3. If no metrics exist, request `n` Cold-start LHS recommendations.
4. Run those parameter vectors in your simulator or model.
5. Add new metrics from CSV, JSON, or optional HDF5.
6. Request the next recommendations until you stop.

```python
from autocalibration import AdaptiveCalibrationSession, ParameterSpec

parameters = [
    ParameterSpec("SfrEfficiency", 0.1, 1.0),
    ParameterSpec("FeedbackReheatingEpsilon", 0.5, 5.0),
]

session = AdaptiveCalibrationSession.from_files(
    parameters=parameters,
    target_profile="examples/sage_like_target_profile.json",
)
initial_runs = session.recommend(n=4, seed=1)

session.add_observations_from_file(
    "examples/sage_like_initial_metrics.csv",
    metric_columns=["stellar_mass_density", "gas_fraction"],
)
next_runs = session.recommend(n=2, seed=5)
```

The notebook [notebooks/sage_like_recommendation_loop.ipynb](notebooks/sage_like_recommendation_loop.ipynb)
walks through this SAGE-like local smoke test using
[examples/sage_like_initial_metrics.csv](examples/sage_like_initial_metrics.csv)
and [examples/sage_like_target_profile.json](examples/sage_like_target_profile.json).
HDF5 metrics are supported through explicit dataset paths when `h5py` is installed.

## Repository Layout

- `index.html` - static project homepage for GitHub Pages from repository root.
- `assets/` - homepage figures and project visual materials.
- `src/autocalibration/` - lightweight local recommendation library.
- `examples/` - SAGE-like CSV/JSON files for the local recommendation loop.
- `notebooks/sage_like_recommendation_loop.ipynb` - notebook smoke test for
  Cold-start LHS and iterative recommendations.
- `supplement/generic_recipe_launcher/` - generic adaptive-calibration recipe
  and command-materialization launcher.
- `supplement/sage_recipe_launcher/` - SAGE worked-example launcher and recipe.
- `profile/README-project-card.md` - short card for a GitHub profile README.
- `tests/test_project_homepage.py` - structural checks for the public bundle.

## Local Check

```bash
python -m unittest discover -s tests
python -m py_compile src/autocalibration/*.py
python -m py_compile supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py
python -m py_compile supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py
```

## GitHub Pages

Enable GitHub Pages from the `main` branch and repository root. The public page
will be:

`https://YvonneWANGYY.github.io/calibration/`
