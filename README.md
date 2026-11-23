# Fully Autonomous Calibration for Astrophysics

**Fully Autonomous Calibration for Astrophysics via Adaptive Parameter-Space Exploration**

A recipe-driven active-learning calibration project for astrophysical simulations
and semi-analytic galaxy-formation models. The installable CLI runs local calibration projects.

For the CRQSF research workflow:
The generic recipe and launcher are
the primary public interface; SAGE is the included worked example. The workflow
learns proposal distributions from completed evaluations, selects new parameter
vectors, runs the case simulator through a wrapper, and records the adaptive
loop as reproducible artifacts.

## Install and calibrate your own simulator

With Python 3.10 or newer, install from the repository root into your active
environment. The installed command works from any directory.

```bash
python -m pip install .
autocalibration init my-problem
autocalibration run my-problem/problem.json --output my-run
```

The starter runs an exponential-decay calibration. To use your own model:

1. Edit `my-problem/problem.json`. Give each parameter a unique name and numeric
   bounds (`lower`, `upper`). An optional `scale` accepts `linear` or `log`.
   Set `target` to the named scalar observables you want to match.
2. Replace `simulate.py` or set `simulator` to your own command as an array
   of arguments. Each evaluation starts the command in the problem file's
   directory. It receives one JSON object on standard input, such as
   `{"amplitude": 1.2, "rate": 0.3}`, and must write one JSON object containing
   a finite number for each target metric to standard output. Send logs to
   standard error and return a nonzero exit status on failure. The special
   argument `{python}` selects the interpreter running the calibration.
3. Choose `initial_samples`, `adaptive_steps`, `candidate_pool_size`, and
   `seed`. Run the command with a fresh output directory.

The local engine evaluates the Latin hypercube design first, then selects and
evaluates one adaptive point per step. It minimizes the unweighted sum of
squared differences between simulated and target metrics. Scale your metrics
and targets consistently when their units differ. Candidate ranking uses
the nearest observed loss and a distance-based exploration bonus. The CRQSF
research launchers are documented separately below.

The run saves the problem configuration in `problem.json` and appends each
completed evaluation to `evaluations.jsonl`. Adaptive rows include their
acquisition score alongside the measured loss. On successful completion,
`summary.json` identifies the lowest-loss evaluated point. If the simulator
fails, completed rows remain available for inspection.

## Reproduce the SAGE results

Version 0.2.0 includes the processed observations and simulation measurements
needed for the paper's SAGE model-comparison table. From an installed copy, run:

```bash
autocalibration reproduce-sage
```

The command reselects the reported model for each of six configurations from
its archived evaluation history. It computes the equally weighted five- and
eight-observable losses for those models and the published manual calibration.
For each observable it divides the model residual by half the range of its
observed target values, squares and averages across bins, then averages the
observable means. An incomplete target profile is ineligible for selection.
The eight-target fixed-baryon row uses the evaluated final recommendation.
The reference and fixed-baryon eight-target rows yield eight-observable losses
of 0.128536 and 0.100687.

To exercise the paper's flow-based proposal code, install the optional PyTorch
dependency and train a first-round ensemble from the archived initial design:

```bash
python -m pip install '.[flow]'
autocalibration propose-sage --smoke --output first-proposal
```

This example uses the five-target, 50% stellar-mass-function, fixed-baryon
configuration. It fits 20 ensemble members on five folds of the original
25 simulations and ranks flow draws using mixture log density and disagreement.
The smoke run uses one training epoch and 64 draws. Omit `--smoke` to request
the original first-round training settings and 5,000 draws. The output contains
the checkpoint, training history, and proposed parameters. An independent-start
experiment requires a fresh design and new forward simulations.

The [study directory](src/autocalibration/paper_sage/) includes the binned
observational targets and their source metadata, parameter transforms, initial
and adaptive settings, logged first-round arguments, and 51 processed records
per configuration. The six histories retain shared initial designs. Model
profiles extracted for the comparison figure are included for computing held-out
observable losses. The [flow source](src/autocalibration/paper_sage/flow/) comes
from the fixed-baryon SMF50 run, with local filesystem and checkpoint
publication dependencies removed. The package covers SAGE proposal generation
and score replay. Fresh forward evaluations require a separate SAGE installation,
its merger trees, and a simulator wrapper. The [hydrodynamic configurations](examples/hydrodynamics/) record the reported
Sedov and Rotating Cylinder settings and outcomes. New hydrodynamic runs
require the corresponding Enzo setups.

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

## Run a local calibration

From the repository root, run the exponential-decay example with Python 3.10 or later.
It uses the standard library and the package in this checkout.

```bash
PYTHONPATH=src python3 examples/local_decay.py --seed 42 > decay-run.json
```

The simulator computes the signal at times 1 and 2 for eight Latin hypercube
samples of amplitude and decay rate. Each entry in `initial_evaluations` records
the parameters, both signals, and the sum of squared errors against the target.
The session then selects one new parameter pair using its nearest-observation
acquisition rule. Its acquisition score appears in `next_recommendation`; the
new pair is pending simulation. Reusing the seed reproduces the run.

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
