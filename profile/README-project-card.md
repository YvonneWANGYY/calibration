### Fully Autonomous Calibration for Astrophysics

Adaptive parameter-space exploration for astrophysical calibration, centered on
a generic recipe interface and demonstrated on Sedov--Taylor, Rotating Cylinder,
and local SAGE.

- **Project page:** https://YvonneWANGYY.github.io/calibration/
- **Repository:** https://github.com/YvonneWANGYY/calibration
- **Paper status:** Paper in prep
- **Generic interface:** `supplement/generic_recipe_launcher/scripts/launch_adaptive_calibration.py`
- **SAGE example:** `supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py`

Highlights:
- Reusable CRQSF-guided calibration engine behind a public generic launcher.
- SAGE eight-observable active learning improves the shared best initial design from `R2 = 0.435944` to a best active `R2 = 0.734845` in the 20-iteration comparison.
- Autonomy begins after researcher-defined inputs are specified: target, parameter domain, simulator wrapper, resource policy, and initial design.
