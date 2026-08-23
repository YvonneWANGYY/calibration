### Fully Autonomous Calibration for Astrophysics

Adaptive parameter-space exploration for astrophysical calibration, with a
recipe-driven active-learning loop demonstrated on Sedov--Taylor, Rotating
Cylinder, and local SAGE.

- **Project page:** https://YvonneWANGYY.github.io/fully-autonomous-calibration/
- **Repository:** https://github.com/YvonneWANGYY/fully-autonomous-calibration
- **Paper PDF:** https://YvonneWANGYY.github.io/fully-autonomous-calibration/assets/new_structured_calibration.pdf
- **SAGE automation:** `supplement/sage_recipe_launcher/scripts/launch_sage_calibration.py`

Highlights:
- Reusable CRQSF-guided calibration engine across three astrophysical targets.
- SAGE eight-observable active learning improves the shared best initial design from `R2 = 0.435944` to a best active `R2 = 0.734845` in the 20-iteration comparison.
- Autonomy begins after researcher-defined inputs are specified: target, parameter domain, simulator wrapper, resource policy, and initial design.
