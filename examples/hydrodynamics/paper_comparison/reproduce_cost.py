"""Recompute the paper's refinement-complexity reductions from measured runs.

Sedov final records come from the v0.5.0 sedov-beta{0,1}.json settings.
Cylinder rows come from rotcyl_completed_summary.csv, the source for the
paper's five-initial-run Rotating Cylinder results table.
"""

import csv
from pathlib import Path


def main():
    with Path(__file__).with_name("measured_costs.csv").open(newline="") as stream:
        costs = {(r["case"], r["stage"], r["beta"]): int(r["refinement_complexity"])
                 for r in csv.DictReader(stream)}
    for case, stage, beta, chosen in [
        ("Sedov", "final", "0", "1"),
        ("Rotating cylinder", "best initial", "", "0.1"),
    ]:
        reference = costs[case, stage, beta]
        final = costs[case, "final", chosen]
        reduction = 100 * (1 - final / reference)
        print(f"{case}: {reference:,} -> {final:,} ({reduction:.1f}% lower)")


if __name__ == "__main__":
    main()
