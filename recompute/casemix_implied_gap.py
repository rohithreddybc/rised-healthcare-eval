"""Implied case-mix AUROC gap, computed from each partition's own geometry.

Why this module exists
----------------------
An earlier version of the manuscript read each partition's measured case-mix
magnitude ``rho_hat`` onto the 16-point simulation sweep by linear interpolation
and reported the resulting flag rate as an "induced false-flag rate". That
mapping had three defects and is withdrawn:

1. It borrowed one geometry for all partitions. The sweep fixes every level's
   linear-predictor *mean* at zero and uses three equal-size levels at
   ``n = 2,000`` and prevalence 0.20. Real partitions have 2 to 9 levels, test
   sets from 820 to 19,890 rows, and level means differing by up to several
   logit units. ``rho`` alone is not a sufficient summary of a partition's
   case-mix geometry -- ``casemix_location_3`` has ``rho = 1.0`` exactly, all
   spread equal, and still induces a true AUROC gap of 0.017 purely from
   location, which no ``rho``-indexed mapping can express.
2. The interpolation rule was load-bearing and unreported. Nearest-neighbour
   rather than linear interpolation moves the median from 0.094 to 0.067.
3. Several partitions bracket into a sweep segment whose endpoints are not
   resolvable at 1,000 simulations, so the interpolated value is reading Monte
   Carlo noise.

This module replaces the mapping with a direct computation. For each partition
it takes that partition's *own* measured per-level linear-predictor mean and
standard deviation and evaluates the closed-form subgroup AUROC of
:func:`recompute.casemix_theory.auroc_gaussian_lp` at each level, then reports
the max-minus-min range. That range is the AUROC gap a perfectly fair,
well-specified model would exhibit on that partition's measured predictor
geometry -- the case-mix-implied gap, in AUROC units, on the same scale as the
observed gap it is compared with.

What the quantity assumes, and what it therefore is not
-------------------------------------------------------
``auroc_gaussian_lp`` is exact under two assumptions: the level's linear
predictor is Gaussian, and the model is well specified in that level (the score
is the true conditional event probability). Neither holds exactly on a real
cohort. The implied gap is therefore a *model-based reference value* -- what a
fair model with this predictor geometry would produce -- and not an estimate of
how much of the observed gap is case mix. It carries no interval and no error
rate, and the observed-minus-implied difference reported alongside it is a
descriptive contrast, not a decomposition. It is a direct computation from
measured quantities, which is its whole advantage over the mapping it replaces:
every number traces to two columns of ``cohort_sd_ratios.csv``.

Usage
-----
    python -m recompute.casemix_implied_gap

Reads  ``recompute/results/cohort_sd_ratios.csv``
Writes ``recompute/results/casemix_implied_gap.csv``
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, List

from recompute.casemix_theory import auroc_gaussian_lp

RESULTS = Path(__file__).resolve().parent / "results"
SOURCE = RESULTS / "cohort_sd_ratios.csv"
DEST = RESULTS / "casemix_implied_gap.csv"

#: The reference admissibility rule used throughout the manuscript.
REFERENCE_RULE = "m30"

FIELDS = [
    "cohort",
    "cohort_label",
    "is_clinical",
    "rule",
    "partition",
    "n_levels_admissible",
    "partition_sd_ratio",
    "min_level_lp_sd",
    "max_level_lp_sd",
    "min_level_lp_mean",
    "max_level_lp_mean",
    "level_lp_mean_spread",
    "implied_auroc_min",
    "implied_auroc_max",
    "implied_casemix_gap",
    "observed_auc_gap",
    "observed_minus_implied",
]


def _rows_by_partition(rule: str, clinical_only: bool) -> Dict[tuple, List[dict]]:
    """Group the per-level rows of ``cohort_sd_ratios.csv`` by partition."""
    groups: Dict[tuple, List[dict]] = {}
    with SOURCE.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["rule"] != rule:
                continue
            if clinical_only and row["is_clinical"] != "True":
                continue
            groups.setdefault((row["cohort"], row["partition"]), []).append(row)
    return groups


def compute(rule: str = REFERENCE_RULE, clinical_only: bool = True) -> List[dict]:
    """One record per partition, with its case-mix-implied AUROC gap."""
    out: List[dict] = []
    for (cohort, partition), levels in _rows_by_partition(rule, clinical_only).items():
        means = [float(r["level_lp_mean"]) for r in levels]
        sds = [float(r["level_lp_sd"]) for r in levels]
        implied = [auroc_gaussian_lp(m, s) for m, s in zip(means, sds)]
        head = levels[0]
        out.append(
            {
                "cohort": cohort,
                "cohort_label": head["cohort_label"],
                "is_clinical": head["is_clinical"],
                "rule": rule,
                "partition": partition,
                "n_levels_admissible": int(head["n_levels_admissible"]),
                "partition_sd_ratio": round(float(head["partition_sd_ratio"]), 6),
                "min_level_lp_sd": round(min(sds), 6),
                "max_level_lp_sd": round(max(sds), 6),
                "min_level_lp_mean": round(min(means), 6),
                "max_level_lp_mean": round(max(means), 6),
                "level_lp_mean_spread": round(max(means) - min(means), 6),
                "implied_auroc_min": round(min(implied), 6),
                "implied_auroc_max": round(max(implied), 6),
                "implied_casemix_gap": round(max(implied) - min(implied), 6),
                "observed_auc_gap": round(float(head["partition_observed_auc_gap"]), 6),
                "observed_minus_implied": round(
                    float(head["partition_observed_auc_gap"])
                    - (max(implied) - min(implied)),
                    6,
                ),
            }
        )
    out.sort(key=lambda r: r["implied_casemix_gap"])
    return out


def main() -> None:
    records = compute()
    with DEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    implied = [r["implied_casemix_gap"] for r in records]
    observed = [r["observed_auc_gap"] for r in records]
    residual = [r["observed_minus_implied"] for r in records]
    print(f"{len(records)} clinical partitions at rule {REFERENCE_RULE}")
    print(
        "implied case-mix gap:  median %.4f  min %.4f  max %.4f"
        % (statistics.median(implied), min(implied), max(implied))
    )
    print(
        "observed gap:          median %.4f  min %.4f  max %.4f"
        % (statistics.median(observed), min(observed), max(observed))
    )
    print(
        "observed - implied:    median %.4f  min %.4f  max %.4f"
        % (statistics.median(residual), min(residual), max(residual))
    )
    print(f"exceeding 0.05: {sum(g >= 0.05 for g in implied)} of {len(implied)}")
    print(f"wrote {DEST}")


if __name__ == "__main__":
    main()
