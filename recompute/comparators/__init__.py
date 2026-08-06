"""
State-of-the-art comparators for the per-cohort permutation null.

JBI requires evaluation against the state of the art. ``recompute/null_reference.py``
implements a per-cohort stratified permutation null for the max-min subgroup
AUROC gap. This package implements four published or conventional alternatives
and evaluates all five procedures on identical inputs.

  ``diciccio``     DiCiccio, Vasudevan, Basu, Kenthapadi & Tomkins (2020, KDD),
                   "Evaluating Fairness Using Permutation Tests" -- a
                   *studentized* permutation test, valid under the composite
                   null of equal metric with unequal score distributions.
  ``lum``          Lum, Zhang & Bower (2022, FAccT), "De-biasing 'bias'
                   measurement" -- the double-corrected variance estimator for
                   group-wise performance disparity.
  ``four_fifths``  The 0.80 (four-fifths) ratio rule, the dominant convention in
                   health-system governance.
  ``naive``        Fixed threshold: flag when the max-min AUROC gap >= 0.05.
  ``incumbent``    The existing permutation null, read back from
                   ``recompute/results/null_joint/`` so that the comparison uses
                   the exact published numbers.

Every method sees the same ten cohorts, the same ``train_test_split`` seeds, the
same fitted models, the same scores, and the same five subgroup-inclusion rules
(:data:`recompute.null_reference.INCLUSION_RULES`). Permutation-based methods use
seed 42 and B = 10,000, matching the incumbent exactly.

Entry point::

    python -m recompute.comparators.run --stage all
"""

from __future__ import annotations

__all__ = [
    "core",
    "diciccio",
    "lum",
    "four_fifths",
    "naive",
    "incumbent",
    "simulate",
    "type1",
    "run",
]

#: Seed shared by every stochastic component, matching the incumbent.
SEED = 42

#: Permutation replicates for the cohort comparison, matching the incumbent.
N_PERM = 10_000

#: Nominal level for every test.
ALPHA = 0.05
