"""
The four-fifths (0.80 ratio) rule.

Origin and standing
-------------------
The four-fifths rule comes from the US Uniform Guidelines on Employee Selection
Procedures (1978, 29 CFR 1607.4D): a selection rate for any group less than 80%
of the rate for the group with the highest rate is regarded as evidence of
adverse impact. It is not a statistical test -- it is a deterministic screening
convention -- and it was written for *selection rates*, which are probabilities
in [0, 1] with a meaningful zero. It is nonetheless the dominant disparity
convention in health-system model governance, which is why it belongs in this
comparison.

How it is applied here
----------------------
Within each demographic partition, over the levels admitted by the inclusion
rule, the disparate-impact ratio is

    ratio = min_k AUROC_k / max_k AUROC_k ,

and the cohort ratio is the minimum over partitions. The rule flags when
ratio < 0.80. There is no p-value; the statistic is the ratio itself and the
runtime is the cost of computing the subgroup AUROCs.

Which implementation
--------------------
The reported cohort numbers are produced by **fairlearn 0.13.0**, already in the
environment: ``MetricFrame(metrics=roc_auc_score, ...).ratio(method="between_groups")``
is exactly ``min_k AUROC_k / max_k AUROC_k``, and
``.difference(method="between_groups")`` is exactly the max-min gap. Using the
published implementation rather than our own is the instruction, and
:func:`ratio_fairlearn` is what :func:`run_cohort` calls.

:func:`ratio` is a dependency-free equivalent used inside the Type I simulation,
where a ``MetricFrame`` per subgroup per replicate would dominate the runtime.
``tests/test_comparators.py::test_four_fifths_matches_fairlearn_on_every_cohort``
asserts the two agree to machine precision on all ten cohorts under all five
inclusion rules.

``aif360`` 0.6.1 and ``aequitas`` 1.1.0 were checked and **not installed**. Both
would have downgraded ``pandas`` from 3.0.2 to 2.3.3 in the environment the
published null results were computed in, which is an unacceptable reproducibility
risk for numbers going to press; and neither adds anything here, since their
disparity modules compute ratios and differences of group metrics -- which
fairlearn already provides -- and neither implements DiCiccio's studentized
permutation test or Lum's double-corrected variance estimator.

A structural caveat that has to be reported
-------------------------------------------
Transplanting a selection-rate rule onto AUROC is not innocent. AUROC has an
uninformative point at 0.5, not at 0. A model that is *perfect* in one subgroup
and *pure noise* in another gives ratio = 0.5/1.0 = 0.50, which flags; but a
model at AUROC 0.85 versus 0.70 -- a 0.15 gap, three times the size of the naive
threshold and larger than every gap in these ten cohorts bar one -- gives
ratio = 0.82 and does not flag. For the rule to fire at all, the worse subgroup's
AUROC must fall below 0.8 times the better one's, which for a model performing at
AUROC 0.8 means the worse subgroup must be at 0.64. :func:`min_detectable_gap`
reports, for each partition, the max-min gap that would have been required, so
the rule's effective sensitivity is visible rather than implied.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from recompute.comparators.core import (
    FLAG,
    NO_FLAG,
    NOT_EVALUABLE,
    RULE_NAMES,
    CohortData,
    Level,
    MethodResult,
    admissible,
    all_level_stats,
)

METHOD = "four_fifths"

#: The 0.80 threshold of 29 CFR 1607.4D.
THRESHOLD = 0.80


def ratio(levels: Sequence[Level]) -> float:
    """min AUROC / max AUROC over the admitted levels of one partition."""
    if len(levels) < 2:
        return float("nan")
    a = np.array([lv.auc for lv in levels], dtype=float)
    hi = float(a.max())
    if hi <= 0:
        return float("nan")
    return float(a.min() / hi)


def ratio_fairlearn(y: np.ndarray, s: np.ndarray, codes: np.ndarray,
                    rule: str) -> Dict[str, float]:
    """The four-fifths ratio via fairlearn's ``MetricFrame`` (published impl.).

    Restricted to the levels the inclusion rule admits, then handed to
    ``MetricFrame``; ``ratio(method="between_groups")`` is the min/max ratio and
    ``difference(method="between_groups")`` is the max-min gap.
    """
    from fairlearn.metrics import MetricFrame
    from sklearn.metrics import roc_auc_score

    from recompute.comparators.core import level_stats

    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    codes = np.asarray(codes)

    # Which level codes the rule admits. ``level_stats`` walks levels in
    # ``np.unique`` order and drops the non-estimable ones, so re-derive the
    # surviving codes the same way.
    estimable = [lv for lv in np.unique(codes)
                 if (int(y[codes == lv].sum()) >= 2
                     and int((codes == lv).sum()) - int(y[codes == lv].sum()) >= 2)]
    keep_stats = admissible(level_stats(y, s, codes), rule)
    keep_codes = [lv for lv, st in zip(estimable, level_stats(y, s, codes))
                  if st in keep_stats]
    if len(keep_codes) < 2:
        return {"ratio": float("nan"), "difference": float("nan"),
                "n_levels": len(keep_codes)}
    m = np.isin(codes, keep_codes)
    mf = MetricFrame(metrics=roc_auc_score, y_true=y[m], y_pred=s[m],
                     sensitive_features=codes[m])
    return {
        "ratio": float(mf.ratio(method="between_groups")),
        "difference": float(mf.difference(method="between_groups")),
        "min_auc": float(mf.by_group.min()),
        "max_auc": float(mf.by_group.max()),
        "n_levels": len(keep_codes),
    }


def min_detectable_gap(levels: Sequence[Level]) -> float:
    """The max-min AUROC gap the 0.80 rule would have needed to fire.

    With the best subgroup fixed at its observed AUROC ``hi``, the rule fires
    only once the worst subgroup drops to ``0.8 * hi``, i.e. once the gap reaches
    ``0.2 * hi``. Reported so the rule's sensitivity on this scale is explicit.
    """
    if len(levels) < 2:
        return float("nan")
    hi = float(max(lv.auc for lv in levels))
    return float((1.0 - THRESHOLD) * hi)


def run_cohort(data: CohortData, rules: Optional[List[str]] = None
               ) -> Dict[str, object]:
    rules = list(rules) if rules is not None else list(RULE_NAMES)
    t0 = time.perf_counter()
    obs_levels = all_level_stats(data.y, data.s, data.codes_by_col)

    results: Dict[str, MethodResult] = {}
    diagnostics: Dict[str, Dict[str, object]] = {}
    for rule in rules:
        per_part = {}
        for col, lv in obs_levels.items():
            keep = admissible(lv, rule)
            if len(keep) >= 2:
                # Published implementation: fairlearn's MetricFrame.
                fl = ratio_fairlearn(data.y, data.s, data.codes_by_col[col],
                                     rule)
                per_part[col] = {
                    "ratio": fl["ratio"],
                    "min_auc": fl["min_auc"],
                    "max_auc": fl["max_auc"],
                    "gap_needed_to_fire": min_detectable_gap(keep),
                    "observed_gap": fl["difference"],
                    "n_levels": len(keep),
                    "implementation": "fairlearn.metrics.MetricFrame.ratio",
                    "ratio_internal": ratio(keep),
                }
        if not per_part:
            results[rule] = MethodResult(
                METHOD, rule, NOT_EVALUABLE, statistic_name="min/max AUROC ratio",
                runtime_s=time.perf_counter() - t0,
                detail="no partition has two admissible levels")
            continue

        worst = min(per_part, key=lambda c: per_part[c]["ratio"])
        r = per_part[worst]["ratio"]
        results[rule] = MethodResult(
            method=METHOD,
            rule=rule,
            conclusion=FLAG if (np.isfinite(r) and r < THRESHOLD) else NO_FLAG,
            statistic=float(r),
            statistic_name="min/max AUROC ratio",
            p_value=None,
            runtime_s=time.perf_counter() - t0,
            detail=(f"worst_partition={worst}; "
                    f"min_auc={per_part[worst]['min_auc']:.4f}; "
                    f"max_auc={per_part[worst]['max_auc']:.4f}; "
                    f"gap_needed_to_fire={per_part[worst]['gap_needed_to_fire']:.4f}; "
                    f"observed_gap={per_part[worst]['observed_gap']:.4f}; "
                    f"impl=fairlearn.metrics.MetricFrame.ratio; "
                    f"no p-value: deterministic screening rule"),
        )
        diagnostics[rule] = per_part

    return {"results": results, "diagnostics": diagnostics,
            "runtime_s": time.perf_counter() - t0}


def decide(ctx, rule: str) -> float:
    """1.0 if the rule flags, 0.0 if not, nan if not evaluable."""
    worst = np.inf
    seen = False
    for lv in ctx.observed().values():
        keep = admissible(lv, rule)
        if len(keep) >= 2:
            seen = True
            worst = min(worst, ratio(keep))
    if not seen or not np.isfinite(worst):
        return float("nan")
    return float(worst < THRESHOLD)
