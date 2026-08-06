"""
Naive fixed-threshold baseline: flag when the max-min subgroup AUROC gap
reaches 0.05.

This is current practice, and it is the correct floor for the comparison. It
computes exactly the incumbent's point statistic -- the maximum over demographic
partitions of the within-partition (max - min) subgroup AUROC, under the same
inclusion rule -- and then compares it to a constant instead of to a null
distribution. Everything the incumbent adds over this baseline is the null; if
the two agree everywhere, the null bought nothing on these cohorts.

The threshold 0.05 is the conventional "clinically meaningful AUROC difference"
cut used in model-governance checklists. It is deliberately not tuned.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from recompute.comparators.core import (
    FLAG,
    NO_FLAG,
    NOT_EVALUABLE,
    RULE_NAMES,
    CohortData,
    MethodResult,
    admissible,
    all_level_stats,
    max_min_gap,
)

METHOD = "fixed_threshold_005"

THRESHOLD = 0.05


def cohort_gap(data: CohortData, rule: str) -> float:
    """The incumbent's observed statistic, recomputed here from scratch.

    Recomputed rather than read from the incumbent's JSON precisely so that the
    test suite can assert the two agree; that assertion is what proves the
    comparators are running on the same statistic as the published procedure.
    """
    obs = all_level_stats(data.y, data.s, data.codes_by_col)
    best = float("nan")
    for lv in obs.values():
        g = max_min_gap(admissible(lv, rule))
        if np.isfinite(g):
            best = g if not np.isfinite(best) else max(best, g)
    return best


def run_cohort(data: CohortData, rules: Optional[List[str]] = None
               ) -> Dict[str, object]:
    rules = list(rules) if rules is not None else list(RULE_NAMES)
    t0 = time.perf_counter()
    obs_levels = all_level_stats(data.y, data.s, data.codes_by_col)

    results: Dict[str, MethodResult] = {}
    for rule in rules:
        best = float("nan")
        worst_col = None
        for col, lv in obs_levels.items():
            g = max_min_gap(admissible(lv, rule))
            if np.isfinite(g) and (not np.isfinite(best) or g > best):
                best, worst_col = g, col
        rt = time.perf_counter() - t0
        if not np.isfinite(best):
            results[rule] = MethodResult(
                METHOD, rule, NOT_EVALUABLE, statistic_name="max-min AUROC gap",
                runtime_s=rt, detail="no partition has two admissible levels")
            continue
        results[rule] = MethodResult(
            method=METHOD,
            rule=rule,
            conclusion=FLAG if best >= THRESHOLD else NO_FLAG,
            statistic=float(best),
            statistic_name="max-min AUROC gap",
            p_value=None,
            runtime_s=rt,
            detail=(f"worst_partition={worst_col}; threshold={THRESHOLD}; "
                    f"no p-value: deterministic threshold rule"),
        )
    return {"results": results, "runtime_s": time.perf_counter() - t0}


def decide(ctx, rule: str) -> float:
    from recompute.comparators.core import gap_from_levels

    best = gap_from_levels(ctx.observed(), rule)
    if not np.isfinite(best):
        return float("nan")
    return float(best >= THRESHOLD)
