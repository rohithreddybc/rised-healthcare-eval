"""
The incumbent: the per-cohort stratified permutation null of
``recompute/null_reference.py``.

Results are read back from ``recompute/results/null_joint/<cohort>.json`` rather
than recomputed, for two reasons. First, those files *are* the published numbers
(B = 10,000, seed 42, joint scheme, all five inclusion rules), so the comparison
table cannot drift from the manuscript. Second, recomputing them would consume
about 40 minutes of wall clock to reproduce values that
``tests/test_null_joint.py`` already pins.

:func:`recompute_gap` recomputes the observed statistic from the comparator
package's own code path; the test suite asserts it matches the stored value,
which is what certifies that every comparator is looking at the same scores,
the same splits and the same subgroup coding as the incumbent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from recompute.comparators.core import (
    FLAG,
    NO_FLAG,
    NOT_EVALUABLE,
    REPO,
    RULE_NAMES,
    MethodResult,
)

METHOD = "permutation_null"

NULL_DIR = REPO / "recompute" / "results" / "null_joint"


def load_payload(cohort: str) -> Dict[str, object]:
    path = NULL_DIR / f"{cohort}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; run `python -m recompute.run_null_joint` first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_cohort(cohort: str, rules: Optional[List[str]] = None,
               alpha: float = 0.05, scheme: str = "joint"
               ) -> Dict[str, object]:
    rules = list(rules) if rules is not None else list(RULE_NAMES)
    payload = load_payload(cohort)
    if payload.get("status") != "ok":
        raise RuntimeError(f"{cohort}: stored null run did not succeed")

    block = payload["results"][scheme]
    # The stored runtime covers one permutation pass shared by all five rules,
    # which is how the incumbent is actually run; attributing the whole pass to
    # each rule would overstate it fivefold.
    total_perm_s = float(block["runtime_s"]) + float(payload["load_runtime_s"])

    results: Dict[str, MethodResult] = {}
    for rule in rules:
        entry = block[rule]
        p = entry.get("p_value_vs_null")
        obs = entry.get("observed_gap")
        if p is None or obs is None or not np.isfinite(obs):
            results[rule] = MethodResult(
                METHOD, rule, NOT_EVALUABLE, statistic_name="max-min AUROC gap",
                runtime_s=total_perm_s,
                detail=str(entry.get("note", "not estimable")))
            continue
        results[rule] = MethodResult(
            method=METHOD,
            rule=rule,
            conclusion=FLAG if p < alpha else NO_FLAG,
            statistic=float(obs),
            statistic_name="max-min AUROC gap",
            p_value=float(p),
            p_is_floor=bool(entry.get("p_is_floor", False)),
            runtime_s=total_perm_s,
            detail=(f"B={payload['n_reps']}; scheme={scheme}; "
                    f"null_p95={entry.get('null_p95_gap'):.4f}; "
                    f"null_mean={entry.get('null_mean_gap'):.4f}"),
        )
    return {"results": results, "runtime_s": total_perm_s, "payload": payload}


def recompute_gap(data, rule: str) -> float:
    """Recompute the incumbent's observed statistic via the comparator code."""
    from recompute.comparators.naive import cohort_gap

    return cohort_gap(data, rule)


def pvalue_only(ctx, rule: str, n_perm: int,
                rng: np.random.Generator) -> float:
    """The incumbent's p-value for one simulated dataset.

    The statistic and the permutation draws are the incumbent's: the draws come
    from :func:`recompute.null_reference.draw_permuted_codes` (called inside
    :meth:`~recompute.comparators.core.PermContext.draw`) and the statistic is
    the max over partitions of the within-partition max-min AUROC gap, which
    ``tests/test_comparators.py`` pins against
    :func:`recompute.null_reference.partition_gaps_by_rule` on real data.
    """
    from recompute.comparators.core import gap_from_levels, mc_p

    obs = gap_from_levels(ctx.observed(), rule)
    if not np.isfinite(obs):
        return float("nan")
    vals = np.full(n_perm, np.nan)
    for b in range(n_perm):
        vals[b] = gap_from_levels(ctx.draw(rng), rule)
    return mc_p(vals, float(obs))[0]
