"""
Cohort-specific null reference for the Inclusivity AUC parity gap.

Why this exists
---------------
``verification/results/p2_summary.json`` establishes that the max-min AUC gap
has a strictly positive expectation under an *exact equality null* (every
subgroup shares the same true AUC), and that the bias grows with the number of
pooled subgroups and shrinks as m^-0.5 in subgroup size. Its headline cell --
10 subgroups of 500 -- gives mean 0.089 and p95 0.130.

That grid is generic. Each real cohort has its own partition geometry: its own
number of demographic columns, its own number of levels per column, its own
wildly unequal level sizes and its own per-level prevalences. A gap of 0.12 is
alarming against a null of 0.089 and unremarkable against a null of 0.15. So we
compute the null *for the actual cohort*, using the same equality-null logic as
verify_p2.py but with the observed geometry rather than a balanced grid.

Design: stratified label-preserving permutation
-----------------------------------------------
The observed scores and outcomes are held FIXED. Only the demographic
assignment is resampled, by permuting subgroup labels *within* the positive rows
and *within* the negative rows separately. This preserves exactly:

  * every subgroup's size n_k,
  * every subgroup's positive count and hence its prevalence,
  * the marginal score distribution and the cohort AUROC,

while making subgroup membership independent of the score *within each outcome
class* -- which is precisely the statement "every subgroup has the same true
AUC". The true parity gap is therefore 0 by construction and everything
observed is selection bias plus sampling noise, at the cohort's real geometry.

Each demographic column is permuted independently, matching the fact that
``evaluate_inclusivity`` computes an independent gap within each column and then
maximises over columns.

The statistic replicated is the 0.2.0 headline: the maximum over demographic
columns of the within-column (max - min) subgroup AUC, applying the same
n >= min_subgroup_n and non-degenerate-labels exclusion rule used in the point
estimate.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney AUC via midranks; nan when a class is absent.

    Identical to ``sklearn.metrics.roc_auc_score`` (verify_p2.py measures the
    max deviation at 1.1e-16) but fast enough for a Monte-Carlo loop.
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s)
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _max_partition_gap(
    y: np.ndarray,
    s: np.ndarray,
    codes_by_col: Dict[str, np.ndarray],
    min_subgroup_n: int,
) -> float:
    """Max over columns of the within-column (max - min) subgroup AUC."""
    best = np.nan
    for codes in codes_by_col.values():
        aucs: List[float] = []
        for lvl in np.unique(codes):
            mask = codes == lvl
            n_grp = int(mask.sum())
            if n_grp < min_subgroup_n:
                continue
            y_g = y[mask]
            n_pos = int(y_g.sum())
            if n_pos < 2 or (n_grp - n_pos) < 2:
                continue
            a = fast_auc(y_g, s[mask])
            if not np.isnan(a):
                aucs.append(a)
        if len(aucs) >= 2:
            gap = max(aucs) - min(aucs)
            best = gap if np.isnan(best) else max(best, gap)
    return float(best)


def cohort_null_reference(
    y_true,
    scores,
    demographic_df: pd.DataFrame,
    subgroup_columns: Optional[List[str]] = None,
    min_subgroup_n: int = 30,
    n_reps: int = 2000,
    random_state: int = 42,
    observed_gap: Optional[float] = None,
) -> Dict[str, object]:
    """Null distribution of the max per-partition AUC gap for THIS cohort.

    Parameters
    ----------
    y_true, scores : array-like
        Observed outcomes and model scores on the evaluation split. Held fixed.
    demographic_df : pd.DataFrame
        The same frame passed to ``evaluate_inclusivity``.
    subgroup_columns : list of str, optional
        Columns to use; defaults to all.
    min_subgroup_n : int
        The same inclusion rule as the point estimate (default 30).
    n_reps : int
        Monte-Carlo replications. 2000 matches verify_p2.py.
    random_state : int
        Seed.
    observed_gap : float, optional
        The measured max per-partition gap. When supplied, a one-sided
        Monte-Carlo p-value and an excess-over-null are reported.

    Returns
    -------
    dict with the null mean / sd / median / p95 / p99, the observed gap, the
    one-sided p-value, and the excess of observed over the null mean.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    cols = (
        subgroup_columns
        if subgroup_columns is not None
        else list(demographic_df.columns)
    )
    demo = demographic_df.reset_index(drop=True)

    # Integer-code each column once; permutation then works on small int arrays.
    codes_by_col: Dict[str, np.ndarray] = {}
    for c in cols:
        _, inv = np.unique(np.asarray(demo[c].astype(str)), return_inverse=True)
        codes_by_col[c] = inv.astype(np.int32)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    rng = np.random.default_rng(random_state)
    vals = np.full(n_reps, np.nan, dtype=float)
    for r in range(n_reps):
        permuted: Dict[str, np.ndarray] = {}
        for c, codes in codes_by_col.items():
            new = np.empty_like(codes)
            # Permute labels within the positives and within the negatives:
            # preserves each level's size and positive count exactly.
            new[pos_idx] = codes[rng.permutation(pos_idx)]
            new[neg_idx] = codes[rng.permutation(neg_idx)]
            permuted[c] = new
        vals[r] = _max_partition_gap(y, s, permuted, min_subgroup_n)

    v = vals[~np.isnan(vals)]
    out: Dict[str, object] = {
        "null_design": (
            "stratified permutation of subgroup labels within outcome classes; "
            "preserves subgroup sizes and prevalences, forces equal true AUC"
        ),
        "n_reps": int(n_reps),
        "n_valid_reps": int(len(v)),
        "min_subgroup_n": int(min_subgroup_n),
        "random_state": int(random_state),
    }
    if len(v) == 0:
        out["null_estimable"] = False
        return out
    out.update({
        "null_estimable": True,
        "null_mean_gap": float(np.mean(v)),
        "null_sd_gap": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "null_median_gap": float(np.median(v)),
        "null_p95_gap": float(np.percentile(v, 95)),
        "null_p99_gap": float(np.percentile(v, 99)),
    })
    if observed_gap is not None and np.isfinite(observed_gap):
        # One-sided Monte-Carlo p-value with the standard +1 correction.
        n_ge = int(np.sum(v >= float(observed_gap)))
        out["observed_gap"] = float(observed_gap)
        out["p_value_vs_null"] = float((n_ge + 1) / (len(v) + 1))
        out["excess_over_null_mean"] = float(observed_gap - np.mean(v))
        out["exceeds_null_p95"] = bool(observed_gap > np.percentile(v, 95))
    return out


#: Generic reference cell from verification/results/p2_summary.json, quoted in
#: the task: 10 subgroups of 500 under exact equality.
P2_REFERENCE_10x500 = {
    "n_groups": 10,
    "group_size": 500,
    "mean_range": 0.08887413961480663,
    "p95_range": 0.1304391085449873,
    "P_gt_0.05": 0.9665,
    "P_gt_0.10": 0.2845,
    "source": "verification/results/p2_summary.json (headline_cells)",
}
