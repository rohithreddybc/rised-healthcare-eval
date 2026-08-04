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

Two permutation schemes
-----------------------
``scheme="independent"`` (the original, and still the default so that every
previously published number reproduces bit-for-bit) draws a *fresh* permutation
for each demographic column. That makes the per-column gaps independent of one
another.

``scheme="joint"`` draws ONE permutation of the row indices per replicate --
within the positives and within the negatives separately -- and carries *all*
demographic columns together on those permuted rows.

The joint scheme is the correct one. Age, sex, race, insurance and income are
strongly associated in these cohorts; the independent scheme destroys that
association and so makes the per-column gaps independent when in reality they
are positively dependent. The maximum of independent components is
stochastically LARGER than the maximum of positively dependent components with
the same margins, so the independent null is too wide, its p-values are too
large, and a negative conclusion drawn from it is partly an artefact of the
resampling scheme. Note that each column's *marginal* null is identical under
the two schemes -- only the cross-column dependence, and hence the maximum,
changes.

Both schemes preserve the same things (subgroup sizes, subgroup prevalences,
the marginal score distribution, the cohort AUROC) and both break the
subgroup-to-score link conditional on outcome, which is the null of interest.
The joint scheme additionally preserves the full demographic contingency table.

The statistic replicated is the 0.2.0 headline: the maximum over demographic
columns of the within-column (max - min) subgroup AUC, applying the same
n >= min_subgroup_n and non-degenerate-labels exclusion rule used in the point
estimate.

Subgroup inclusion rules
------------------------
``m_min = 30`` was never varied anywhere in the package, and the null's width is
dominated by its noisiest estimable levels, so the inclusion rule is now a swept
parameter (:data:`INCLUSION_RULES`) rather than a hard-coded constant. All rules
are evaluated on the *same* permutation draws, so the sweep costs almost nothing
and the settings are perfectly paired.
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


# ── Subgroup inclusion rules (swept; m30 is the published default) ───────────
#: Every rule additionally requires >= 2 positives and >= 2 negatives, which is
#: the minimum for an estimable within-level AUROC and is what the 0.2.0 point
#: estimate already enforced.
#:
#: ``ev10`` is the events-based rule the prognostic-model literature argues for:
#: a level needs at least 10 observations in EACH outcome class before its
#: AUROC is treated as estimable. At NHIS 2024's prevalence of 0.075 a level of
#: n=30 carries about 2 events, so ``m30`` admits levels whose AUROC is nearly
#: pure noise -- and noise inflates a max-min range.
INCLUSION_RULES: Dict[str, Dict[str, object]] = {
    "m20": {"kind": "size", "min_n": 20,
            "label": "n >= 20"},
    "m30": {"kind": "size", "min_n": 30,
            "label": "n >= 30 (published default)"},
    "m50": {"kind": "size", "min_n": 50,
            "label": "n >= 50"},
    "m100": {"kind": "size", "min_n": 100,
             "label": "n >= 100"},
    "ev10": {"kind": "events", "min_events": 10,
             "label": "n_pos >= 10 and n_neg >= 10"},
}

DEFAULT_RULE = "m30"
SCHEMES = ("independent", "joint")


def _rule_admits(rule: Dict[str, object], n: int, n_pos: int, n_neg: int) -> bool:
    if n_pos < 2 or n_neg < 2:
        return False
    if rule["kind"] == "size":
        return n >= int(rule["min_n"])          # type: ignore[arg-type]
    return min(n_pos, n_neg) >= int(rule["min_events"])   # type: ignore[arg-type]


def _column_level_stats(y: np.ndarray, s: np.ndarray, codes: np.ndarray):
    """(n, n_pos, n_neg, auc) for every level of one column with estimable AUC.

    Computed once per column per replicate and then filtered by each inclusion
    rule, so the whole m_min sweep costs one pass rather than five.
    """
    stats = []
    for lvl in np.unique(codes):
        mask = codes == lvl
        n = int(mask.sum())
        y_g = y[mask]
        n_pos = int(y_g.sum())
        n_neg = n - n_pos
        if n_pos < 2 or n_neg < 2:
            continue
        a = fast_auc(y_g, s[mask])
        if np.isnan(a):
            continue
        stats.append((n, n_pos, n_neg, float(a)))
    return stats


def partition_gaps_by_rule(
    y: np.ndarray,
    s: np.ndarray,
    codes_by_col: Dict[str, np.ndarray],
    rules: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Max over columns of the within-column (max-min) AUC, for EVERY rule.

    One pass over the columns and levels; the rules only differ in which levels
    they keep, so they share the AUROC computations.
    """
    names = list(rules) if rules is not None else list(INCLUSION_RULES)
    best = {k: np.nan for k in names}
    for codes in codes_by_col.values():
        stats = _column_level_stats(y, s, codes)
        for k in names:
            rule = INCLUSION_RULES[k]
            aucs = [a for (n, npos, nneg, a) in stats
                    if _rule_admits(rule, n, npos, nneg)]
            if len(aucs) >= 2:
                gap = max(aucs) - min(aucs)
                best[k] = gap if np.isnan(best[k]) else max(best[k], gap)
    return {k: float(v) for k, v in best.items()}


def code_columns(
    demographic_df: pd.DataFrame,
    cols: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Integer-code each demographic column once, in the given column order."""
    demo = demographic_df.reset_index(drop=True)
    use = cols if cols is not None else list(demo.columns)
    out: Dict[str, np.ndarray] = {}
    for c in use:
        _, inv = np.unique(np.asarray(demo[c].astype(str)), return_inverse=True)
        out[c] = inv.astype(np.int32)
    return out


def draw_permuted_codes(
    codes_by_col: Dict[str, np.ndarray],
    pos_idx: np.ndarray,
    neg_idx: np.ndarray,
    rng: np.random.Generator,
    scheme: str = "independent",
) -> Dict[str, np.ndarray]:
    """One replicate's permuted demographic assignment.

    ``independent`` -- a fresh within-class permutation per column, which
    destroys the association between age, sex, race and insurance.
    ``joint`` -- ONE within-class permutation of the row indices, applied to
    every column, which carries whole demographic rows and therefore preserves
    the joint contingency structure exactly.
    """
    if scheme == "joint":
        n = len(next(iter(codes_by_col.values())))
        perm = np.empty(n, dtype=np.int64)
        perm[pos_idx] = rng.permutation(pos_idx)
        perm[neg_idx] = rng.permutation(neg_idx)
        return {c: codes[perm] for c, codes in codes_by_col.items()}
    if scheme != "independent":
        raise ValueError(f"unknown permutation scheme {scheme!r}")
    permuted: Dict[str, np.ndarray] = {}
    for c, codes in codes_by_col.items():
        new = np.empty_like(codes)
        # Permute labels within the positives and within the negatives:
        # preserves each level's size and positive count exactly.
        new[pos_idx] = codes[rng.permutation(pos_idx)]
        new[neg_idx] = codes[rng.permutation(neg_idx)]
        permuted[c] = new
    return permuted


def cohort_null_reference(
    y_true,
    scores,
    demographic_df: pd.DataFrame,
    subgroup_columns: Optional[List[str]] = None,
    min_subgroup_n: int = 30,
    n_reps: int = 2000,
    random_state: int = 42,
    observed_gap: Optional[float] = None,
    scheme: str = "independent",
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
    scheme : {'independent', 'joint'}
        Permutation scheme; see the module docstring. ``independent`` is the
        default only for backward compatibility -- ``joint`` is the correct one.

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
    # Integer-code each column once; permutation then works on small int arrays.
    codes_by_col = code_columns(demographic_df, cols)

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    rng = np.random.default_rng(random_state)
    vals = np.full(n_reps, np.nan, dtype=float)
    for r in range(n_reps):
        permuted = draw_permuted_codes(
            codes_by_col, pos_idx, neg_idx, rng, scheme=scheme)
        vals[r] = _max_partition_gap(y, s, permuted, min_subgroup_n)

    v = vals[~np.isnan(vals)]
    out: Dict[str, object] = {
        "null_design": (
            f"{scheme} stratified permutation of subgroup labels within "
            "outcome classes; preserves subgroup sizes and prevalences, "
            "forces equal true AUC"
            + ("; one row permutation carried across all demographic columns, "
               "preserving their joint contingency structure"
               if scheme == "joint" else
               "; a fresh permutation per demographic column, which destroys "
               "the association between the columns")
        ),
        "permutation_scheme": scheme,
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
        out.update(mc_pvalue(v, float(observed_gap)))
    return out


def mc_pvalue(null_vals: np.ndarray, observed: float) -> Dict[str, object]:
    """One-sided Monte-Carlo p-value with its Monte-Carlo standard error.

    The +1 correction makes the test exact-valid; its consequence is a hard
    floor at 1/(B+1), which must be REPORTED as an inequality rather than
    printed as if it were an attained value. ``p_is_floor`` flags that case and
    ``p_report`` renders it as ``<= 1/(B+1)``.
    """
    v = np.asarray(null_vals, dtype=float)
    b = len(v)
    n_ge = int(np.sum(v >= observed))
    p = (n_ge + 1) / (b + 1)
    floor = 1.0 / (b + 1)
    # SE of the plug-in binomial proportion; the +1 correction shifts the point
    # estimate but not the sampling variability, and at the floor the estimate
    # is censored so the usual SE understates -- flagged via p_is_floor.
    p_hat = n_ge / b
    se = float(np.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / b))
    return {
        "observed_gap": float(observed),
        "n_null_ge_observed": n_ge,
        "p_value_vs_null": float(p),
        "p_value_mc_se": se,
        "p_value_floor": float(floor),
        "p_is_floor": bool(n_ge == 0),
        "p_report": (f"<= {floor:.2e}" if n_ge == 0 else f"{p:.4f}"),
        "excess_over_null_mean": float(observed - np.mean(v)),
        "exceeds_null_p95": bool(observed > np.percentile(v, 95)),
        "exceeds_null_median": bool(observed > np.median(v)),
        # The smallest observed gap that would have reached p < 0.05, i.e. the
        # null's 95th percentile: the minimum detectable effect of this design.
        "minimum_detectable_gap_p05": float(np.percentile(v, 95)),
    }


def self_check(n_reps: int = 2000, random_state: int = 42) -> Dict[str, float]:
    """Reproduce the p2 headline cell to validate this null against verify_p2.

    verify_p2.py draws a fresh cohort each replicate and assigns group ids at
    random; this module holds the observed cohort fixed and permutes labels
    within outcome classes. The two designs should agree, and do: 10 disjoint
    groups of 500 at true AUC 0.70 and prevalence 0.20 give mean 0.0879 / p95
    0.1285 here against the published 0.0889 / 0.1304.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(random_state)
    mu = norm.ppf(0.70) * np.sqrt(2.0)
    n = 5000
    y = (rng.random(n) < 0.20).astype(int)
    s = rng.normal(loc=mu * y, scale=1.0, size=n)
    demo = pd.DataFrame({"g": rng.permutation(np.repeat(np.arange(10), 500))})
    out = cohort_null_reference(
        y, s, demo, min_subgroup_n=30, n_reps=n_reps, random_state=random_state)
    return {
        "this_module_mean": out["null_mean_gap"],
        "this_module_p95": out["null_p95_gap"],
        "verify_p2_mean": P2_REFERENCE_10x500["mean_range"],
        "verify_p2_p95": P2_REFERENCE_10x500["p95_range"],
    }


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


if __name__ == "__main__":
    res = self_check()
    print(f"this module : mean {res['this_module_mean']:.4f}  "
          f"p95 {res['this_module_p95']:.4f}")
    print(f"verify_p2   : mean {res['verify_p2_mean']:.4f}  "
          f"p95 {res['verify_p2_p95']:.4f}")
