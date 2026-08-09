"""
Shared machinery: AUROC with its DeLong variance, level statistics, and the
cohort bundle every comparator consumes.

The single most important design constraint is that all five procedures must see
*identical* inputs, so that any difference in what they conclude is attributable
to the procedure and to nothing else. That is enforced here:

  * :func:`load_cohort` calls the same ``recompute.cohorts.LOADERS`` the
    incumbent calls, so the split, the seed and the fitted model are the same
    objects the published numbers came from.
  * :func:`level_stats` produces, for one demographic column, the per-level
    ``(n, n_pos, n_neg, auc, var_auc)`` tuple that every comparator filters with
    the *same* :data:`recompute.null_reference.INCLUSION_RULES` predicate the
    incumbent uses.
  * Permutation-based comparators re-use
    :func:`recompute.null_reference.draw_permuted_codes` with ``scheme="joint"``
    and seed 42, so they consume the same permutation draws as the incumbent.

Why DeLong rather than a bootstrap
----------------------------------
Both DiCiccio's studentization and Lum's bias correction need a per-subgroup
standard error for the AUROC. DeLong's estimator is the standard closed-form
choice: it is the exact variance of the Mann-Whitney U-statistic under its
own structural components, it handles ties by midranks, and it costs one sort
per subgroup rather than a nested resampling loop. Inside a 10,000-replicate
permutation loop a bootstrap SE is not affordable, and using a *different* SE
for the observed statistic than for the permuted ones would invalidate the test.
:func:`bootstrap_auc_var` is provided and is checked against DeLong in the test
suite, but is not used in the production path.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for _p in (str(REPO), str(REPO / "recompute" / "_vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recompute.null_reference import (  # noqa: E402
    INCLUSION_RULES,
    _rule_admits,
    code_columns,
    draw_permuted_codes,
)

#: Variances below this are treated as numerically degenerate. A subgroup AUROC
#: whose DeLong variance is zero has perfectly tied placement values (e.g. every
#: positive above every negative), which makes a studentized statistic undefined.
VAR_FLOOR = 1e-12

CLINICAL = ("uci_heart", "diabetes130", "nhis2024", "nhis2023",
            "nhanes2123", "brfss2024")

COHORT_LABELS = {
    "synthetic": "Synthetic baseline",
    "uci_heart": "UCI Heart",
    "diabetes130": "Diabetes 130",
    "nhis2024": "NHIS 2024",
    "nhis2023": "NHIS 2023",
    "nhanes2123": "NHANES 21-23",
    "brfss2024": "BRFSS 2024",
    "adult_income": "Adult Income",
    "acs_income": "ACS-Income",
    "german_credit": "German Credit",
}

#: Same order the incumbent runs in (longest cohort first).
COHORT_ORDER = [
    "diabetes130", "brfss2024", "adult_income", "nhis2023", "nhis2024",
    "acs_income", "synthetic", "nhanes2123", "german_credit", "uci_heart",
]

RULE_NAMES = list(INCLUSION_RULES)


# ── AUROC and its DeLong variance ────────────────────────────────────────────
def auc_delong(y: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    """AUROC and its DeLong variance for one subgroup.

    Returns ``(nan, nan)`` when either outcome class has fewer than two members,
    which is the point at which the variance itself stops being estimable.

    The estimator is the standard structural-component form. Writing V10 for the
    placement of each positive among the negatives and V01 for the placement of
    each negative among the positives,

        AUC = mean(V10) = mean(V01)
        Var(AUC) = S10 / n_pos + S01 / n_neg

    with S10, S01 the sample variances of the placement values. Ties are handled
    by midranks, giving each tied pair a contribution of 0.5, which matches
    ``sklearn.metrics.roc_auc_score`` and matches
    ``recompute.null_reference.fast_auc``.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    m = pos.size
    n = neg.size
    if m < 2 or n < 2:
        return float("nan"), float("nan")

    combined = rankdata(np.concatenate([pos, neg]))
    r_pos_all = combined[:m]
    r_neg_all = combined[m:]
    # Rank of each observation among its own class only; the difference between
    # the combined midrank and the own-class midrank is exactly the number of
    # opposite-class observations below it, counting ties as one half.
    r_pos_own = rankdata(pos)
    r_neg_own = rankdata(neg)

    v10 = (r_pos_all - r_pos_own) / n
    v01 = 1.0 - (r_neg_all - r_neg_own) / m

    auc = float(v10.mean())
    var = float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)
    return auc, max(var, 0.0)


def bootstrap_auc_var(y: np.ndarray, s: np.ndarray, n_boot: int = 2000,
                      seed: int = 42) -> float:
    """Stratified bootstrap variance of the AUROC.

    Validation only -- the production path uses :func:`auc_delong`. Resampling is
    stratified by outcome class, which holds n_pos and n_neg fixed and is the
    resampling scheme that matches DeLong's conditioning.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size < 2 or neg.size < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    yy = np.concatenate([np.ones(pos.size, int), np.zeros(neg.size, int)])
    for b in range(n_boot):
        ss = np.concatenate([rng.choice(pos, pos.size, replace=True),
                             rng.choice(neg, neg.size, replace=True)])
        out[b] = auc_delong(yy, ss)[0]
    return float(np.nanvar(out, ddof=1))


# ── Per-level statistics, computed once and filtered by every rule ───────────
@dataclass(frozen=True)
class Level:
    """One subgroup level of one demographic column."""

    n: int
    n_pos: int
    n_neg: int
    auc: float
    var: float


def level_stats(y: np.ndarray, s: np.ndarray, codes: np.ndarray) -> List[Level]:
    """Every estimable level of one demographic column.

    "Estimable" here is the weakest possible requirement -- two positives and two
    negatives -- exactly as ``recompute.null_reference._column_level_stats``
    defines it. The five inclusion rules are applied afterwards by
    :func:`admissible`, so one pass over the data serves the whole sweep and all
    five rules see literally the same AUROC computations.
    """
    out: List[Level] = []
    for lvl in np.unique(codes):
        mask = codes == lvl
        n = int(mask.sum())
        y_g = y[mask]
        n_pos = int(y_g.sum())
        n_neg = n - n_pos
        if n_pos < 2 or n_neg < 2:
            continue
        a, v = auc_delong(y_g, s[mask])
        if not np.isfinite(a):
            continue
        out.append(Level(n, n_pos, n_neg, float(a), float(v)))
    return out


# ── Vectorised kernel ────────────────────────────────────────────────────────
# :func:`level_stats` is the readable reference implementation and calls
# ``rankdata`` three times per subgroup. Inside a permutation loop that dominates
# the runtime, and the Type I error study runs 999 permutations inside each of
# 1000 simulated datasets inside each of 16 cells. The kernel below computes the
# identical quantities in O(n) per column, using the fact that the SCORES ARE
# FIXED across replicates -- only labels move -- so the sort can be hoisted out
# of the loop entirely.
#
# Ties are handled exactly, by aggregating over tie groups: every member of a tie
# group gets the same placement value, which is the midrank convention and gives
# tied pairs a contribution of one half. ``tests/test_comparators.py`` asserts
# bit-level agreement with :func:`level_stats` and with
# ``recompute.null_reference.partition_gaps_by_rule`` on real cohort data.
# Nothing is approximated; this is a speed transformation only.
class SortedCohort:
    """Score ordering hoisted out of the permutation loop."""

    __slots__ = ("order", "y", "gval", "n")

    def __init__(self, y: np.ndarray, s: np.ndarray):
        s = np.asarray(s, dtype=float)
        self.order = np.argsort(s, kind="stable")
        s_sorted = s[self.order]
        self.y = np.asarray(y).astype(np.int8)[self.order]
        # Dense integer id of each distinct score value, ascending.
        self.gval = np.cumsum(
            np.r_[True, s_sorted[1:] != s_sorted[:-1]]).astype(np.int64) - 1
        self.n = len(self.y)

    def sort_codes(self, codes: np.ndarray) -> np.ndarray:
        return np.asarray(codes)[self.order]


def fast_level_stats(sc: SortedCohort, codes_sorted: np.ndarray) -> List[Level]:
    """Same output as :func:`level_stats`, computed without sorting.

    ``codes_sorted`` must already be in score order (``sc.sort_codes(codes)``).
    """
    out: List[Level] = []
    y = sc.y
    for lvl in np.unique(codes_sorted):
        idx = np.flatnonzero(codes_sorted == lvl)
        n = idx.size
        yk = y[idx]
        n_pos = int(yk.sum())
        n_neg = n - n_pos
        if n_pos < 2 or n_neg < 2:
            continue
        gk = sc.gval[idx]
        new = np.empty(n, dtype=bool)
        new[0] = True
        np.not_equal(gk[1:], gk[:-1], out=new[1:])
        gid = np.cumsum(new) - 1
        ng = int(gid[-1]) + 1
        a = np.bincount(gid, weights=yk, minlength=ng)              # positives
        b = np.bincount(gid, weights=1 - yk, minlength=ng)          # negatives
        cb = np.cumsum(b)
        ca = np.cumsum(a)
        b_lt = cb - b                       # negatives strictly below the group
        a_gt = n_pos - ca                   # positives strictly above the group
        v10 = (b_lt + 0.5 * b) / n_neg
        v01 = (a_gt + 0.5 * a) / n_pos
        auc = float(np.dot(a, v10) / n_pos)
        s10 = float(np.dot(a, (v10 - auc) ** 2) / (n_pos - 1))
        s01 = float(np.dot(b, (v01 - auc) ** 2) / (n_neg - 1))
        var = s10 / n_pos + s01 / n_neg
        out.append(Level(n, n_pos, n_neg, auc, max(var, 0.0)))
    return out


def admissible(levels: Sequence[Level], rule: str) -> List[Level]:
    """Filter levels by one of the five published inclusion rules."""
    r = INCLUSION_RULES[rule]
    return [lv for lv in levels if _rule_admits(r, lv.n, lv.n_pos, lv.n_neg)]


def all_level_stats(y: np.ndarray, s: np.ndarray,
                    codes_by_col: Dict[str, np.ndarray]
                    ) -> Dict[str, List[Level]]:
    return {c: level_stats(y, s, codes) for c, codes in codes_by_col.items()}


def max_min_gap(levels: Sequence[Level]) -> float:
    """The incumbent's within-partition statistic: max AUROC minus min AUROC."""
    if len(levels) < 2:
        return float("nan")
    a = [lv.auc for lv in levels]
    return float(max(a) - min(a))


# ── Cohort bundle ────────────────────────────────────────────────────────────
@dataclass
class CohortData:
    """Everything the comparators need, loaded exactly once per cohort."""

    name: str
    label: str
    is_clinical: bool
    y: np.ndarray
    s: np.ndarray
    codes_by_col: Dict[str, np.ndarray]
    pos_idx: np.ndarray
    neg_idx: np.ndarray
    n_test: int
    prevalence: float
    load_runtime_s: float


def load_cohort(name: str) -> CohortData:
    """Load one cohort through the same loader the incumbent uses.

    ``recompute.cohorts.LOADERS`` fixes the split (``test_size=0.2``,
    ``random_state=42``, stratified) and returns an already-fitted model, so the
    scores here are bit-for-bit the scores the published null was computed on.
    """
    from recompute.cohorts import LOADERS

    t0 = time.perf_counter()
    bundle = LOADERS[name]()
    X_te = np.asarray(bundle["X_test"], dtype=float)
    y = np.asarray(bundle["y_test"]).astype(int)
    s = bundle["model"].predict_proba(X_te)[:, 1]
    cols: List[str] = list(bundle["subgroup_columns"])
    codes_by_col = code_columns(bundle["demo_test"], cols)
    load_s = time.perf_counter() - t0

    return CohortData(
        name=name,
        label=COHORT_LABELS.get(name, name),
        is_clinical=name in CLINICAL,
        y=y,
        s=s,
        codes_by_col=codes_by_col,
        pos_idx=np.flatnonzero(y == 1),
        neg_idx=np.flatnonzero(y == 0),
        n_test=int(len(y)),
        prevalence=float(y.mean()),
        load_runtime_s=load_s,
    )


class PermContext:
    """Everything needed to run a stratified permutation loop efficiently.

    The permutation itself is still drawn by the incumbent's own
    :func:`recompute.null_reference.draw_permuted_codes`, in the original row
    order and from the caller's generator, so a comparator driven through this
    context consumes exactly the draws the incumbent would. Only the *evaluation*
    of each draw is accelerated, by the sort-hoisting kernel above.
    """

    __slots__ = ("sc", "codes_by_col", "pos_idx", "neg_idx", "y", "s")

    def __init__(self, y: np.ndarray, s: np.ndarray,
                 codes_by_col: Dict[str, np.ndarray]):
        self.y = np.asarray(y).astype(int)
        self.s = np.asarray(s, dtype=float)
        self.codes_by_col = codes_by_col
        self.sc = SortedCohort(self.y, self.s)
        self.pos_idx = np.flatnonzero(self.y == 1)
        self.neg_idx = np.flatnonzero(self.y == 0)

    def observed(self) -> Dict[str, List[Level]]:
        return {c: fast_level_stats(self.sc, self.sc.sort_codes(codes))
                for c, codes in self.codes_by_col.items()}

    def draw(self, rng: np.random.Generator, scheme: str = "joint"
             ) -> Dict[str, List[Level]]:
        permuted = draw_permuted_codes(self.codes_by_col, self.pos_idx,
                                       self.neg_idx, rng, scheme=scheme)
        return {c: fast_level_stats(self.sc, self.sc.sort_codes(codes))
                for c, codes in permuted.items()}


def gap_from_levels(levels_by_col: Dict[str, List[Level]], rule: str) -> float:
    """Max over partitions of the within-partition max-min gap, under one rule.

    Equivalent to ``recompute.null_reference.partition_gaps_by_rule(...)[rule]``;
    the test suite asserts that on real cohort data and on random data with ties.
    """
    best = float("nan")
    for lv in levels_by_col.values():
        g = max_min_gap(admissible(lv, rule))
        if np.isfinite(g):
            best = g if not np.isfinite(best) else max(best, g)
    return best


def permutation_draws(data: CohortData, n_perm: int, seed: int = 42,
                      scheme: str = "joint"):
    """Yield ``n_perm`` permuted demographic assignments.

    Delegates to the incumbent's own :func:`draw_permuted_codes` with the same
    seed and the same scheme, so a comparator's null is built on the *same*
    random draws as the incumbent's. Any difference between them is then the
    statistic, never the randomness.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        yield draw_permuted_codes(
            data.codes_by_col, data.pos_idx, data.neg_idx, rng, scheme=scheme)


# ── Multiplicity ─────────────────────────────────────────────────────────────
def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (FWER control, any dependence)."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    out = np.full(p.shape, np.nan)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for i, j in enumerate(order):
        running = max(running, (m - i) * p[j])
        out[j] = min(running, 1.0)
    return out


def mc_p(null_vals: np.ndarray, observed: float) -> Tuple[float, bool]:
    """One-sided Monte-Carlo p-value with the +1 correction, and a floor flag.

    Matches :func:`recompute.null_reference.mc_pvalue` so that a comparator's
    p-value and the incumbent's are on exactly the same scale, including the
    hard floor at 1/(B+1).
    """
    v = np.asarray(null_vals, dtype=float)
    v = v[np.isfinite(v)]
    b = len(v)
    if b == 0 or not np.isfinite(observed):
        return float("nan"), False
    n_ge = int(np.sum(v >= observed))
    return float((n_ge + 1) / (b + 1)), bool(n_ge == 0)


def p_report(p: Optional[float], n_perm: int, is_floor: bool,
             digits: int = 4) -> str:
    """Render a Monte-Carlo p-value, never as an attained value at the floor.

    The +1 correction that makes the permutation test exact-valid also puts a
    hard floor at ``1 / (B + 1)``: with B = 10,000 no p-value below 9.999e-05 can
    be produced, whatever the data. Printing that as "0.000" -- or worse, letting
    a downstream table round it to zero -- states an attained precision the
    design cannot deliver and invites a reader to treat it as overwhelming
    evidence. At the floor the correct statement is an inequality.

    Matches ``recompute.null_reference.mc_pvalue``'s ``p_report`` field so the
    comparator tables and the incumbent's tables render the same value the same
    way.
    """
    if p is None or not np.isfinite(p):
        return "n/a"
    floor = 1.0 / (n_perm + 1)
    if is_floor or p <= floor:
        return f"<= {floor:.2e}"
    return f"{p:.{digits}f}"


# ── Result record shared by every comparator ─────────────────────────────────
@dataclass
class MethodResult:
    """One (cohort, rule, method) cell of the comparison table."""

    method: str
    rule: str
    conclusion: str                      # flag | no_flag | not_evaluable
    statistic: Optional[float] = None
    statistic_name: str = ""
    p_value: Optional[float] = None
    p_is_floor: bool = False
    runtime_s: float = 0.0
    detail: str = ""
    #: Number of permutation replicates behind ``p_value``; sets the 1/(B+1)
    #: floor. ``None`` for closed-form and deterministic methods, which have no
    #: floor.
    n_perm: Optional[int] = None

    @property
    def p_value_report(self) -> str:
        """The p-value as it must be printed: an inequality when at the floor."""
        if self.p_value is None or self.n_perm is None:
            return ("n/a" if self.p_value is None
                    else f"{float(self.p_value):.4g}")
        return p_report(self.p_value, int(self.n_perm), self.p_is_floor)

    def as_row(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "rule": self.rule,
            "conclusion": self.conclusion,
            "statistic_name": self.statistic_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "p_is_floor": self.p_is_floor,
            # Never let a downstream table round a floored p-value to 0.000.
            "p_value_report": self.p_value_report,
            "p_floor": (1.0 / (self.n_perm + 1)
                        if self.n_perm is not None else None),
            "runtime_s": self.runtime_s,
            "detail": self.detail,
        }


NOT_EVALUABLE = "not_evaluable"
FLAG = "flag"
NO_FLAG = "no_flag"
