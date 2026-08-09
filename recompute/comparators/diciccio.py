"""
DiCiccio, Vasudevan, Basu, Kenthapadi & Tomkins (2020, KDD),
"Evaluating Fairness Using Permutation Tests".

What the method is
------------------
A permutation test that a model is fair across **two** groups with respect to an
arbitrary metric. The paper's central point is that the *naive* permutation test
-- permute the group label, recompute the raw metric difference -- is invalid for
this problem. Permutation inference is exact only under full exchangeability,
i.e. only when the two groups' entire score distributions coincide. The null a
fairness audit actually cares about is far weaker and *composite*: the two groups
have equal metric values, while their score distributions may differ arbitrarily.
Under that composite null the naive permutation distribution is not the sampling
distribution of the statistic, and the test's Type I error can be badly wrong.

The fix is **studentization**. Replace the raw difference by

    T = (theta_a - theta_b) / sqrt( Var(theta_a) + Var(theta_b) )

which is asymptotically pivotal (standard normal) whatever the two score
distributions are. A permutation distribution of an asymptotically pivotal
statistic converges to that same pivotal limit, so the permutation test recovers
asymptotic validity under the composite null while retaining finite-sample
exactness under the stronger null. This is the Janssen (1997) / Chung & Romano
(2013) studentized-permutation argument, applied by DiCiccio et al. to fairness
metrics. The studentization is the whole point of the paper and is implemented
here literally.

How it is applied here
----------------------
The metric is subgroup AUROC and ``Var`` is DeLong's variance
(:func:`recompute.comparators.core.auc_delong`). The two subgroups are disjoint
samples, so their variances add.

**Two adaptations, both stated explicitly.**

1. *Stratified permutation.* DiCiccio et al. permute the group label freely. For
   AUROC that would resample each group's positive and negative counts, so a
   permuted "group" could contain almost no events and its AUROC would not be
   estimable. We permute the group label *within outcome class*, which holds
   every subgroup's size and prevalence fixed. This is the same stratification
   the incumbent uses, so the two procedures consume identical permutation draws
   (``scheme="joint"``, seed 42, B = 10,000) and differ in the statistic alone.
   It is also the only stratification under which the AUROC comparison is well
   posed.

2. *Extension from two groups to many.* The paper tests one pair. A demographic
   partition here has up to eight levels and a cohort has up to five partitions.
   We compute the studentized statistic for every admissible pair within every
   partition and combine by **single-step max-T**: the cohort statistic is
   ``max |T|`` over all pairs, and its null is the distribution of that same
   maximum over the same permutation draws. This is the Westfall-Young
   permutation max-T combination. It tests exactly the incumbent's hypothesis
   ("does any subgroup pair differ?"), controls the family-wise error rate over
   all pairs, and -- unlike Bonferroni -- exploits the dependence between pairs
   rather than paying for it. It is the strongest valid combination available,
   which is what the method is owed.

   Holm-adjusted per-pair p-values (permutation and asymptotic-normal) are
   reported alongside as a conservative cross-check.

Note on level identity across replicates
----------------------------------------
The stratified permutation preserves every level's n_pos and n_neg exactly, so a
level admissible in the observed data is admissible in every replicate and pair
``(i, j)`` of partition ``c`` means the same thing in every replicate. That is
what makes per-pair marginal permutation p-values well defined here.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from recompute.comparators.core import (
    FLAG,
    NO_FLAG,
    NOT_EVALUABLE,
    RULE_NAMES,
    VAR_FLOOR,
    CohortData,
    Level,
    MethodResult,
    PermContext,
    admissible,
    holm,
    mc_p,
    p_report,
)

METHOD = "diciccio2020"


# ── The statistic ────────────────────────────────────────────────────────────
def studentized(a: Level, b: Level) -> float:
    """|AUC_a - AUC_b| / sqrt(Var_a + Var_b); nan when the denominator vanishes.

    A zero denominator means both subgroups have perfectly degenerate placement
    values (every positive strictly above every negative, say), so the metric
    carries no sampling variability and the studentized statistic is undefined.
    Those pairs are dropped rather than given an infinite statistic.
    """
    denom = a.var + b.var
    if not np.isfinite(denom) or denom <= VAR_FLOOR:
        return float("nan")
    return float(abs(a.auc - b.auc) / np.sqrt(denom))


def _pair_index(levels_by_col: Dict[str, List[Level]]
                ) -> List[Tuple[str, int, int]]:
    """Enumerate (column, i, j) for every pair of estimable levels."""
    pairs: List[Tuple[str, int, int]] = []
    for col, lv in levels_by_col.items():
        for i in range(len(lv)):
            for j in range(i + 1, len(lv)):
                pairs.append((col, i, j))
    return pairs


def _pair_stats(levels_by_col: Dict[str, List[Level]],
                pairs: List[Tuple[str, int, int]]) -> np.ndarray:
    """Studentized statistic for every enumerated pair, in ``pairs`` order."""
    out = np.empty(len(pairs), dtype=float)
    for k, (col, i, j) in enumerate(pairs):
        lv = levels_by_col[col]
        out[k] = studentized(lv[i], lv[j])
    return out


def _admissible_flags(levels: List[Level], rule: str) -> np.ndarray:
    from recompute.null_reference import INCLUSION_RULES, _rule_admits

    r = INCLUSION_RULES[rule]
    return np.array([_rule_admits(r, lv.n, lv.n_pos, lv.n_neg) for lv in levels],
                    dtype=bool)


def rule_masks(levels_by_col: Dict[str, List[Level]],
               pairs: List[Tuple[str, int, int]],
               rules: List[str]) -> Dict[str, np.ndarray]:
    """Which enumerated pairs each inclusion rule admits.

    A pair counts under a rule only when *both* of its levels pass the rule.
    Because the stratified permutation preserves every level's (n, n_pos, n_neg),
    this mask is identical in every replicate and is computed once.
    """
    flags = {rule: {c: _admissible_flags(lv, rule)
                    for c, lv in levels_by_col.items()}
             for rule in rules}
    masks: Dict[str, np.ndarray] = {}
    for rule in rules:
        f = flags[rule]
        masks[rule] = np.array([f[c][i] and f[c][j] for (c, i, j) in pairs],
                               dtype=bool)
    return masks


# ── Full cohort run ──────────────────────────────────────────────────────────
def run_cohort(data: CohortData, n_perm: int = 10_000, seed: int = 42,
               rules: Optional[List[str]] = None, alpha: float = 0.05,
               scheme: str = "joint") -> Dict[str, object]:
    """Studentized permutation test on one cohort, for every inclusion rule.

    All rules share one permutation pass, exactly as the incumbent's sweep does,
    so the rules are perfectly paired and the sweep is nearly free.
    """
    rules = list(rules) if rules is not None else list(RULE_NAMES)
    t0 = time.perf_counter()

    ctx = PermContext(data.y, data.s, data.codes_by_col)
    obs_levels = ctx.observed()
    pairs = _pair_index(obs_levels)
    if not pairs:
        rt = time.perf_counter() - t0
        return {
            "pairs": [],
            "results": {r: MethodResult(
                METHOD, r, NOT_EVALUABLE, runtime_s=rt,
                statistic_name="max |T|",
                detail="no partition has two estimable levels") for r in rules},
            "runtime_s": rt,
        }

    masks = rule_masks(obs_levels, pairs, rules)
    t_obs = _pair_stats(obs_levels, pairs)
    # VAR_FLOOR accounting. `studentized` returns nan when Var_a + Var_b <=
    # VAR_FLOOR, i.e. when the pair carries no estimable sampling variability and
    # the studentized statistic is undefined. Those pairs are silently dropped
    # from the max-T family, which changes the family the FWER is controlled over
    # -- so the count has to be reported, not merely handled. It is per rule
    # because each rule admits a different set of pairs.
    var_floor_drops = {
        rule: int((masks[rule] & ~np.isfinite(t_obs)).sum()) for rule in rules}

    # Permutation pass: one row permutation per replicate, carried across every
    # demographic column (the incumbent's joint scheme, same seed).
    t_null = np.full((len(pairs), n_perm), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    for b in range(n_perm):
        t_null[:, b] = _pair_stats(ctx.draw(rng, scheme), pairs)
    perm_runtime = time.perf_counter() - t0

    results: Dict[str, MethodResult] = {}
    diagnostics: Dict[str, Dict[str, object]] = {}
    # Permuted replicates can also lose pairs to the variance floor even when the
    # observed pair survives; counted so the null's family size is auditable too.
    null_floor_drops = {
        rule: int((~np.isfinite(t_null[masks[rule], :])).sum()) for rule in rules}
    for rule in rules:
        m = masks[rule] & np.isfinite(t_obs)
        n_admissible = int(masks[rule].sum())
        if m.sum() == 0:
            results[rule] = MethodResult(
                METHOD, rule, NOT_EVALUABLE, runtime_s=perm_runtime,
                statistic_name="max |T|",
                detail=(f"no admissible pair with a defined studentized "
                        f"statistic; n_pairs_admissible={n_admissible}, "
                        f"n_pairs_dropped_var_floor={var_floor_drops[rule]}"))
            continue

        obs_max = float(np.nanmax(t_obs[m]))
        with np.errstate(invalid="ignore"):
            null_max = np.nanmax(t_null[m, :], axis=0)
        p_maxt, is_floor = mc_p(null_max, obs_max)

        # Conservative cross-checks.
        per_pair_p = np.array([
            mc_p(t_null[k, :], t_obs[k])[0] for k in np.flatnonzero(m)])
        p_holm_perm = float(np.nanmin(holm(per_pair_p)))
        p_asym = 2.0 * norm.sf(np.abs(t_obs[m]))
        p_holm_asym = float(np.nanmin(holm(p_asym)))

        results[rule] = MethodResult(
            method=METHOD,
            rule=rule,
            conclusion=FLAG if p_maxt < alpha else NO_FLAG,
            statistic=obs_max,
            statistic_name="max |T| (studentized)",
            p_value=p_maxt,
            p_is_floor=is_floor,
            n_perm=n_perm,
            runtime_s=perm_runtime,
            detail=(f"n_pairs={int(m.sum())}; "
                    f"n_pairs_dropped_var_floor={var_floor_drops[rule]}; "
                    f"p_report={p_report(p_maxt, n_perm, is_floor)}; "
                    f"p_holm_perm={p_holm_perm:.4g}; "
                    f"p_holm_asymptotic={p_holm_asym:.4g}"),
        )
        k_arg = int(np.flatnonzero(m)[int(np.nanargmax(t_obs[m]))])
        diagnostics[rule] = {
            "n_pairs": int(m.sum()),
            "n_pairs_admissible": n_admissible,
            # Pairs the variance floor removed from the max-T family, observed
            # and (summed over replicates) in the null. VAR_FLOOR = 1e-12; a pair
            # is dropped when Var_a + Var_b <= that, which happens when both
            # subgroups have perfectly degenerate placement values.
            "n_pairs_dropped_var_floor": var_floor_drops[rule],
            "frac_pairs_dropped_var_floor": (
                var_floor_drops[rule] / n_admissible if n_admissible else 0.0),
            "n_null_pair_evaluations_dropped_var_floor": null_floor_drops[rule],
            "n_null_pair_evaluations": int(n_admissible * n_perm),
            "max_pair": pairs[k_arg],
            "p_maxT": p_maxt,
            "p_maxT_report": p_report(p_maxt, n_perm, is_floor),
            "p_is_floor": bool(is_floor),
            "p_floor_value": 1.0 / (n_perm + 1),
            "p_holm_permutation": p_holm_perm,
            "p_holm_asymptotic": p_holm_asym,
            "null_max_p95": float(np.nanpercentile(null_max, 95)),
        }

    return {
        "pairs": pairs,
        "results": results,
        "diagnostics": diagnostics,
        "runtime_s": perm_runtime,
        "n_perm": n_perm,
        "seed": seed,
    }


# ── Lightweight path used by the Type I error simulation ─────────────────────
def pvalue_only(ctx: PermContext, rule: str, n_perm: int,
                rng: np.random.Generator) -> float:
    """max-T permutation p-value for one simulated dataset.

    Same statistic and same stratified joint permutation as
    :func:`run_cohort`; separated only so the simulation can drive it with its
    own generator and its own (smaller) B without touching the cohort path.
    """
    obs_levels = {c: admissible(lv, rule) for c, lv in ctx.observed().items()}
    pairs = _pair_index(obs_levels)
    if not pairs:
        return float("nan")
    t_obs = _pair_stats(obs_levels, pairs)
    if not np.isfinite(t_obs).any():
        return float("nan")
    obs_max = float(np.nanmax(t_obs))

    null_max = np.full(n_perm, np.nan)
    for b in range(n_perm):
        lv = {c: admissible(x, rule) for c, x in ctx.draw(rng).items()}
        vals = _pair_stats(lv, pairs)
        if np.isfinite(vals).any():
            null_max[b] = np.nanmax(vals)
    return mc_p(null_max, obs_max)[0]
