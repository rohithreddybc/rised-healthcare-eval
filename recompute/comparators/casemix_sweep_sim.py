"""
Cell runner for the case-mix SD-ratio sweep and the positive controls.

It reuses :func:`recompute.comparators.type1._one_sim` verbatim, so the five
procedures see exactly the code path -- and, for a given ``(geometry, replicate,
seed)``, exactly the data -- that the published Type I table was produced with.
Three things are added on top, all of them diagnostics rather than alterations:

**Per-replicate statistics are kept.** The six-geometry case-mix study
(:mod:`recompute.comparators.simulate`) aggregated each cell to a flag rate and
discarded everything else. The question "can any procedure distinguish case mix
from unfairness" cannot be answered from two saturated flag rates (0.92 against
0.92 is not evidence of *anything*); it needs the underlying statistic, replicate
by replicate, on both arms. :func:`discrimination_auc` then asks the only
question that matters: given one dataset from each arm, how often does the
statistic order them correctly? 0.5 means the procedure carries no information
about the mechanism at all.

**Level drops are counted.** The six-geometry case-mix study assumed that m30
and ev10 give identical numbers on every case-mix geometry, "as they must: both
admit every level there". That is false for ``casemix_location_3``, whose level
0 has true prevalence 0.0223 -- about 15 expected events in 668 rows -- and which
ev10 therefore drops in a few percent of replicates. The dropped level is the one
with the *highest* true AUROC, i.e. the one generating the gap, so the inclusion
rule is silently interacting with the geometry in the very cell built to isolate
location from spread. ``n_admissible_p0`` records it per replicate.

**Two things that are not among the five procedures are measured anyway.** Both
of the added diagnostics look at *calibration*, which no procedure in the study
looks at:

``calibration_cox``
    Cox's two-degree-of-freedom recalibration test, ``y ~ a + b logit(s)`` within
    each level against ``(a, b) = (0, 1)``, Holm-adjusted over levels. A fair,
    correctly specified model passes in every subgroup; a model that is worse for
    one subgroup fails there.
``mbc_excess``
    van Klaveren et al. (2016) model-based concordance minus observed AUROC, per
    level, maximised over levels. Under case mix with a calibrated model the two
    agree, so the statistic is ~0 whatever the AUROC gap; under genuine
    unfairness the model's own predictions imply a discrimination it does not
    deliver, and the statistic is positive.

They are reported as *diagnostics*, clearly separated from the five procedures.
Neither is a recommendation: both assume within-subgroup calibration is
achievable and checkable, which on a real cohort with 60 events it is not.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from recompute.comparators.core import REPO

RESULTS = REPO / "recompute" / "results"
CELL_DIR = RESULTS / "casemix_sweep_cells"

#: The five procedures under study, plus the two secondary Lum readings, exactly
#: as :mod:`recompute.comparators.type1` names them.
METHODS = ("permutation_null", "diciccio2020", "lum2022",
           "four_fifths", "fixed_threshold_005",
           "lum2022_cochranQ", "lum2022_bootstrapCI")

#: Added diagnostics. NOT part of the five; reported separately everywhere.
DIAGNOSTICS = ("calibration_cox", "mbc_excess")

#: Per-replicate statistics kept for the discrimination analysis. Each maps to
#: the direction in which "more evidence of a problem" points.
STATISTIC_SIGN = {
    "permutation_null__p": -1,
    "diciccio2020__p": -1,
    "lum2022__p": -1,
    "maxmin_gap": +1,
    "four_fifths_ratio": -1,
    "calibration_cox__p": -1,
    "mbc_excess": +1,
}


# ── model-based concordance ──────────────────────────────────────────────────
def model_based_concordance(p: np.ndarray) -> float:
    """van Klaveren et al. (2016) model-based concordance of a risk vector.

    The c-statistic the model *implies* for the case mix it was applied to, if
    the model were correct: treat each subject's predicted risk ``p_i`` as the
    true event probability, and average concordance over every ordered pair
    weighted by the probability that the pair is informative,

        mbc = sum_{i != j} p_i (1 - p_j) [ 1(p_i > p_j) + 0.5 * 1(p_i = p_j) ]
              / sum_{i != j} p_i (1 - p_j)

    This is the population quantity that ``simulate.true_subgroup_auc`` computes
    by quadrature, estimated from predictions alone. Comparing it with the
    *observed* AUROC is the whole point: they agree when the gap between two
    subgroups is a consequence of their differing case mix, and disagree when the
    model is genuinely delivering less than its own predictions imply.

    Computed in O(n log n) by aggregating over tie groups rather than the O(n^2)
    double sum, so it is affordable inside a 1,000-replicate loop and on a
    20,000-row cohort.
    """
    p = np.asarray(p, dtype=float)
    p = p[np.isfinite(p)]
    if p.size < 2:
        return float("nan")
    q = 1.0 - p
    order = np.argsort(p, kind="stable")
    ps, qs = p[order], q[order]
    new = np.r_[True, ps[1:] != ps[:-1]]
    gid = np.cumsum(new) - 1
    ng = int(gid[-1]) + 1
    A = np.bincount(gid, weights=ps, minlength=ng)       # sum of p in tie group
    B = np.bincount(gid, weights=qs, minlength=ng)       # sum of 1-p in group
    S = np.bincount(gid, weights=ps * qs, minlength=ng)  # the i == j terms
    b_below = np.cumsum(B) - B
    num = float(np.dot(A, b_below) + 0.5 * float(np.sum(A * B - S)))
    den = float(p.sum() * q.sum() - float(np.sum(ps * qs)))
    return num / den if den > 0 else float("nan")


# ── Cox two-parameter recalibration test ─────────────────────────────────────
def cox_calibration_test(y: np.ndarray, s: np.ndarray,
                         eps: float = 1e-12) -> Tuple[float, float, float]:
    """``(p_value, intercept, slope)`` of ``y ~ a + b logit(s)`` against (0, 1).

    Likelihood-ratio test on 2 degrees of freedom against the model that already
    has the right answer, i.e. the offset model ``logit(pi) = logit(s)``. Fitted
    by Newton-Raphson, which converges in a handful of steps on two parameters
    and needs no external dependency inside the simulation loop.
    """
    from scipy.stats import chi2

    y = np.asarray(y, dtype=float)
    s = np.clip(np.asarray(s, dtype=float), eps, 1.0 - eps)
    x = np.log(s) - np.log1p(-s)
    n = y.size
    if n < 10 or y.sum() < 2 or (n - y.sum()) < 2 or np.ptp(x) <= 0:
        return float("nan"), float("nan"), float("nan")

    X = np.column_stack([np.ones(n), x])
    beta = np.array([0.0, 1.0])
    for _ in range(60):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        grad = X.T @ (y - mu)
        hess = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return float("nan"), float("nan"), float("nan")
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        return float("nan"), float("nan"), float("nan")

    def loglik(eta: np.ndarray) -> float:
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    stat = 2.0 * (loglik(X @ beta) - loglik(x))
    stat = max(stat, 0.0)
    return float(chi2.sf(stat, 2)), float(beta[0]), float(beta[1])


# ── one replicate ────────────────────────────────────────────────────────────
def _one_replicate(geom, rep: int, rule: str, n_perm: int, seed: int,
                   alpha: float) -> Dict[str, float]:
    """The five procedures plus the two calibration diagnostics, on one dataset."""
    from recompute.comparators.core import (
        admissible,
        auc_delong,
        holm,
        max_min_gap,
    )
    from recompute.comparators.simulate import make_dataset
    from recompute.comparators.type1 import _one_sim

    out = dict(_one_sim(geom, rep, rule, n_perm, seed, alpha))

    y, s, codes = make_dataset(geom, rep, seed)

    # Level bookkeeping on partition 0 -- what the inclusion rule actually kept.
    from recompute.comparators.core import PermContext

    ctx = PermContext(y, s, codes)
    obs = ctx.observed()
    lv0 = obs["p0"]
    adm0 = admissible(lv0, rule)
    out["n_levels_p0"] = float(len(geom.partitions[0]))
    out["n_estimable_p0"] = float(len(lv0))
    out["n_admissible_p0"] = float(len(adm0))
    out["maxmin_gap"] = float(max_min_gap(adm0))
    aucs = [lvl.auc for lvl in adm0]
    out["four_fifths_ratio"] = (float(min(aucs) / max(aucs))
                                if len(aucs) >= 2 and max(aucs) > 0
                                else float("nan"))

    # ── diagnostics: calibration, on the same admissible levels ──────────────
    pvals: List[float] = []
    excess: List[float] = []
    for k in np.unique(codes["p0"]):
        m = codes["p0"] == k
        n_pos = int(y[m].sum())
        n_neg = int(m.sum() - n_pos)
        if n_pos < 2 or n_neg < 2:
            continue
        from recompute.null_reference import INCLUSION_RULES, _rule_admits

        if not _rule_admits(INCLUSION_RULES[rule], int(m.sum()), n_pos, n_neg):
            continue
        p_cal, _, _ = cox_calibration_test(y[m], s[m])
        if np.isfinite(p_cal):
            pvals.append(p_cal)
        obs_auc = auc_delong(y[m], s[m])[0]
        mbc = model_based_concordance(s[m])
        if np.isfinite(obs_auc) and np.isfinite(mbc):
            excess.append(mbc - obs_auc)

    if pvals:
        adj = holm(pvals)
        out["calibration_cox__p"] = float(np.nanmin(adj))
        out["calibration_cox"] = float(np.nanmin(adj) < alpha)
    else:
        out["calibration_cox__p"] = float("nan")
        out["calibration_cox"] = float("nan")
    out["mbc_excess"] = float(max(excess)) if excess else float("nan")
    return out


# ── one cell ─────────────────────────────────────────────────────────────────
def cell_path(geom_name: str, rule: str, n_sims: int, n_perm: int) -> Path:
    return CELL_DIR / f"{geom_name}__{rule}__{n_sims}_{n_perm}.json"


def run_cell(args) -> Dict[str, object]:
    """Run one ``(geometry, rule)`` cell and checkpoint it.

    ``args`` is ``(geometry, rule, n_sims, n_perm, seed, alpha, force)``. The
    geometry is passed as an object rather than a name so that the sweep's own
    geometry registries do not have to be merged into
    ``simulate.GEOMETRY_BY_NAME`` and the published 23-geometry table stays
    exactly the table it is.
    """
    geom, rule, n_sims, n_perm, seed, alpha, force = args
    path = cell_path(geom.name, rule, n_sims, n_perm)
    if path.exists() and not force:
        try:
            print(f"  [cached] {geom.name} / {rule}", flush=True)
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass

    t0 = time.perf_counter()
    sims = [_one_replicate(geom, r, rule, n_perm, seed, alpha)
            for r in range(n_sims)]
    wall = time.perf_counter() - t0

    from recompute.comparators.simulate import geometry_seed_word, true_subgroup_auc

    tsa = true_subgroup_auc(geom)
    cm = geom.case_mix
    n_lv = len(cm.scales) if cm is not None else 0
    rates: Dict[str, object] = {}
    for m in METHODS + DIAGNOSTICS:
        vals = np.array([sd.get(m, np.nan) for sd in sims], dtype=float)
        ok = np.isfinite(vals)
        n_ok = int(ok.sum())
        rate = float(vals[ok].mean()) if n_ok else float("nan")
        rates[m] = {
            "n_evaluable": n_ok,
            "n_flag": int(np.nansum(vals)),
            "flag_rate": rate,
            "mc_se": (float(np.sqrt(rate * (1 - rate) / n_ok))
                      if n_ok else float("nan")),
        }
    stats = {k: [float(sd.get(k, np.nan)) for sd in sims]
             for k in STATISTIC_SIGN}
    adm = np.array([sd["n_admissible_p0"] for sd in sims], dtype=float)

    row: Dict[str, object] = {
        "geometry": geom.name,
        "description": geom.description,
        "n": geom.n,
        "prevalence": geom.prevalence,
        "n_partitions": len(geom.partitions),
        "n_levels": n_lv,
        "lp_dist": cm.lp_dist if cm is not None else "",
        "equalize_prevalence": bool(cm.equalize_prevalence) if cm else False,
        "sd_ratio": float(cm.sd_ratio) if cm is not None else float("nan"),
        "scales": list(cm.scales) if cm is not None else [],
        "unfair_w": list(cm.unfair_w) if (cm and cm.unfair_w) else [],
        "miscal_slope": list(cm.miscal_slope) if (cm and cm.miscal_slope) else [],
        "miscal_intercept": (list(cm.miscal_intercept)
                             if (cm and cm.miscal_intercept) else []),
        "true_auc_by_level": [tsa.get(f"level_{k}") for k in range(n_lv)],
        "oracle_auc_by_level": [tsa.get(f"oracle_{k}") for k in range(n_lv)],
        "prevalence_by_level": [tsa.get(f"prev_{k}") for k in range(n_lv)],
        "true_auc_gap": float(tsa.get("max_gap", 0.0)),
        "true_max_excess_auc": float(tsa.get("max_excess", 0.0)),
        "prevalence_ratio": float(tsa.get("prev_ratio", 1.0)),
        "rule": rule,
        "n_sims": n_sims,
        "n_perm": n_perm,
        "alpha": alpha,
        "seed": seed,
        "geometry_seed_word": geometry_seed_word(geom.name),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "wall_s": wall,
        "mean_n_admissible_p0": float(np.mean(adm)),
        "frac_reps_with_level_dropped": float(np.mean(adm < n_lv)),
        "rates": rates,
        "statistics": stats,
    }
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row), encoding="utf-8")
    tmp.replace(path)
    print(f"  [done] {geom.name} / {rule}  ({wall/60:.1f} min)", flush=True)
    return row


# ── discrimination between two arms ──────────────────────────────────────────
def discrimination_auc(a: Sequence[float], b: Sequence[float]
                       ) -> Tuple[float, float]:
    """``(AUC, se)`` for separating arm ``b`` from arm ``a`` by a statistic.

    Mann-Whitney with midranks for ties, so a statistic that is constant on both
    arms scores exactly 0.5 rather than being undefined. The standard error is
    the DeLong estimate. 0.5 means: shown one dataset generated by case mix and
    one by genuine unfairness, with the same true AUROC gap, this statistic
    orders them no better than a coin.
    """
    from recompute.comparators.core import auc_delong

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok_a = a[np.isfinite(a)]
    ok_b = b[np.isfinite(b)]
    if ok_a.size < 2 or ok_b.size < 2:
        return float("nan"), float("nan")
    y = np.r_[np.zeros(ok_a.size, int), np.ones(ok_b.size, int)]
    v = np.r_[ok_a, ok_b]
    if np.ptp(v) == 0:
        return 0.5, 0.0
    auc, var = auc_delong(y, v)
    return float(auc), float(np.sqrt(var)) if np.isfinite(var) else float("nan")
