"""
The manuscript's own recommended replacement metrics, measured.

Why this module exists
----------------------
``recompute.comparators.type1`` shows that under case mix -- one shared model,
identical coefficients, Bayes-optimal, perfectly calibrated in every subgroup,
differing only in covariate distribution -- every subgroup AUROC-gap procedure
flags the perfectly fair model at high rates. The manuscript concludes that the
equal-AUROC null is the wrong estimand and recommends **subgroup calibration**
and **subgroup net benefit** instead.

That recommendation was never evaluated. This module evaluates it, on the *same*
geometries, the *same* seeds and the *same* draws, so that the comparison with
the AUROC result is like-for-like rather than merely adjacent.

What is measured
----------------
Per subgroup, and then as a max-over-subgroups gap (the same statistic shape the
AUROC study indicts):

``citl``        Calibration-in-the-large: the intercept ``a`` of the offset model
                ``logit(P(y=1)) = a + logit(s)``. Perfect calibration => 0.
``mean_cal``    The moment form of the same thing, ``mean(y) - mean(s)`` (the
                O-E difference on the probability scale). Perfect => 0. Reported
                alongside ``citl`` because it is the version a permutation test
                can afford and because it is what most clinical papers plot.
``cal_slope``   The slope ``b`` of ``logit(P(y=1)) = a' + b * logit(s)``.
                Perfect calibration => 1.
``ece``         Expected calibration error, 10 equal-width bins. Reported ONLY
                with its finite-sample bias, which is not a nuisance here but the
                dominant term: under the case-mix geometries the score IS the
                true probability, so the true ECE is exactly zero and every
                reported value is pure estimation bias (Nixon et al. 2019;
                Roelofs et al. 2022). ``ece_null`` gives the same quantity
                computed on labels redrawn as ``y* ~ Bernoulli(s)``, which is the
                bias floor at that subgroup's size and score distribution
                whatever the truth, and is therefore the reference for the
                geometries where the truth is not zero.
``nb_t``        Net benefit at threshold ``t``:
                ``TP/n - (FP/n) * t/(1-t)`` (Vickers & Elkin 2006).
``snb_t``       Standardised net benefit, ``nb_t / prevalence`` -- net benefit
                expressed as a fraction of the maximum attainable, which is the
                form recommended for comparing populations *because* it divides
                out one power of the prevalence. It does not divide out the
                others; see :func:`true_case_mix_metrics`.

The essential asymmetry
-----------------------
Calibration and net benefit are not in the same position under case mix, and the
distinction is the whole result:

* Calibration is a property of the *conditional* law ``P(y | s)``. The case-mix
  construction fixes ``s = P(y = 1 | x)`` exactly, so that conditional law is the
  same in every subgroup by construction. The true subgroup calibration gap is
  therefore exactly zero and any observed gap is finite-sample noise.

* Net benefit is a property of the *joint* law of ``(y, s)``. It depends on where
  the subgroup's score mass sits relative to the threshold and on the subgroup's
  prevalence -- both of which the case-mix construction deliberately varies.
  The true subgroup net-benefit gap is therefore **not** zero. It is computed
  exactly by :func:`true_case_mix_metrics`, and on these geometries it is
  *larger* than the true AUROC gap the manuscript indicts.

Everything below is a deterministic function of ``(geometry, replicate, seed)``
through :func:`recompute.comparators.simulate.make_dataset`; nothing here draws
its own data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from recompute.comparators.simulate import (
    Geometry,
    _expit,
    _mixture_weights,
    case_mix_intercept,
)
from recompute.null_reference import INCLUSION_RULES, _rule_admits

#: Decision thresholds. The case-mix geometries sit at prevalence 0.20, and a
#: threshold worth acting on lies in a band around the prevalence: below it the
#: model is dominated by treat-all, far above it by treat-none. 0.05 to 0.30
#: brackets that band. The same set is used for every geometry so the tables are
#: comparable; geometries at other prevalences are labelled as such in the CSV.
THRESHOLDS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)

#: Clip applied before ``logit``. The case-mix scores are ``expit`` of a normal,
#: so they can legitimately reach 1e-9 in the tail of a 10,000-row draw; the clip
#: is far enough out to be inactive on all but genuinely saturated values.
LOGIT_EPS = 1e-12

#: Equal-width bins for ECE. Ten is the field default and is the setting whose
#: bias Nixon et al. (2019) and Roelofs et al. (2022) characterise.
ECE_BINS = 10

#: Newton iterations for the two-parameter logistic fits, and the gradient-norm
#: tolerance for declaring convergence.
_MAX_NEWTON = 50
_NEWTON_TOL = 1e-10

#: Fixed-threshold "conventional" flag rules, one per metric family. These are
#: the decision rules an auditor would plausibly write down, and they are the
#: direct analogue of the manuscript's ``fixed_threshold_005`` AUROC rule.
FIXED_RULE_CUT: Dict[str, float] = {
    "citl": 0.20,        # log-odds; |CITL| > 0.2 is a common "miscalibrated" cut
    "mean_cal": 0.05,    # 5 absolute percentage points of O-E
    "cal_slope": 0.20,   # width of the conventional 0.8-1.2 acceptable band
    "ece": 0.05,
    "nb": 0.05,
    "snb": 0.05,         # same 0.05 the AUROC rule uses, on the same 0-1 scale
}


def metric_family(name: str) -> str:
    """``snb_0.10`` -> ``snb``; the key into :data:`FIXED_RULE_CUT`."""
    return name.split("_")[0] if "_" in name and name[-1].isdigit() else name


def _logit(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), LOGIT_EPS, 1.0 - LOGIT_EPS)
    return np.log(s) - np.log1p(-s)


# ── per-subgroup metric estimates, each with a standard error ────────────────
@dataclass(frozen=True)
class MetricValue:
    """One metric on one subgroup: point estimate and its standard error.

    ``se`` is what a naive studentized test would divide by. It is ``nan``
    whenever the estimate is not identified (a class absent, a degenerate score
    distribution, a logistic fit that did not converge), and every consumer must
    drop such levels rather than impute them.
    """

    value: float
    se: float


_NA = MetricValue(float("nan"), float("nan"))


def fit_citl(y: np.ndarray, s: np.ndarray) -> MetricValue:
    """Calibration-in-the-large: intercept of ``logit(p) = a + logit(s)``.

    One-parameter logistic fit with the linear predictor entering as a fixed
    offset. Newton's method on a strictly concave log-likelihood, so convergence
    is global whenever the MLE is finite; it is infinite exactly when the
    subgroup's outcomes are all one class, which is caught first.
    """
    y = np.asarray(y, dtype=float)
    n_pos = float(y.sum())
    if y.size < 2 or n_pos <= 0 or n_pos >= y.size:
        return _NA
    off = _logit(s)
    a = 0.0
    for _ in range(_MAX_NEWTON):
        p = _expit(a + off)
        g = float(np.sum(y - p))
        h = float(np.sum(p * (1.0 - p)))
        if h <= 0.0:
            return _NA
        step = g / h
        a += step
        if abs(g) < _NEWTON_TOL:
            break
    else:
        return _NA
    p = _expit(a + off)
    info = float(np.sum(p * (1.0 - p)))
    if info <= 0.0:
        return _NA
    return MetricValue(float(a), float(np.sqrt(1.0 / info)))


def fit_cal_slope(y: np.ndarray, s: np.ndarray) -> MetricValue:
    """Calibration slope: ``b`` in ``logit(p) = a + b * logit(s)``.

    Two-parameter Newton fit. Returns ``nan`` when the design is degenerate (no
    variation in ``logit(s)`` within the subgroup), when the outcome is constant,
    or under complete separation, where the MLE of ``b`` is ``+inf`` and any
    reported number would be an artefact of the iteration cap. Separation is
    detected by the fitted information matrix going numerically singular, which
    is what actually happens to it.
    """
    y = np.asarray(y, dtype=float)
    n_pos = float(y.sum())
    if y.size < 3 or n_pos <= 0 or n_pos >= y.size:
        return _NA
    x = _logit(s)
    if not np.isfinite(x).all() or float(np.ptp(x)) <= 0.0:
        return _NA
    beta = np.zeros(2)
    X = np.column_stack([np.ones_like(x), x])
    for _ in range(_MAX_NEWTON):
        p = _expit(X @ beta)
        w = p * (1.0 - p)
        g = X.T @ (y - p)
        if float(np.max(np.abs(g))) < _NEWTON_TOL:
            break
        H = X.T @ (X * w[:, None])
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if not np.isfinite(det) or abs(det) < 1e-14:
            return _NA
        beta = beta + np.linalg.solve(H, g)
        if not np.isfinite(beta).all() or float(np.max(np.abs(beta))) > 1e6:
            return _NA
    else:
        return _NA
    p = _expit(X @ beta)
    w = p * (1.0 - p)
    H = X.T @ (X * w[:, None])
    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if not np.isfinite(det) or abs(det) < 1e-14:
        return _NA
    var_b = H[0, 0] / det                      # (H^-1)[1,1]
    if var_b <= 0:
        return _NA
    return MetricValue(float(beta[1]), float(np.sqrt(var_b)))


def mean_calibration(y: np.ndarray, s: np.ndarray) -> MetricValue:
    """``mean(y) - mean(s)``: the O-E difference on the probability scale."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n = y.size
    if n < 2:
        return _NA
    d = y - s
    return MetricValue(float(d.mean()), float(np.sqrt(d.var(ddof=1) / n)))


def ece(y: np.ndarray, s: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected calibration error with ``n_bins`` equal-width bins on [0, 1].

    No standard error is returned. ECE is a sum of absolute values, so it is
    biased *upward* by an amount that grows as the bin counts shrink, and the
    bias -- not the sampling error -- is what dominates its subgroup comparison.
    A standard error would invite exactly the studentized test that the bias
    makes invalid. Use :func:`ece_null_reference` for the bias floor instead.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n = y.size
    if n < 1:
        return float("nan")
    idx = np.clip((s * n_bins).astype(int), 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    sy = np.bincount(idx, weights=y, minlength=n_bins)
    ss = np.bincount(idx, weights=s, minlength=n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(sy[nz] - ss[nz])) / n)


def ece_null_reference(s: np.ndarray, rng: np.random.Generator,
                       n_bins: int = ECE_BINS) -> float:
    """ECE of one draw of ``y* ~ Bernoulli(s)``: the finite-sample bias floor.

    By construction ``y*`` is *perfectly* calibrated with respect to ``s``, so
    the true ECE of ``(y*, s)`` is exactly zero and every unit of the returned
    value is estimation bias at this subgroup's size and score distribution. It
    is the only honest reference for an observed ECE, and it is subgroup-specific
    precisely because the bias is: a subgroup with fewer rows, or with its score
    mass concentrated in fewer bins, has a different floor.
    """
    s = np.asarray(s, dtype=float)
    return ece(rng.random(s.size) < s, s, n_bins)


def net_benefit(y: np.ndarray, s: np.ndarray, t: float
                ) -> Tuple[MetricValue, MetricValue]:
    """Net benefit and standardised net benefit at threshold ``t``.

    ``NB(t) = TP/n - (FP/n) * t/(1-t)`` is the sample mean of the per-subject
    contribution ``c_i = 1{s_i >= t} * (y_i - w (1 - y_i))`` with ``w = t/(1-t)``,
    so its variance is ``var(c)/n`` with no asymptotics beyond the CLT.

    ``sNB(t) = NB(t) / mean(y)`` is a ratio of two means over the same rows; its
    standard error is the delta-method one, which carries the covariance term
    ``cov(c, y)``. That covariance is large -- ``c`` is built from ``y`` -- and
    dropping it (as a naive "divide the SE by the prevalence" would) understates
    the variance substantially.
    """
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    n = y.size
    if n < 2:
        return _NA, _NA
    w = t / (1.0 - t)
    flag = (s >= t).astype(float)
    c = flag * (y - w * (1.0 - y))
    nb = float(c.mean())
    nb_se = float(np.sqrt(c.var(ddof=1) / n))

    prev = float(y.mean())
    if prev <= 0.0:
        return MetricValue(nb, nb_se), _NA
    cov = np.cov(np.vstack([c, y]), ddof=1)
    v_c, v_y, c_cy = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    var_snb = (v_c / prev ** 2
               - 2.0 * c_cy * nb / prev ** 3
               + v_y * nb ** 2 / prev ** 4) / n
    snb_se = float(np.sqrt(var_snb)) if var_snb > 0 else float("nan")
    return MetricValue(nb, nb_se), MetricValue(nb / prev, snb_se)


# ── one subgroup level, all metrics at once ──────────────────────────────────
@dataclass(frozen=True)
class LevelMetrics:
    """Every replacement metric on one level, plus what the rules filter on."""

    n: int
    n_pos: int
    n_neg: int
    metrics: Dict[str, MetricValue]
    ece: float
    ece_null: float

    def admits(self, rule: str) -> bool:
        return _rule_admits(INCLUSION_RULES[rule], self.n, self.n_pos, self.n_neg)


def metric_names(thresholds: Sequence[float] = THRESHOLDS) -> List[str]:
    """Every metric key, in table order. ``ece`` is handled separately."""
    out = ["citl", "mean_cal", "cal_slope"]
    out += [f"nb_{t:.2f}" for t in thresholds]
    out += [f"snb_{t:.2f}" for t in thresholds]
    return out


def level_metrics(y: np.ndarray, s: np.ndarray, codes: np.ndarray,
                  rng: np.random.Generator,
                  thresholds: Sequence[float] = THRESHOLDS
                  ) -> List[LevelMetrics]:
    """All replacement metrics for every level of one demographic column.

    Mirrors :func:`recompute.comparators.core.level_stats`: one pass over the
    data producing every level's statistics, with the five inclusion rules
    applied afterwards, so every rule sees literally the same estimates.
    """
    out: List[LevelMetrics] = []
    for lvl in np.unique(codes):
        m = codes == lvl
        y_g, s_g = y[m], s[m]
        n = int(m.sum())
        n_pos = int(y_g.sum())
        vals: Dict[str, MetricValue] = {
            "citl": fit_citl(y_g, s_g),
            "mean_cal": mean_calibration(y_g, s_g),
            "cal_slope": fit_cal_slope(y_g, s_g),
            # Not a fairness metric: the mechanism. Net benefit is a function of
            # where the score mass sits relative to the threshold AND of the
            # subgroup's event rate, and under case mix the event rate moves.
            # Tabulated so the net-benefit result can be read rather than
            # merely observed.
            "prevalence": MetricValue(
                float(y_g.mean()) if n else float("nan"),
                float(np.sqrt(y_g.mean() * (1 - y_g.mean()) / n)) if n else
                float("nan")),
        }
        for t in thresholds:
            nb, snb = net_benefit(y_g, s_g, t)
            vals[f"nb_{t:.2f}"] = nb
            vals[f"snb_{t:.2f}"] = snb
        out.append(LevelMetrics(
            n=n, n_pos=n_pos, n_neg=n - n_pos, metrics=vals,
            ece=ece(y_g, s_g), ece_null=ece_null_reference(s_g, rng)))
    return out


# ── gap statistics and the naive tests built on them ─────────────────────────
def max_min_gap(vals: Sequence[float]) -> float:
    v = [x for x in vals if np.isfinite(x)]
    return float(max(v) - min(v)) if len(v) >= 2 else float("nan")


def gap_over_partitions(levels_by_col: Dict[str, List[LevelMetrics]],
                        metric: str, rule: str) -> float:
    """Max over partitions of the within-partition max-min gap of ``metric``.

    Exactly the statistic shape :func:`recompute.comparators.core.gap_from_levels`
    computes for the AUROC, so the replacement-metric result and the AUROC result
    are the same functional of different per-level numbers.
    """
    best = float("nan")
    for lv in levels_by_col.values():
        adm = [x for x in lv if x.admits(rule)]
        if metric == "ece":
            g = max_min_gap([x.ece for x in adm])
        elif metric == "ece_null":
            g = max_min_gap([x.ece_null for x in adm])
        else:
            g = max_min_gap([x.metrics[metric].value for x in adm])
        if np.isfinite(g):
            best = g if not np.isfinite(best) else max(best, g)
    return best


def wald_maxt_pvalue(levels_by_col: Dict[str, List[LevelMetrics]],
                     metric: str, rule: str) -> float:
    """Bonferroni-adjusted p-value of the largest studentized pairwise contrast.

    The natural naive test on any of these metrics: for every pair of admissible
    levels within a partition form ``|m_i - m_j| / sqrt(se_i^2 + se_j^2)``, refer
    it to a two-sided standard normal, and Bonferroni over every pair in every
    partition. Under a true null of equal subgroup metric this is asymptotically
    valid and mildly conservative. It is included so the question "what would a
    naive test's Type I error be" is answered for a test that is actually
    reasonable, not only for a fixed-threshold rule that never claimed a level.

    ECE is deliberately excluded: it has no usable standard error (see
    :func:`ece`), which is itself part of the finding.
    """
    from scipy.stats import norm

    zs: List[float] = []
    for lv in levels_by_col.values():
        adm = [x for x in lv if x.admits(rule)]
        est = [(x.metrics[metric].value, x.metrics[metric].se) for x in adm]
        est = [(v, e) for v, e in est if np.isfinite(v) and np.isfinite(e) and e > 0]
        for i in range(len(est)):
            for j in range(i + 1, len(est)):
                denom = np.sqrt(est[i][1] ** 2 + est[j][1] ** 2)
                if denom > 0:
                    zs.append(abs(est[i][0] - est[j][0]) / denom)
    if not zs:
        return float("nan")
    n_pairs = len(zs)
    p_raw = 2.0 * norm.sf(max(zs))
    return float(min(1.0, p_raw * n_pairs))


def fixed_rule_flag(gap: float, metric: str) -> float:
    """The deterministic rule: flag when the gap reaches the conventional cut."""
    cut = FIXED_RULE_CUT.get(metric_family(metric))
    if cut is None or not np.isfinite(gap):
        return float("nan")
    return float(gap >= cut)


# ── permutation null on the gap, the exact analogue of the incumbent ─────────
def permutation_pvalue(y: np.ndarray, s: np.ndarray,
                       codes_by_col: Dict[str, np.ndarray],
                       observed: Dict[str, float], rule: str,
                       n_perm: int, rng: np.random.Generator,
                       thresholds: Sequence[float] = THRESHOLDS
                       ) -> Dict[str, float]:
    """Monte-Carlo p-value of each cheap metric's gap under label permutation.

    Uses :func:`recompute.null_reference.draw_permuted_codes` with
    ``scheme="joint"`` and a generator seeded exactly as
    ``recompute.comparators.type1._one_sim`` seeds the incumbent's, so these
    p-values are computed on the *same* permutation draws the AUROC procedure
    consumed on the same dataset. Any difference between the two flag rates is
    the metric and nothing else.

    Only the metrics whose per-level value is a ratio of segment sums are done
    here -- ``mean_cal``, ``nb_t``, ``snb_t``, ``ece`` -- because those cost one
    ``bincount`` per replicate. ``citl`` and ``cal_slope`` need an iterative fit
    per level per replicate; they are covered by :func:`wald_maxt_pvalue`
    instead, and the CSV records which test backs which metric.

    A note on what this null can and cannot represent. Permuting labels makes
    every level a random sample of the same pooled cohort, so the permutation
    distribution is built from levels whose *score distributions are all alike*.
    Under case mix the real levels' score distributions are deliberately not
    alike. Any metric whose value depends on the score distribution -- net
    benefit through the threshold crossing, ECE through its binwise
    finite-sample bias -- therefore has a permutation null that is too narrow by
    construction, and the resulting test is anticonservative for reasons that
    have nothing to do with unfairness. Measuring that is the point.
    """
    from recompute.null_reference import draw_permuted_codes

    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    # Per-row contributions, fixed across replicates because only labels move.
    contrib: Dict[str, np.ndarray] = {"mean_cal": y - s}
    for t in thresholds:
        w = t / (1.0 - t)
        flag = (s >= t).astype(float)
        contrib[f"nb_{t:.2f}"] = flag * (y - w * (1.0 - y))
    keys = list(contrib)
    want_ece = "ece" in observed
    bin_idx = np.clip((s * ECE_BINS).astype(int), 0, ECE_BINS - 1)

    r = INCLUSION_RULES[rule]
    ge = {k: 0 for k in observed}
    n_valid = {k: 0 for k in observed}

    for _ in range(n_perm):
        permuted = draw_permuted_codes(codes_by_col, pos_idx, neg_idx, rng,
                                       scheme="joint")
        best: Dict[str, float] = {k: float("nan") for k in observed}
        for codes in permuted.values():
            k_max = int(codes.max()) + 1
            cnt = np.bincount(codes, minlength=k_max).astype(float)
            n_pos = np.bincount(codes, weights=y, minlength=k_max)
            adm = np.array([_rule_admits(r, int(c), int(p), int(c - p))
                            for c, p in zip(cnt, n_pos)])
            if adm.sum() < 2:
                continue
            sums = {k: np.bincount(codes, weights=contrib[k], minlength=k_max)
                    for k in keys}
            with np.errstate(invalid="ignore", divide="ignore"):
                means = {k: sums[k] / cnt for k in keys}
                prev = n_pos / cnt
            for k in keys:
                v = means[k][adm]
                g = max_min_gap(v)
                if np.isfinite(g):
                    best[k] = g if not np.isfinite(best[k]) else max(best[k], g)
                if k == "mean_cal":
                    continue
                sk = "s" + k
                with np.errstate(invalid="ignore", divide="ignore"):
                    vs = np.where(prev[adm] > 0, means[k][adm] / prev[adm], np.nan)
                gs = max_min_gap(vs)
                if np.isfinite(gs):
                    best[sk] = gs if not np.isfinite(best[sk]) else max(best[sk], gs)
            if want_ece:
                joint = codes * ECE_BINS + bin_idx
                m = k_max * ECE_BINS
                sy = np.bincount(joint, weights=y, minlength=m).reshape(k_max, -1)
                ss = np.bincount(joint, weights=s, minlength=m).reshape(k_max, -1)
                # Empty bins contribute |0 - 0| = 0, so no mask is needed.
                with np.errstate(invalid="ignore", divide="ignore"):
                    e = np.abs(sy - ss).sum(axis=1) / cnt
                ge_ = max_min_gap(e[adm])
                if np.isfinite(ge_):
                    best["ece"] = (ge_ if not np.isfinite(best["ece"])
                                   else max(best["ece"], ge_))
        for k, obs in observed.items():
            b = best.get(k, float("nan"))
            if np.isfinite(b) and np.isfinite(obs):
                n_valid[k] += 1
                ge[k] += int(b >= obs)

    out: Dict[str, float] = {}
    for k in observed:
        out[k] = ((ge[k] + 1) / (n_valid[k] + 1)) if n_valid[k] else float("nan")
    return out


# ── exact truth under the case-mix geometries ────────────────────────────────
_QZ = np.linspace(-12.0, 12.0, 24_001)
_QW = np.exp(-0.5 * _QZ ** 2) / np.sqrt(2.0 * np.pi)


def true_case_mix_metrics(geom: Geometry,
                          thresholds: Sequence[float] = THRESHOLDS
                          ) -> Dict[str, float]:
    """Exact population value of every replacement metric, by quadrature.

    The analogue of :func:`recompute.comparators.simulate.true_subgroup_auc`, and
    the number that decides the whole question. Under a case-mix geometry the
    score is ``p = expit(b0 + L)`` with ``L ~ N(loc_g, scale_g^2)`` and the
    outcome is ``y ~ Bernoulli(p)``, so within level ``g``

        prevalence_g = E[p]
        NB_g(t)      = E[ 1{p >= t} (p - t) ] / (1 - t)
        sNB_g(t)     = NB_g(t) / prevalence_g

    while the true CITL is 0, the true calibration slope is 1 and the true ECE is
    0 in *every* level, because the score is the exact conditional probability
    there. Calibration therefore has a true subgroup gap of exactly zero and net
    benefit does not, which is the asymmetry the whole module exists to measure.

    Returned keys are ``<metric>__level_<k>``, plus ``<metric>__max_gap`` and
    ``<metric>__mean``. No Monte-Carlo error: this is quadrature, not simulation.
    """
    if geom.case_mix is None:
        return {}
    cm = geom.case_mix
    b0 = case_mix_intercept(geom)
    w = _mixture_weights(geom)
    out: Dict[str, float] = {"intercept": b0}

    prevs: List[float] = []
    per_t: Dict[float, Tuple[List[float], List[float]]] = {t: ([], []) for t in thresholds}
    for k, (loc, scale) in enumerate(zip(cm.locs, cm.scales)):
        p = _expit(b0 + loc + scale * _QZ)
        prev = float(np.trapz(p * _QW, _QZ))
        prevs.append(prev)
        out[f"prevalence__level_{k}"] = prev
        for t in thresholds:
            nb = float(np.trapz(np.where(p >= t, p - t, 0.0) * _QW, _QZ) / (1.0 - t))
            snb = nb / prev if prev > 0 else float("nan")
            per_t[t][0].append(nb)
            per_t[t][1].append(snb)
            out[f"nb_{t:.2f}__level_{k}"] = nb
            out[f"snb_{t:.2f}__level_{k}"] = snb

    out["prevalence__max_gap"] = float(max(prevs) - min(prevs))
    out["prevalence__mean"] = float(np.dot(w, prevs))
    for t in thresholds:
        for tag, vals in (("nb", per_t[t][0]), ("snb", per_t[t][1])):
            key = f"{tag}_{t:.2f}"
            out[f"{key}__max_gap"] = float(max(vals) - min(vals))
            out[f"{key}__mean"] = float(np.dot(w, vals))
    # Calibration is exactly right in every level, by construction.
    for tag, truth in (("citl", 0.0), ("mean_cal", 0.0),
                       ("cal_slope", 1.0), ("ece", 0.0)):
        for k in range(len(cm.locs)):
            out[f"{tag}__level_{k}"] = truth
        out[f"{tag}__max_gap"] = 0.0
        out[f"{tag}__mean"] = truth
    return out


def true_gap(geom: Geometry, metric: str,
             thresholds: Sequence[float] = THRESHOLDS) -> Optional[float]:
    """The exact true subgroup gap of ``metric``, or ``None`` when unknown.

    Known exactly for the case-mix geometries (quadrature above) and for the
    *simple* null, where the subgroups are fully exchangeable so every subgroup
    metric is identical and every true gap is zero. Under the *composite* null it
    is deliberately ``None``: the per-level monotone score map preserves each
    subgroup's AUROC exactly but does **not** preserve its calibration or its net
    benefit, so those subgroups genuinely differ and a flag there is a true
    positive, not a Type I error. Claiming zero would be false.

    ``ece_null`` is the *expected finite-sample bias* of ECE, not a population
    metric, and its subgroup gap is genuinely non-zero wherever the subgroups'
    sizes or score distributions differ. It has no true value to compare against
    and returns ``None`` outside the exchangeable case.
    """
    if metric == "ece_null" and geom.case_mix is not None:
        return None
    if geom.case_mix is not None:
        return true_case_mix_metrics(geom, thresholds).get(f"{metric}__max_gap")
    if geom.is_composite:
        return None
    return 0.0
