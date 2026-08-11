"""Closed-form true subgroup AUROC under case-mix heterogeneity.

What this module is for
-----------------------
``recompute.comparators.simulate.true_subgroup_auc`` computes the true subgroup
AUROC of a case-mix geometry by trapezoid quadrature on a 40,001-point grid.
That is a *numerical* answer for six specific geometries. This module supplies
the *analytic* answer that those six numbers instantiate, so the manuscript can
state a general result rather than six simulated ones.

The identity
------------
Let the model be well specified in subgroup ``g``: the score ``S`` it reports is
the true conditional event probability, and ``Y | S ~ Bernoulli(S)``. Write
``pi_g = E[S]`` for the subgroup prevalence and

    Delta_g = E|S - S'|,   S, S' iid copies of the subgroup's risk

for the Gini mean difference of the risk distribution. Then (Proposition 1)

    AUROC_g = 1/2 + Delta_g / (4 pi_g (1 - pi_g)).

No distributional assumption is used: not normality, not a logit link, not even
continuity beyond the usual mid-rank handling of ties. Only well-specification.

Provenance of the identity
--------------------------
This is *not* new. It is the population form of the model-based concordance of
van Klaveren et al. (2016, Stat Med 35:4124-4135), which the manuscript already
cites for the case-mix point. The Gaussian-linear-predictor special case, in its
rare-outcome limit, is the c-statistic expression of Austin & Steyerberg (2012,
BMC Med Res Methodol 12:82) -- see :func:`auroc_rare_outcome_limit` and the
verification in ``CASEMIX_DERIVATION.md``. The contribution of this module is
what those sources do not supply: the induced max-min gap across G subgroups
(:func:`subgroup_auroc_gap`), the consequence for testing equal subgroup AUROC
(:func:`power_lower_bound`, :func:`n_for_power`), and the converse -- the exact
condition under which a well-specified model *does* satisfy the equal-AUROC null
(:func:`equal_auroc_partner_mean`).

Numerics
--------
Every quantity is an adaptive one-dimensional quadrature (``scipy.integrate.quad``)
rather than a fixed grid, so the values here are accurate to ~1e-12 rather than
the ~6e-8 of the 40,001-point trapezoid rule in ``simulate.py``. The two agree to
the trapezoid rule's own O(dt^2) truncation error; see
``recompute.verify_casemix_theory``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

#: Pinned seed for every stochastic check that uses this module.
CASEMIX_SEED = 20240810

__all__ = [
    "CASEMIX_SEED",
    "expit",
    "subgroup_prevalence",
    "gini_mean_difference_gaussian",
    "auroc_gaussian_lp",
    "auroc_rare_outcome_limit",
    "auroc_from_risk",
    "auroc_definition_from_risk",
    "subgroup_auroc_gap",
    "power_lower_bound",
    "n_for_power",
    "equal_auroc_partner_mean",
]


def expit(x):
    """Logistic function, overflow-safe for large negative ``x``."""
    return np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def _scalar_expit(x: float) -> float:
    return float(expit(np.asarray(float(x))))


# ── Gaussian linear predictor ────────────────────────────────────────────────

def _piecewise_quad(f, lo: float, hi: float, breaks: Sequence[float]) -> float:
    """Integrate ``f`` over ``[lo, hi]``, forcing the panels at ``breaks``.

    Both integrands below carry structure on *two* different scales: the risk
    factor ``p(1-p)`` varies on a unit scale around the linear predictor value 0,
    while the placement factor varies on a ``sd_lp`` scale around ``mean_lp``.
    When those scales are far apart -- a narrow subgroup whose risk sits well
    away from 1/2, which is exactly the rare-outcome clinical case -- the
    integrand is a narrow spike inside a wide interval, and a single adaptive
    ``quad`` call over the whole range steps straight over it and silently
    returns 0. Splitting at known breakpoints puts at least one panel on each
    feature, so the adaptive refinement always has something to bite on.
    """
    pts = sorted({float(lo), float(hi)}
                 | {float(b) for b in breaks if lo < float(b) < hi})
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        val, _ = quad(f, a, b, limit=400, epsabs=1e-16, epsrel=1e-13)
        total += float(val)
    return total


def subgroup_prevalence(mean_lp: float, sd_lp: float) -> float:
    """``pi_g = E[expit(L)]`` for ``L ~ N(mean_lp, sd_lp^2)``.

    No elementary closed form exists; this is the exact one-dimensional integral,
    evaluated adaptively. ``mean_lp`` is the *total* linear-predictor mean, i.e.
    the intercept plus the subgroup's covariate contribution.
    """
    mean_lp = float(mean_lp)
    sd_lp = float(sd_lp)
    if sd_lp <= 0.0:
        return _scalar_expit(mean_lp)
    # Break at the logistic transition z = -mean/sd as well as at the standard
    # normal's own scale, so a narrow subgroup far from risk 1/2 is still resolved.
    breaks = [float(j) for j in range(-14, 15)]
    breaks += [-mean_lp / sd_lp + j for j in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    return _piecewise_quad(
        lambda z: _scalar_expit(mean_lp + sd_lp * z) * float(norm.pdf(z)),
        -14.0, 14.0, breaks)


def gini_mean_difference_gaussian(mean_lp: float, sd_lp: float) -> float:
    """``E|S - S'|`` for ``S = expit(L)``, ``L ~ N(mean_lp, sd_lp^2)``.

    Uses the layer-cake identity for a strictly increasing ``p``::

        E|p(T) - p(U)| = 2 * int p'(t) F(t) (1 - F(t)) dt

    which turns a two-dimensional expectation into a one-dimensional integral.
    Here ``p' = p(1-p)`` and ``F`` is the normal cdf of the linear predictor.
    """
    sd_lp = float(sd_lp)
    if sd_lp <= 0.0:
        return 0.0

    mean_lp = float(mean_lp)

    def integrand(t: float) -> float:
        p = _scalar_expit(t)
        f = float(norm.cdf((t - mean_lp) / sd_lp))
        return p * (1.0 - p) * f * (1.0 - f)

    lo = min(mean_lp - 16.0 * sd_lp, -80.0)
    hi = max(mean_lp + 16.0 * sd_lp, 80.0)
    # Two feature scales: the placement factor F(1-F) lives on ``sd_lp`` around
    # ``mean_lp``; the risk factor p(1-p) lives on a unit scale around 0.
    breaks = [mean_lp + j * sd_lp for j in np.arange(-16.0, 16.5, 0.5)]
    breaks += list(np.arange(-40.0, 40.5, 1.0))
    return 2.0 * _piecewise_quad(integrand, lo, hi, breaks)


def auroc_gaussian_lp(mean_lp: float, sd_lp: float) -> float:
    """True subgroup AUROC for a Gaussian linear predictor (Proposition 2).

    ``AUROC = 1/2 + Delta / (4 pi (1 - pi))`` with ``pi`` and ``Delta`` the exact
    one-dimensional integrals above. Degenerate linear predictor gives 1/2.
    """
    sd_lp = float(sd_lp)
    if sd_lp <= 0.0:
        return 0.5
    pi = subgroup_prevalence(mean_lp, sd_lp)
    if not (0.0 < pi < 1.0):
        return float("nan")
    return 0.5 + gini_mean_difference_gaussian(mean_lp, sd_lp) / (4.0 * pi * (1.0 - pi))


def auroc_rare_outcome_limit(sd_lp: float) -> float:
    """``Phi(sd_lp / sqrt(2))`` -- the ``pi -> 0`` (or ``pi -> 1``) limit.

    This is Austin & Steyerberg's (2012) binormal c-statistic expression written
    in terms of the linear predictor's standard deviation. It is exact only in
    the limit of vanishing (or saturating) prevalence, where the exponentially
    tilted case and control distributions of the linear predictor both become
    Gaussian with a common variance. At finite prevalence it is an upper bound:
    see :func:`auroc_gaussian_lp` and the table in ``CASEMIX_DERIVATION.md``.
    """
    return float(norm.cdf(float(sd_lp) / np.sqrt(2.0)))


# ── Distribution-free forms, for arbitrary linear predictors ─────────────────

def auroc_from_risk(risk: Sequence[float]) -> float:
    """Proposition 1 applied to an explicit risk population ``risk``.

    ``risk`` is treated as the subgroup's risk *distribution* (each entry equally
    likely), not as a sample with outcomes attached. Exact for that distribution;
    ``O(n log n)`` via the sorted-sample form of the Gini mean difference.
    """
    s = np.sort(np.asarray(risk, dtype=float))
    n = s.size
    if n < 2:
        return float("nan")
    i = np.arange(1, n + 1, dtype=np.float64)
    gmd = (2.0 / n ** 2) * float(np.sum((2.0 * i - n - 1.0) * s))
    pi = float(s.mean())
    if not (0.0 < pi < 1.0):
        return float("nan")
    return 0.5 + gmd / (4.0 * pi * (1.0 - pi))


def auroc_definition_from_risk(risk: Sequence[float]) -> float:
    """AUROC of the same population computed straight from the definition.

    ``P(S_case > S_control) + 0.5 P(S_case = S_control)`` with case weight ``s``
    and control weight ``1 - s`` on each atom. Independent of
    :func:`auroc_from_risk`; the two agreeing to machine precision is the check
    that Proposition 1 is an identity rather than an approximation.
    """
    s = np.sort(np.asarray(risk, dtype=float))
    if s.size < 2:
        return float("nan")
    w1 = s
    w0 = 1.0 - s
    tot1 = float(w1.sum())
    tot0 = float(w0.sum())
    if tot1 <= 0.0 or tot0 <= 0.0:
        return float("nan")
    cum0_strict = np.cumsum(w0) - w0
    return float((w1 * (cum0_strict + 0.5 * w0)).sum() / (tot1 * tot0))


# ── The induced gap and its testing consequences ─────────────────────────────

def subgroup_auroc_gap(means: Sequence[float], sds: Sequence[float]) -> float:
    """Max-min true subgroup AUROC across G subgroups (Proposition 3).

    ``means`` are total linear-predictor means (intercept included), ``sds`` the
    per-subgroup linear-predictor standard deviations.
    """
    vals = [auroc_gaussian_lp(m, s) for m, s in zip(means, sds)]
    finite = [v for v in vals if np.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    return float(max(finite) - min(finite))


def power_lower_bound(gap: float, n_events_min: int, n_groups: int) -> float:
    """Hoeffding lower bound on the rejection probability (Proposition 4).

    For each subgroup the AUROC estimator is a two-sample U-statistic with a
    kernel bounded in [0, 1], so Hoeffding's inequality gives

        P(|AUC_hat_g - AUC_g| >= t) <= 2 exp(-2 m_g t^2),

    with ``m_g = min(n_g_pos, n_g_neg)`` the binding class count in subgroup g.
    Any test that rejects once the estimated max-min gap exceeds ``gap / 2``
    -- which every test with a critical value shrinking to zero eventually does
    -- therefore fails to reject with probability at most
    ``2 G exp(-m_min gap^2 / 8)``. Returns the resulting power lower bound,
    clipped at zero. ``n_events_min`` is ``min_g m_g``.

    This is a bound, not the actual power: it is deliberately loose (it ignores
    the variance structure that DeLong exploits) but it is non-asymptotic and it
    is what makes the ``-> 1`` statement precise, with an exponential rate in
    ``n gap^2``.
    """
    gap = float(gap)
    if gap <= 0.0:
        return 0.0
    bound = 1.0 - 2.0 * n_groups * np.exp(-float(n_events_min) * gap ** 2 / 8.0)
    return float(max(bound, 0.0))


def n_for_power(gap: float, power: float, n_groups: int) -> float:
    """Events per subgroup sufficient for ``power``, from the same bound.

    Inverts :func:`power_lower_bound`. The ``gap ** -2`` scaling is the content:
    halving the true case-mix gap quadruples the cohort needed to make the
    equal-AUROC null reject reliably -- and every clinical cohort large enough to
    matter is on the far side of that threshold for realistic gaps.
    """
    gap = float(gap)
    if gap <= 0.0:
        return float("inf")
    return float(8.0 * np.log(2.0 * n_groups / (1.0 - power)) / gap ** 2)


def equal_auroc_partner_mean(mean_lp: float, sd_lp: float,
                             bracket: Tuple[float, float] = (-40.0, 40.0)
                             ) -> float:
    """The mirror mean giving *exactly* equal AUROC at the same SD (Prop. 5).

    ``AUROC(m, sigma) = AUROC(-m, sigma)`` exactly, because ``L -> -L`` maps the
    logistic model to itself while swapping cases with controls and reversing
    the score order. So two subgroups sharing an SD and having linear-predictor
    means of opposite sign satisfy the equal-AUROC null exactly, even though
    their prevalences are ``pi`` and ``1 - pi`` and their risk distributions are
    entirely different. Returns ``-mean_lp``; ``bracket`` is accepted only so
    callers can assert the value lies in a sane range.
    """
    lo, hi = bracket
    partner = -float(mean_lp)
    if not (lo <= partner <= hi):
        raise ValueError(f"partner mean {partner} outside bracket {bracket}")
    return partner
