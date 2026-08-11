"""Guards for the closed-form case-mix AUROC result.

The propositions these pin are stated in ``CASEMIX_DERIVATION.md``. The one that
matters most is :func:`test_closed_form_matches_repository_quadrature`: if the
closed form and ``simulate.true_subgroup_auc`` ever disagree by more than the
trapezoid rule's own truncation error, one of them is wrong and the manuscript's
case-mix numbers are not what it says they are.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from recompute.casemix_theory import (
    CASEMIX_SEED,
    auroc_definition_from_risk,
    auroc_from_risk,
    auroc_gaussian_lp,
    auroc_rare_outcome_limit,
    equal_auroc_partner_mean,
    expit,
    n_for_power,
    power_lower_bound,
    subgroup_auroc_gap,
    subgroup_prevalence,
)
from recompute.comparators.simulate import GEOMETRIES, true_subgroup_auc
from recompute.verify_casemix_theory import _is_gaussian_shared_intercept

#: The six case-mix geometries the manuscript's headline rests on.
HEADLINE_SIX = [
    "casemix_mild_3", "casemix_moderate_3", "casemix_strong_4",
    "casemix_location_3", "casemix_moderate_3_n10000", "casemix_moderate_3part",
]


def _gaussian_case_mix_geometries():
    return [g for g in GEOMETRIES
            if g.case_mix is not None and _is_gaussian_shared_intercept(g.case_mix)]


# ── Proposition 2 against the existing quadrature ────────────────────────────

def test_headline_six_are_still_present():
    """The six geometries the paper reports must not vanish from the study."""
    names = {g.name for g in GEOMETRIES if g.case_mix is not None}
    missing = set(HEADLINE_SIX) - names
    assert not missing, f"headline case-mix geometries disappeared: {missing}"


@pytest.mark.parametrize("name", HEADLINE_SIX)
def test_closed_form_matches_repository_quadrature(name):
    geom = next(g for g in GEOMETRIES if g.name == name)
    assert _is_gaussian_shared_intercept(geom.case_mix)
    truth = true_subgroup_auc(geom)
    b0 = truth["intercept"]
    for k, (loc, scale) in enumerate(zip(geom.case_mix.locs, geom.case_mix.scales)):
        closed = auroc_gaussian_lp(b0 + loc, scale)
        assert closed == pytest.approx(truth[f"level_{k}"], abs=1e-6), (
            f"{name} level_{k}: closed form {closed} vs quadrature "
            f"{truth[f'level_{k}']}")
    gap_closed = subgroup_auroc_gap([b0 + l for l in geom.case_mix.locs],
                                    geom.case_mix.scales)
    assert gap_closed == pytest.approx(truth["max_gap"], abs=1e-6)


def test_closed_form_matches_quadrature_for_every_gaussian_geometry():
    worst = 0.0
    for geom in _gaussian_case_mix_geometries():
        truth = true_subgroup_auc(geom)
        b0 = truth["intercept"]
        for k, (loc, scale) in enumerate(zip(geom.case_mix.locs,
                                             geom.case_mix.scales)):
            worst = max(worst, abs(auroc_gaussian_lp(b0 + loc, scale)
                                   - truth[f"level_{k}"]))
    assert worst < 1e-6, f"worst closed-form/quadrature disagreement {worst:.3e}"


# ── Proposition 1 is an identity, and needs no shape assumption ──────────────

@pytest.mark.parametrize("shape", ["normal", "laplace", "lognormal", "bimodal",
                                   "uniform", "t3"])
def test_gini_form_equals_auroc_definition(shape):
    rng = np.random.default_rng(CASEMIX_SEED)
    n = 200_000
    lp = {
        "normal": lambda: rng.standard_normal(n),
        "laplace": lambda: rng.laplace(0.0, 1.0, n),
        "lognormal": lambda: rng.lognormal(0.0, 0.8, n) - 3.0,
        "bimodal": lambda: np.where(rng.random(n) < 0.5, -2.0, 2.0)
                           + 0.3 * rng.standard_normal(n),
        "uniform": lambda: rng.uniform(-3.0, 3.0, n),
        "t3": lambda: rng.standard_t(3, n),
    }[shape]()
    risk = expit(lp)
    assert auroc_from_risk(risk) == pytest.approx(
        auroc_definition_from_risk(risk), abs=1e-10)


def test_gini_form_agrees_with_the_gaussian_quadrature():
    """Proposition 1 (sampled) and Proposition 2 (quadrature) are the same thing."""
    rng = np.random.default_rng(CASEMIX_SEED + 3)
    for mean_lp, sd_lp in [(0.0, 1.0), (-1.67, 0.7), (-1.75, 1.9), (2.0, 1.4)]:
        lp = mean_lp + sd_lp * rng.standard_normal(4_000_000)
        assert auroc_from_risk(expit(lp)) == pytest.approx(
            auroc_gaussian_lp(mean_lp, sd_lp), abs=2e-3)


# ── shape of the AUROC surface ───────────────────────────────────────────────

def test_degenerate_linear_predictor_gives_one_half():
    assert auroc_gaussian_lp(-1.5, 0.0) == 0.5


def test_quadrature_resolves_narrow_subgroups_far_from_risk_one_half():
    """Regression: the integrand is a spike on two very different scales.

    ``gini_mean_difference_gaussian``'s integrand carries the placement factor on
    a ``sd_lp`` scale around ``mean_lp`` and the risk factor on a unit scale
    around 0. With ``sd_lp = 0.2`` and ``mean_lp = -4`` those are 20 standard
    deviations apart inside an integration range of width 160, and a single
    adaptive ``quad`` call stepped over the spike and returned exactly 0 --
    reporting AUROC 0.5 for a subgroup whose true AUROC is 0.55. The reference
    below is a deterministic quantile lattice, so this test has no seed and no
    Monte-Carlo error.
    """
    n = 200_000
    lattice = norm.ppf((np.arange(n) + 0.5) / n)
    worst = 0.0
    for mean_lp in (-8.0, -4.0, -1.67, 0.0, 3.0, 6.0):
        for sd_lp in (0.1, 0.2, 0.5, 1.0, 1.9, 5.0):
            closed = auroc_gaussian_lp(mean_lp, sd_lp)
            reference = auroc_from_risk(expit(mean_lp + sd_lp * lattice))
            worst = max(worst, abs(closed - reference))
            assert closed > 0.5, f"collapsed to 1/2 at m={mean_lp}, sd={sd_lp}"
    assert worst < 1e-4, f"worst disagreement with the lattice reference {worst:.3e}"


@pytest.mark.parametrize("sd_lp", [0.6, 1.0, 1.9])
def test_auroc_is_even_in_the_linear_predictor_mean(sd_lp):
    """``AUROC(m, sd) == AUROC(-m, sd)`` exactly: the basis of Proposition 5."""
    for m in (0.5, 2.0, 4.0):
        assert auroc_gaussian_lp(m, sd_lp) == pytest.approx(
            auroc_gaussian_lp(-m, sd_lp), abs=1e-11)


@pytest.mark.parametrize("mean_lp", [-4.0, -1.67, 0.0, 2.0])
def test_auroc_increases_with_linear_predictor_sd(mean_lp):
    sds = np.linspace(0.2, 4.0, 30)
    vals = [auroc_gaussian_lp(mean_lp, s) for s in sds]
    assert np.all(np.diff(vals) > 0), f"not increasing in sd at m={mean_lp}"
    assert vals[0] > 0.5


@pytest.mark.parametrize("sd_lp", [0.7, 1.0, 1.9])
def test_auroc_increases_with_absolute_mean_at_fixed_sd(sd_lp):
    """Discrimination is *worst* at prevalence 1/2 and improves as it drifts."""
    ms = np.linspace(0.0, 8.0, 25)
    vals = [auroc_gaussian_lp(m, sd_lp) for m in ms]
    assert np.all(np.diff(vals) > 0)
    assert subgroup_prevalence(0.0, sd_lp) == pytest.approx(0.5, abs=1e-10)


@pytest.mark.parametrize("sd_lp", [0.6, 1.0, 1.4, 1.9])
def test_rare_outcome_limit_is_austin_steyerberg(sd_lp):
    """``Phi(sd/sqrt 2)`` is the pi -> 0 limit, and an upper bound at any pi."""
    limit = auroc_rare_outcome_limit(sd_lp)
    assert auroc_gaussian_lp(-14.0, sd_lp) == pytest.approx(limit, abs=1e-4)
    assert auroc_gaussian_lp(0.0, sd_lp) < limit
    assert norm.cdf(sd_lp / np.sqrt(2.0)) == pytest.approx(limit)


# ── the Gaussian assumption is load-bearing only for the SD parametrisation ──

def test_sd_does_not_order_auroc_across_shapes():
    """A linear predictor with 4x the SD can have a *lower* true AUROC.

    This is the honest limit of the result: 'wider spread means higher AUROC' is
    a statement about a scale family, not about the standard deviation.
    """
    rng = np.random.default_rng(CASEMIX_SEED + 2)
    n = 1_000_000
    sign = lambda: np.where(rng.random(n) < 0.5, -1.0, 1.0)
    narrow = 0.6 * rng.standard_normal(n)
    wide = np.where(rng.random(n) < 0.01, 25.0 * sign(), 0.35 * sign())
    assert np.std(wide) > 3.0 * np.std(narrow)
    assert auroc_from_risk(wide_risk := expit(wide)) < auroc_from_risk(expit(narrow))
    assert auroc_from_risk(wide_risk) < 0.65


def test_scale_monotonicity_holds_for_non_gaussian_symmetric_shapes():
    rng = np.random.default_rng(CASEMIX_SEED + 1)
    n = 400_000
    for base in (rng.laplace(0.0, 1.0, n), rng.uniform(-1.0, 1.0, n),
                 rng.standard_t(3, n)):
        sym = np.concatenate([base, -base])       # exact symmetry about zero
        vals = [auroc_from_risk(expit(c * sym)) for c in np.linspace(0.2, 3.0, 15)]
        assert np.all(np.diff(vals) > 0)


# ── the gap, and its testing consequences ────────────────────────────────────

def test_gap_is_driven_by_the_extreme_sds_when_means_agree():
    means = [-1.67] * 4
    sds = [0.6, 0.9, 1.3, 1.9]
    gap = subgroup_auroc_gap(means, sds)
    assert gap == pytest.approx(auroc_gaussian_lp(-1.67, 1.9)
                                - auroc_gaussian_lp(-1.67, 0.6), abs=1e-12)
    # Adding an interior subgroup cannot change the gap.
    assert subgroup_auroc_gap(means + [-1.67], sds + [1.1]) == pytest.approx(
        gap, abs=1e-12)


def test_equal_sd_and_mirrored_means_satisfy_the_null_exactly():
    """Proposition 5: unequal prevalence, different risk distributions, gap 0."""
    m = 1.6
    partner = equal_auroc_partner_mean(m, 1.0)
    assert partner == -m
    assert subgroup_auroc_gap([m, partner], [1.0, 1.0]) == pytest.approx(0.0, abs=1e-11)
    # ... and the two subgroups really are different populations.
    assert subgroup_prevalence(m, 1.0) != pytest.approx(
        subgroup_prevalence(partner, 1.0), abs=1e-3)


def test_power_bound_is_monotone_and_scales_as_gap_squared():
    assert power_lower_bound(0.0, 10_000, 3) == 0.0
    assert power_lower_bound(0.2, 5_000, 3) > power_lower_bound(0.2, 500, 3)
    assert power_lower_bound(0.3, 2_000, 3) > power_lower_bound(0.1, 2_000, 3)
    # n ~ gap^-2: halving the gap quadruples the requirement.
    assert n_for_power(0.05, 0.8, 3) == pytest.approx(
        4.0 * n_for_power(0.10, 0.8, 3), rel=1e-12)
    assert np.isinf(n_for_power(0.0, 0.8, 3))


def test_headline_gaps_are_what_the_manuscript_reports():
    """The three magnitudes quoted in the case-mix narrative."""
    expected = {"casemix_mild_3": 0.036, "casemix_moderate_3": 0.123,
                "casemix_strong_4": 0.199}
    for name, want in expected.items():
        geom = next(g for g in GEOMETRIES if g.name == name)
        b0 = true_subgroup_auc(geom)["intercept"]
        gap = subgroup_auroc_gap([b0 + l for l in geom.case_mix.locs],
                                 geom.case_mix.scales)
        assert gap == pytest.approx(want, abs=5e-4), f"{name}: gap {gap}"
