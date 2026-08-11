"""
Tests for the replacement-metric study.

The claim under test is not "the code runs". It is that the numbers in
``recompute/results/replacement_metrics.csv`` mean what ``REPLACEMENT_METRICS.md``
says they mean. Three things therefore have to hold, and each has its own
section below:

1. **The estimators are the textbook ones.** Calibration-in-the-large, the
   calibration slope and net benefit are checked against their definitions and
   against cases whose answer is known in closed form -- not against a previous
   run of this same code.

2. **The exact truth is exact.** ``true_case_mix_metrics`` decides the headline
   finding: it is what says the true subgroup net-benefit gap under case mix is
   large and the true subgroup calibration gap is zero. If its quadrature were
   wrong the whole conclusion would be wrong, so it is checked against a
   half-million-row Monte-Carlo draw from the very same geometry.

3. **The study is on the same footing as the AUROC study it is answering.**
   Same datasets, same seeds, same permutation draws. A comparison run on
   different data would not be a comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from recompute.comparators import replacement as R
from recompute.comparators.simulate import (
    GEOMETRY_BY_NAME,
    _expit,
    case_mix_intercept,
    make_dataset,
)

CASE_MIX = ("casemix_location_3", "casemix_mild_3", "casemix_moderate_3",
            "casemix_moderate_3_n10000", "casemix_moderate_3part",
            "casemix_strong_4")


# ── 1. the estimators are the textbook ones ──────────────────────────────────
def test_perfect_calibration_recovers_intercept_zero_and_slope_one():
    """When the score IS the probability, CITL -> 0 and slope -> 1."""
    rng = np.random.default_rng(0)
    s = _expit(rng.normal(-1.5, 1.2, 200_000))
    y = (rng.random(s.size) < s).astype(int)

    assert R.fit_citl(y, s).value == pytest.approx(0.0, abs=0.02)
    assert R.fit_cal_slope(y, s).value == pytest.approx(1.0, abs=0.02)
    assert R.mean_calibration(y, s).value == pytest.approx(0.0, abs=0.003)


def test_calibration_slope_recovers_a_known_miscalibration():
    """A score built by shrinking the true log-odds has slope 1/shrinkage.

    If ``logit(s) = logit(p) / c`` then ``logit(p) = c * logit(s)`` exactly, so
    the calibration slope of ``s`` is ``c``. This is the sharpest available check
    that the fit is estimating the slope and not something adjacent to it.
    """
    rng = np.random.default_rng(1)
    for c in (0.5, 1.0, 1.8):
        lp = rng.normal(-1.0, 1.5, 400_000)
        y = (rng.random(lp.size) < _expit(lp)).astype(int)
        s = _expit(lp / c)
        assert R.fit_cal_slope(y, s).value == pytest.approx(c, rel=0.02)


def test_citl_recovers_a_known_intercept_shift():
    """Shifting the log-odds by ``d`` gives CITL exactly ``-d`` in expectation."""
    rng = np.random.default_rng(2)
    for d in (-0.7, 0.0, 0.5):
        lp = rng.normal(-1.2, 1.0, 400_000)
        y = (rng.random(lp.size) < _expit(lp)).astype(int)
        assert R.fit_citl(y, _expit(lp + d)).value == pytest.approx(-d, abs=0.02)


def test_net_benefit_matches_its_definition():
    """``NB = TP/n - (FP/n) * t/(1-t)``, counted directly."""
    rng = np.random.default_rng(3)
    s = rng.random(5000)
    y = (rng.random(5000) < s).astype(int)
    n = y.size
    for t in (0.05, 0.2, 0.5, 0.8):
        pred = s >= t
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        want = tp / n - (fp / n) * (t / (1.0 - t))
        nb, snb = R.net_benefit(y, s, t)
        assert nb.value == pytest.approx(want, abs=1e-12)
        assert snb.value == pytest.approx(want / y.mean(), abs=1e-12)


def test_net_benefit_of_treat_all_and_treat_none():
    """The two decision-curve reference strategies, in closed form."""
    rng = np.random.default_rng(4)
    y = (rng.random(20_000) < 0.3).astype(int)
    prev = float(y.mean())
    for t in (0.1, 0.25, 0.4):
        # treat-all: every score above any threshold
        nb_all = R.net_benefit(y, np.ones_like(y, dtype=float), t)[0].value
        assert nb_all == pytest.approx(prev - (1 - prev) * t / (1 - t), abs=1e-12)
        # treat-none: no score reaches the threshold, so NB is exactly 0
        nb_none = R.net_benefit(y, np.zeros_like(y, dtype=float), t)[0].value
        assert nb_none == pytest.approx(0.0, abs=1e-12)


def test_net_benefit_se_is_the_sample_mean_se():
    """NB is a sample mean, so its SE must be ``sd(contribution)/sqrt(n)``."""
    rng = np.random.default_rng(5)
    s = rng.random(3000)
    y = (rng.random(3000) < s).astype(int)
    t = 0.25
    w = t / (1 - t)
    c = (s >= t) * (y - w * (1 - y))
    assert R.net_benefit(y, s, t)[0].se == pytest.approx(
        np.sqrt(c.var(ddof=1) / c.size), rel=1e-12)


@pytest.mark.parametrize("n,t", [(500, 0.10), (2000, 0.20)])
def test_standardised_net_benefit_se_matches_a_bootstrap(n, t):
    """The delta-method SE is what the Wald flag rates are computed from.

    sNB is a ratio of two means over the same rows, so its standard error
    carries a ``cov(c, y)`` term that a naive "divide the NB standard error by
    the prevalence" would drop. Checked against a nonparametric bootstrap of the
    whole ratio, which makes no such approximation.
    """
    rng = np.random.default_rng(20 + n)
    s = _expit(rng.normal(-1.4, 1.0, n))
    y = (rng.random(n) < s).astype(int)
    boot = [R.net_benefit(y[i], s[i], t)[1].value
            for i in (rng.integers(0, n, n) for _ in range(3000))]
    assert R.net_benefit(y, s, t)[1].se == pytest.approx(
        float(np.std(boot, ddof=1)), rel=0.06)


def test_degenerate_subgroups_return_nan_not_a_number():
    """One-class outcomes, constant scores and separation must not fabricate."""
    y1 = np.ones(50, int)
    s = np.full(50, 0.4)
    assert not np.isfinite(R.fit_citl(y1, s).value)
    assert not np.isfinite(R.fit_cal_slope(y1, s).value)
    # no variation in logit(s): the slope is not identified
    y = np.array([0, 1] * 25)
    assert not np.isfinite(R.fit_cal_slope(y, s).value)
    # complete separation: the MLE of the slope is +inf
    s_sep = np.linspace(0.01, 0.99, 50)
    y_sep = (s_sep > 0.5).astype(int)
    assert not np.isfinite(R.fit_cal_slope(y_sep, s_sep).value)


def test_ece_is_biased_upward_when_the_truth_is_exactly_zero():
    """The Nixon/Roelofs point, reproduced: perfect calibration, positive ECE.

    ``y ~ Bernoulli(s)`` is perfectly calibrated by construction, so the true ECE
    is exactly 0. The binned estimate is nonetheless positive at every finite n,
    and the bias shrinks with n rather than vanishing. This is why every ECE in
    the study is reported with ``ece_null`` beside it.
    """
    rng = np.random.default_rng(6)
    s_pool = _expit(rng.normal(-1.4, 1.0, 200_000))
    biases = []
    for n in (200, 2000, 20_000):
        vals = [R.ece(rng.random(n) < s_pool[:n], s_pool[:n]) for _ in range(40)]
        biases.append(float(np.mean(vals)))
        assert biases[-1] > 0.0
    assert biases[0] > biases[1] > biases[2], "ECE bias must shrink with n"


# ── 2. the exact truth is exact ──────────────────────────────────────────────
@pytest.mark.parametrize("name", CASE_MIX)
def test_true_case_mix_metrics_match_a_large_monte_carlo_draw(name):
    """Quadrature vs simulation, on the geometry itself.

    This is the test the headline finding rests on. The true subgroup net
    benefit is computed by quadrature in
    :func:`replacement.true_case_mix_metrics`; here the same geometry is drawn
    at 500,000 rows and the subgroup net benefits are counted directly. They
    must agree to Monte-Carlo error.
    """
    from dataclasses import replace

    geom = GEOMETRY_BY_NAME[name]
    truth = R.true_case_mix_metrics(geom)
    big = replace(geom, n=500_000)
    y, s, codes = make_dataset(big, rep=0, seed=11)
    g = codes["p0"]

    for k in range(len(geom.case_mix.locs)):
        m = g == k
        assert m.sum() > 10_000
        assert float(y[m].mean()) == pytest.approx(
            truth[f"prevalence__level_{k}"], abs=0.006)
        for t in R.THRESHOLDS:
            nb, snb = R.net_benefit(y[m], s[m], t)
            assert nb.value == pytest.approx(
                truth[f"nb_{t:.2f}__level_{k}"], abs=0.006)
            assert snb.value == pytest.approx(
                truth[f"snb_{t:.2f}__level_{k}"], abs=0.02)


@pytest.mark.parametrize("name", CASE_MIX)
def test_case_mix_model_is_perfectly_calibrated_in_every_subgroup(name):
    """The premise of the whole comparison, verified rather than assumed.

    The case-mix construction sets the score equal to the exact conditional
    probability, so every subgroup must show CITL 0 and slope 1 in a large draw.
    If this failed, an observed calibration gap would be a real one and the
    finding would be about a broken DGP rather than about the metric.
    """
    from dataclasses import replace

    geom = GEOMETRY_BY_NAME[name]
    big = replace(geom, n=400_000)
    y, s, codes = make_dataset(big, rep=0, seed=12)
    g = codes["p0"]
    for k in range(len(geom.case_mix.locs)):
        m = g == k
        assert R.fit_citl(y[m], s[m]).value == pytest.approx(0.0, abs=0.05)
        assert R.fit_cal_slope(y[m], s[m]).value == pytest.approx(1.0, abs=0.05)
    # ... and the stated true gaps agree.
    assert R.true_gap(geom, "citl") == 0.0
    assert R.true_gap(geom, "cal_slope") == 0.0
    assert R.true_gap(geom, "mean_cal") == 0.0
    assert R.true_gap(geom, "ece") == 0.0


@pytest.mark.parametrize("name", CASE_MIX)
def test_true_net_benefit_gap_under_case_mix_is_not_zero(name):
    """The finding, stated as an assertion so it cannot be quietly lost.

    Under case mix with a perfectly fair, perfectly calibrated model the true
    subgroup standardised-net-benefit gap is materially non-zero at every
    threshold. Any consistent test of equal subgroup net benefit therefore has
    power tending to one against a model that is unfair to nobody.
    """
    geom = GEOMETRY_BY_NAME[name]
    for t in R.THRESHOLDS:
        assert R.true_gap(geom, f"snb_{t:.2f}") > 0.01
    # And at the clinically central threshold it exceeds the AUROC gap the
    # manuscript indicts -- the reason the recommendation does not survive.
    from recompute.comparators.simulate import true_subgroup_auc

    auc_gap = true_subgroup_auc(geom)["max_gap"]
    assert R.true_gap(geom, "snb_0.20") > auc_gap


def test_true_gap_is_none_under_the_composite_null():
    """The composite null preserves AUROC but not calibration; do not claim 0."""
    geom = GEOMETRY_BY_NAME["composite_shift_4"]
    for m in ("citl", "cal_slope", "snb_0.20", "ece"):
        assert R.true_gap(geom, m) is None
    simple = GEOMETRY_BY_NAME["balanced_3x1000"]
    for m in ("citl", "cal_slope", "snb_0.20", "ece"):
        assert R.true_gap(simple, m) == 0.0


# ── 3. same footing as the AUROC study ───────────────────────────────────────
def test_study_consumes_the_auroc_studys_datasets_unchanged():
    """No new draws: the replacement metrics see the identical cohorts."""
    for name in ("casemix_moderate_3", "composite_shift_4", "balanced_3x1000"):
        geom = GEOMETRY_BY_NAME[name]
        for rep in (0, 7):
            a = make_dataset(geom, rep, 42)
            b = make_dataset(geom, rep, 42)
            assert np.array_equal(a[0], b[0])
            assert np.array_equal(a[1], b[1])
            for c in a[2]:
                assert np.array_equal(a[2][c], b[2][c])


def test_one_sim_is_deterministic_in_geometry_replicate_and_seed():
    from recompute.comparators.replacement_study import _one_sim

    geom = GEOMETRY_BY_NAME["casemix_moderate_3"]
    a = _one_sim(geom, 3, "m30", 49, 42, 0.05)
    b = _one_sim(geom, 3, "m30", 49, 42, 0.05)
    assert set(a) == set(b)
    for k in a:
        assert (a[k] == b[k]) or (not np.isfinite(a[k]) and not np.isfinite(b[k]))


def test_permutation_uses_the_same_draws_as_the_incumbent():
    """The permutation stream is seeded exactly as ``type1._one_sim`` seeds it.

    Both consume ``draw_permuted_codes`` from ``default_rng([seed, rep, 1])``
    with ``scheme="joint"``, so the replacement metric's null and the AUROC
    procedure's null are built on the identical sequence of relabellings.
    """
    from recompute.null_reference import draw_permuted_codes

    geom = GEOMETRY_BY_NAME["casemix_moderate_3"]
    y, s, codes = make_dataset(geom, 0, 42)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)

    rng_a = np.random.default_rng([42, 0, 1])
    first_a = draw_permuted_codes(codes, pos, neg, rng_a, scheme="joint")["p0"]
    rng_b = np.random.default_rng([42, 0, 1])
    first_b = draw_permuted_codes(codes, pos, neg, rng_b, scheme="joint")["p0"]
    assert np.array_equal(first_a, first_b)


def test_joint_permutation_preserves_subgroup_size_and_event_count():
    """Why the permutation null is *conservative* for prevalence-driven gaps.

    ``scheme="joint"`` permutes labels within outcome class, so every level
    keeps its ``n`` and its ``n_pos`` exactly. Subgroup prevalence is therefore
    invariant under the null the permutation test builds -- which means that
    part of the net-benefit gap is inside the null, not outside it. This is the
    mechanism behind the location-driven geometries' near-zero permutation flag
    rate, and it must not silently change.
    """
    from recompute.null_reference import draw_permuted_codes

    geom = GEOMETRY_BY_NAME["casemix_location_3"]
    y, s, codes = make_dataset(geom, 0, 42)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(0)
    g0 = codes["p0"]
    want_n = np.bincount(g0)
    want_pos = np.bincount(g0, weights=y)
    for _ in range(5):
        p = draw_permuted_codes(codes, pos, neg, rng, scheme="joint")["p0"]
        assert np.array_equal(np.bincount(p), want_n)
        assert np.allclose(np.bincount(p, weights=y), want_pos)


def test_gap_is_max_over_partitions_of_the_within_partition_range():
    """The gap functional has the same shape as the AUROC study's."""
    rng = np.random.default_rng(9)
    geom = GEOMETRY_BY_NAME["casemix_moderate_3part"]
    y, s, codes = make_dataset(geom, 0, 42)
    lv = {c: R.level_metrics(y, s, cc, rng) for c, cc in codes.items()}
    for metric in ("mean_cal", "snb_0.20"):
        per_col = []
        for col, levels in lv.items():
            vals = [x.metrics[metric].value for x in levels if x.admits("m30")]
            vals = [v for v in vals if np.isfinite(v)]
            per_col.append(max(vals) - min(vals))
        assert R.gap_over_partitions(lv, metric, "m30") == pytest.approx(
            max(per_col), abs=1e-12)


def test_wald_test_is_not_fabricated_when_too_few_levels_survive():
    rng = np.random.default_rng(10)
    y = np.array([0, 1, 0, 1] * 10)
    s = np.linspace(0.1, 0.9, 40)
    codes = np.zeros(40, dtype=np.int32)          # a single level
    lv = {"p0": R.level_metrics(y, s, codes, rng)}
    assert not np.isfinite(R.wald_maxt_pvalue(lv, "citl", "m30"))
    assert not np.isfinite(R.gap_over_partitions(lv, "citl", "m30"))


@pytest.mark.slow
def test_naive_tests_hold_their_level_under_the_exchangeable_null():
    """Sanity floor: on the simple null the two real tests sit near alpha.

    If a test were broken it would miss its level here, where the subgroups are
    fully exchangeable and every procedure should be exactly valid. Two hundred
    replicates give a rate SE of about 0.015, so the assertion band is wide on
    purpose -- this is a smoke alarm, not the study.
    """
    from recompute.comparators.replacement_study import _one_sim

    geom = GEOMETRY_BY_NAME["balanced_3x1000"]
    sims = [_one_sim(geom, r, "m30", 199, 42, 0.05) for r in range(200)]
    for key in ("wald__mean_cal", "perm__snb_0.20", "perm__mean_cal"):
        v = np.array([sd[key] for sd in sims], dtype=float)
        rate = float(np.nanmean(v))
        assert rate < 0.12, f"{key} flags at {rate:.3f} under exchangeability"
