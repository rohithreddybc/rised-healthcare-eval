"""
Tests for :mod:`recompute.comparators`.

Three things have to hold for the comparison to mean anything, and they are what
this file checks.

1. **The comparators see the same data as the incumbent.** The observed max-min
   AUROC gap recomputed through the comparator code path must equal, to machine
   precision, the value stored in ``recompute/results/null_joint/<cohort>.json``
   for every cohort and every inclusion rule. If that fails, the two procedures
   are not being run on the same statistic and no comparison is valid.

2. **The fast kernel is a pure speed transformation.** ``fast_level_stats`` must
   agree bit-for-bit with the reference ``level_stats`` (which is built on
   ``scipy.rankdata``), including under heavy ties, and ``gap_from_levels`` must
   agree with ``recompute.null_reference.partition_gaps_by_rule``.

3. **The methods are implemented correctly.** DeLong's variance against a
   bootstrap; the studentized statistic's known invariances; the double-corrected
   variance estimator's unbiasedness in a simulation where the truth is known;
   the four-fifths rule and the fixed threshold against hand-computed cases.

The slower cohort-level checks are marked and can be deselected with
``-m "not slow"``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from recompute.comparators import core, diciccio, four_fifths, incumbent, lum, naive
from recompute.comparators.core import (
    PermContext,
    SortedCohort,
    admissible,
    auc_delong,
    bootstrap_auc_var,
    fast_level_stats,
    gap_from_levels,
    holm,
    level_stats,
    load_cohort,
    max_min_gap,
)
from recompute.null_reference import INCLUSION_RULES, partition_gaps_by_rule

RULES = list(INCLUSION_RULES)
NULL_DIR = core.REPO / "recompute" / "results" / "null_joint"

SMOKE_COHORTS = ["german_credit", "uci_heart", "synthetic"]
ALL_COHORTS = [
    "synthetic", "uci_heart", "diabetes130", "nhis2024", "nhis2023",
    "nhanes2123", "brfss2024", "adult_income", "acs_income", "german_credit",
]


# ── helpers ──────────────────────────────────────────────────────────────────
def _random_case(rng, n=None, n_levels=None, decimals=None):
    """A random (y, s, codes) with controllable tie density."""
    n = n or int(rng.integers(150, 900))
    n_levels = n_levels or int(rng.integers(2, 7))
    decimals = 3 if decimals is None else decimals
    y = (rng.random(n) < rng.uniform(0.08, 0.6)).astype(int)
    s = np.round(rng.normal(0.8 * y, 1.0, n), decimals)
    codes = rng.integers(0, n_levels, n).astype(np.int32)
    return y, s, codes


# ── 1. AUROC and DeLong variance ─────────────────────────────────────────────
def test_auc_matches_sklearn_exactly():
    rng = np.random.default_rng(0)
    for _ in range(25):
        y, s, _ = _random_case(rng)
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            continue
        assert auc_delong(y, s)[0] == pytest.approx(roc_auc_score(y, s),
                                                    abs=1e-12)


def test_auc_matches_fast_auc_of_null_reference():
    from recompute.null_reference import fast_auc

    rng = np.random.default_rng(1)
    for _ in range(25):
        y, s, _ = _random_case(rng)
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            continue
        assert auc_delong(y, s)[0] == pytest.approx(fast_auc(y, s), abs=1e-12)


def test_delong_variance_agrees_with_bootstrap():
    """DeLong is analytic; a stratified bootstrap should land on top of it."""
    rng = np.random.default_rng(2)
    ratios = []
    for _ in range(8):
        n = int(rng.integers(200, 600))
        y = (rng.random(n) < 0.3).astype(int)
        s = rng.normal(0.9 * y, 1.0, n)
        v_dl = auc_delong(y, s)[1]
        v_bs = bootstrap_auc_var(y, s, n_boot=3000, seed=7)
        ratios.append(v_bs / v_dl)
    # Bootstrap noise at B=3000 is a few percent; a systematic error in the
    # DeLong formula would show up as a ratio far from 1.
    assert 0.85 < float(np.mean(ratios)) < 1.15


def test_delong_variance_shrinks_like_one_over_n():
    """Var(AUC) must fall roughly as 1/n at fixed prevalence and separation."""
    rng = np.random.default_rng(3)
    v = {}
    for n in (500, 5000):
        y = (rng.random(n) < 0.3).astype(int)
        s = rng.normal(0.9 * y, 1.0, n)
        v[n] = auc_delong(y, s)[1]
    assert 5.0 < v[500] / v[5000] < 20.0


def test_auc_undefined_with_too_few_of_a_class():
    y = np.array([1, 0, 0, 0, 0])
    s = np.arange(5, dtype=float)
    assert math.isnan(auc_delong(y, s)[0])


# ── 2. The fast kernel is exactly the reference ──────────────────────────────
@pytest.mark.parametrize("decimals", [0, 1, 3, 8])
def test_fast_level_stats_matches_reference_including_ties(decimals):
    rng = np.random.default_rng(10 + decimals)
    for _ in range(15):
        y, s, codes = _random_case(rng, decimals=decimals)
        sc = SortedCohort(y, s)
        ref = level_stats(y, s, codes)
        fast = fast_level_stats(sc, sc.sort_codes(codes))
        assert len(ref) == len(fast)
        for a, b in zip(ref, fast):
            assert (a.n, a.n_pos, a.n_neg) == (b.n, b.n_pos, b.n_neg)
            assert a.auc == pytest.approx(b.auc, abs=1e-12)
            assert a.var == pytest.approx(b.var, abs=1e-12)


def test_fast_kernel_handles_all_scores_identical():
    """Every score tied: AUROC is exactly 0.5 and the variance is 0."""
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.full(6, 0.5)
    sc = SortedCohort(y, s)
    codes = np.zeros(6, dtype=np.int32)
    (lv,) = fast_level_stats(sc, sc.sort_codes(codes))
    assert lv.auc == pytest.approx(0.5)
    assert lv.var == pytest.approx(0.0, abs=1e-12)
    assert level_stats(y, s, codes)[0].auc == pytest.approx(0.5)


@pytest.mark.parametrize("rule", RULES)
def test_gap_from_levels_matches_null_reference(rule):
    rng = np.random.default_rng(20)
    for _ in range(12):
        y, s, codes = _random_case(rng, n=700, n_levels=5)
        cbc = {"a": codes, "b": rng.integers(0, 3, len(y)).astype(np.int32)}
        ctx = PermContext(y, s, cbc)
        mine = gap_from_levels(ctx.observed(), rule)
        theirs = partition_gaps_by_rule(y, s, cbc, [rule])[rule]
        if math.isnan(theirs):
            assert math.isnan(mine)
        else:
            assert mine == pytest.approx(theirs, abs=1e-12)


# ── 3. The comparators see the incumbent's data ──────────────────────────────
@pytest.mark.parametrize("cohort", SMOKE_COHORTS)
def test_observed_gap_matches_published_null_run(cohort):
    """The single most important test in this file.

    If the comparator package reproduces the incumbent's observed statistic for
    every inclusion rule, then it is running on the same split, the same fitted
    model, the same scores and the same subgroup coding.
    """
    path = NULL_DIR / f"{cohort}.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    stored = json.loads(path.read_text(encoding="utf-8"))["observed_gap_by_rule"]
    data = load_cohort(cohort)
    for rule in RULES:
        mine = naive.cohort_gap(data, rule)
        theirs = stored[rule]
        if theirs is None or not np.isfinite(theirs):
            assert math.isnan(mine), f"{cohort}/{rule}"
        else:
            assert mine == pytest.approx(theirs, abs=1e-12), f"{cohort}/{rule}"


@pytest.mark.slow
@pytest.mark.parametrize("cohort", ALL_COHORTS)
def test_observed_gap_matches_published_null_run_all(cohort):
    test_observed_gap_matches_published_null_run(cohort)


@pytest.mark.slow
@pytest.mark.parametrize("cohort", ["german_credit", "nhanes2123", "synthetic"])
def test_recomputed_null_reproduces_the_published_pvalues(cohort):
    """The strongest equivalence claim available.

    Reproducing the observed statistic shows the comparators see the same inputs.
    Reproducing the entire null -- every p-value, at B=10,000 and seed 42 --
    shows the comparator kernel reproduces the incumbent's whole *procedure*, so
    a runtime measured on that kernel is a fair basis for comparison and any
    difference in verdict is the statistic and nothing else.
    """
    path = NULL_DIR / f"{cohort}.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    stored = json.loads(path.read_text(encoding="utf-8"))
    if stored.get("status") != "ok":
        pytest.skip("stored run did not succeed")
    block = stored["results"]["joint"]
    data = load_cohort(cohort)
    got = incumbent.recompute_null(data, n_perm=stored["n_reps"],
                                   seed=stored["seed"])
    for rule in RULES:
        want = block[rule].get("p_value_vs_null")
        mine = got[rule]["p_value"]
        if want is None:
            assert math.isnan(mine), f"{cohort}/{rule}"
        else:
            assert mine == pytest.approx(want, abs=1e-12), f"{cohort}/{rule}"


@pytest.mark.slow
def test_permutation_draws_match_the_incumbents():
    """A comparator's permutation stream must be the incumbent's stream.

    Both call ``draw_permuted_codes`` with the same seed and the same scheme, so
    the sequence of permuted code arrays must be identical element by element.
    """
    from recompute.null_reference import draw_permuted_codes

    data = load_cohort("german_credit")
    ctx = PermContext(data.y, data.s, data.codes_by_col)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    for _ in range(20):
        ref = draw_permuted_codes(data.codes_by_col, data.pos_idx,
                                  data.neg_idx, rng_a, scheme="joint")
        got = ctx.draw(rng_b)
        for col in ref:
            expected = fast_level_stats(ctx.sc, ctx.sc.sort_codes(ref[col]))
            assert [lv.auc for lv in expected] == [lv.auc for lv in got[col]]


# ── 4. DiCiccio ──────────────────────────────────────────────────────────────
def test_studentized_statistic_is_scale_free_in_a_monotone_reparametrisation():
    """T must not move when the scores are relabelled monotonically.

    AUROC and its DeLong variance are both rank functionals, so a strictly
    increasing transform of the scores cannot change either -- nor T.
    """
    rng = np.random.default_rng(30)
    y, s, codes = _random_case(rng, n=600, n_levels=2, decimals=8)
    a, b = level_stats(y, s, codes)
    t1 = diciccio.studentized(a, b)
    a2, b2 = level_stats(y, np.exp(3 * s), codes)
    assert diciccio.studentized(a2, b2) == pytest.approx(t1, rel=1e-9)


def test_studentized_statistic_is_zero_for_identical_subgroups():
    lv = core.Level(100, 30, 70, 0.7, 0.01)
    assert diciccio.studentized(lv, lv) == pytest.approx(0.0)


def test_studentized_statistic_undefined_when_variance_vanishes():
    a = core.Level(20, 10, 10, 1.0, 0.0)
    b = core.Level(20, 10, 10, 0.0, 0.0)
    assert math.isnan(diciccio.studentized(a, b))


def test_studentized_grows_with_the_gap_at_fixed_variance():
    a = core.Level(200, 60, 140, 0.60, 0.004)
    b1 = core.Level(200, 60, 140, 0.65, 0.004)
    b2 = core.Level(200, 60, 140, 0.80, 0.004)
    assert diciccio.studentized(a, b2) > diciccio.studentized(a, b1) > 0


def test_studentized_penalises_a_noisy_subgroup():
    """The same raw gap must give a smaller T when one subgroup is noisier.

    This is what studentization buys and what the raw max-min gap cannot see.
    """
    a = core.Level(1000, 300, 700, 0.60, 0.0005)
    tight = core.Level(1000, 300, 700, 0.75, 0.0005)
    loose = core.Level(60, 20, 40, 0.75, 0.02)
    assert diciccio.studentized(a, tight) > diciccio.studentized(a, loose)


def test_diciccio_pvalue_is_a_valid_probability_and_respects_the_floor():
    data = load_cohort("german_credit")
    out = diciccio.run_cohort(data, n_perm=200, seed=42)
    for rule, res in out["results"].items():
        if res.p_value is None:
            continue
        assert 1.0 / 201 <= res.p_value <= 1.0


def test_diciccio_is_deterministic_under_a_fixed_seed():
    data = load_cohort("german_credit")
    a = diciccio.run_cohort(data, n_perm=150, seed=42)
    b = diciccio.run_cohort(data, n_perm=150, seed=42)
    for rule in a["results"]:
        assert a["results"][rule].p_value == b["results"][rule].p_value


def test_diciccio_detects_a_planted_subgroup_effect():
    """Power check: a real, large AUROC difference must be flagged."""
    rng = np.random.default_rng(31)
    n = 3000
    y = (rng.random(n) < 0.3).astype(int)
    codes = rng.integers(0, 2, n).astype(np.int32)
    # Group 0 has a strong signal, group 1 has none at all.
    sep = np.where(codes == 0, 1.4, 0.0)
    s = rng.normal(sep * y, 1.0, n)
    ctx = PermContext(y, s, {"g": codes})
    p = diciccio.pvalue_only(ctx, "m30", 199, np.random.default_rng(1))
    assert p < 0.05


# ── 5. Lum ───────────────────────────────────────────────────────────────────
def test_double_corrected_variance_is_unbiased_under_the_null():
    """The estimator's whole claim: E[V_dc] = 0 when the truth is homogeneous.

    Simulates many partitions in which every subgroup shares one true AUROC. The
    naive sample variance must be systematically positive (that is the bias the
    paper is about) and the corrected estimate must average to zero.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(40)
    naive_vals, corrected = [], []
    mu = norm.ppf(0.75) * np.sqrt(2.0)
    for _ in range(400):
        thetas, variances = [], []
        for n_g in (120, 200, 400, 80, 150):
            y = (rng.random(n_g) < 0.3).astype(int)
            if y.sum() < 2 or (n_g - y.sum()) < 2:
                break
            s = rng.normal(mu * y, 1.0, n_g)
            a, v = auc_delong(y, s)
            thetas.append(a)
            variances.append(v)
        if len(thetas) < 5:
            continue
        out = lum.double_corrected_variance(thetas, variances)
        naive_vals.append(out["naive_variance"])
        corrected.append(out["V_dc"])

    assert np.mean(naive_vals) > 0.0
    # The corrected estimator must sit at zero within Monte-Carlo error, and be
    # an order of magnitude closer to zero than the naive one.
    assert abs(np.mean(corrected)) < 0.15 * np.mean(naive_vals)


def test_double_corrected_variance_recovers_a_planted_between_group_variance():
    rng = np.random.default_rng(41)
    true_tau2 = 0.01
    ests = []
    for _ in range(400):
        thetas = 0.75 + rng.normal(0, np.sqrt(true_tau2), 5)
        variances = rng.uniform(0.002, 0.01, 5)
        obs = thetas + rng.normal(0, np.sqrt(variances))
        ests.append(lum.double_corrected_variance(obs, variances)["V_dc"])
    assert np.mean(ests) == pytest.approx(true_tau2, abs=0.004)


def test_double_corrected_variance_algebra():
    th = [0.6, 0.7, 0.8]
    v = [0.01, 0.02, 0.03]
    out = lum.double_corrected_variance(th, v)
    assert out["naive_variance"] == pytest.approx(np.var(th, ddof=1))
    assert out["mean_sampling_variance"] == pytest.approx(np.mean(v))
    assert out["V_dc"] == pytest.approx(np.var(th, ddof=1) - np.mean(v))


def test_double_corrected_variance_can_be_negative():
    """Negative is meaningful, not an error: the noise exceeds the dispersion."""
    out = lum.double_corrected_variance([0.70, 0.71, 0.70], [0.05, 0.05, 0.05])
    assert out["V_dc"] < 0
    assert out["V_dc_truncated"] == 0.0


def test_cochran_q_is_calibrated_under_homogeneity():
    rng = np.random.default_rng(42)
    ps = []
    for _ in range(2000):
        v = rng.uniform(0.002, 0.01, 6)
        th = 0.75 + rng.normal(0, np.sqrt(v))
        ps.append(lum.cochran_q(th, v)["p_value"])
    # Uniform p-values under the null: the 5% tail should be near 5%.
    assert 0.03 < np.mean(np.array(ps) < 0.05) < 0.08


def test_cochran_q_rejects_real_heterogeneity():
    v = np.full(5, 0.001)
    th = np.array([0.60, 0.68, 0.75, 0.82, 0.90])
    assert lum.cochran_q(th, v)["p_value"] < 1e-6


def test_expected_normal_range_matches_known_values():
    """E[range of L iid standard normals] against tabulated constants."""
    assert lum.expected_normal_range(2) == pytest.approx(1.128, abs=0.01)
    assert lum.expected_normal_range(5) == pytest.approx(2.326, abs=0.01)
    assert lum.expected_normal_range(10) == pytest.approx(3.078, abs=0.01)


def test_shrinkage_range_is_never_larger_than_the_observed_range():
    rng = np.random.default_rng(43)
    for _ in range(50):
        th = rng.uniform(0.5, 0.9, 5)
        v = rng.uniform(0.001, 0.02, 5)
        tau2 = lum.double_corrected_variance(th, v)["V_dc_truncated"]
        assert lum.shrunken_range(th, v, tau2) <= (th.max() - th.min()) + 1e-12


def test_lum_bootstrap_ci_brackets_the_estimate():
    th = [0.55, 0.65, 0.75, 0.85]
    v = [0.002] * 4
    est = lum.double_corrected_variance(th, v)["V_dc"]
    ci = lum.bootstrap_ci(th, v, conf=0.95, n_boot=4000, seed=5)
    assert ci["lo"] < est < ci["hi"]


def test_lum_bootstrap_ci_rule_is_nearly_powerless_with_few_subgroups():
    """Pins the reason the bootstrap-CI rule is not the reported Lum verdict.

    Four subgroups whose AUROCs run 0.55 to 0.85 -- almost seven standard errors
    apart, an effect far larger than anything in the ten cohorts -- still do not
    clear zero. With L subgroups the percentile interval on a variance carries
    L-1 degrees of freedom, and at three degrees of freedom its lower 2.5%
    quantile is 7% of the mean, so the interval cannot exclude zero until the
    between-group variance is roughly thirteen times the sampling variance.
    """
    v = [0.002] * 4
    assert lum.bootstrap_ci([0.55, 0.65, 0.75, 0.85], v, n_boot=4000,
                            seed=5)["lo"] < 0.0
    # It is low power, not zero power: a big enough effect does clear zero.
    assert lum.bootstrap_ci([0.30, 0.55, 0.80, 1.00], v, n_boot=4000,
                            seed=5)["lo"] > 0.0
    # And with many subgroups the same effect size is detected.
    assert lum.bootstrap_ci([0.55, 0.62, 0.68, 0.72, 0.78, 0.85] * 3,
                            [0.002] * 18, n_boot=4000, seed=5)["lo"] > 0.0


def test_dc_variance_se_reduces_to_the_textbook_case():
    """With all sampling variances equal, Var(S2) must be 2 sigma^4 / (L-1)."""
    for L in (3, 5, 10, 25):
        sigma2 = 0.004
        got = lum.dc_variance_se(np.full(L, sigma2), tau2=0.0)
        assert got == pytest.approx(math.sqrt(2 * sigma2 ** 2 / (L - 1)),
                                    rel=1e-12)


def test_dc_variance_se_matches_monte_carlo_with_unequal_variances():
    """The general closed form, checked against simulation."""
    rng = np.random.default_rng(50)
    v = np.array([0.001, 0.004, 0.010, 0.002, 0.007])
    draws = rng.normal(0.7, np.sqrt(v), size=(300_000, len(v)))
    mc_sd = float(draws.var(axis=1, ddof=1).std(ddof=1))
    assert lum.dc_variance_se(v, tau2=0.0) == pytest.approx(mc_sd, rel=0.03)


def test_dc_ztest_rejects_a_real_disparity_and_not_a_null_one():
    v = np.full(5, 0.001)
    assert lum.dc_ztest([0.60, 0.68, 0.75, 0.82, 0.90], v)["p_value"] < 1e-4
    assert lum.dc_ztest([0.750, 0.751, 0.749, 0.750, 0.752], v)["p_value"] > 0.5


def test_dc_ztest_tail_pvalue_is_unreliable_at_two_levels():
    """Pins the caveat carried in the report.

    With two subgroups the numerator of V_dc is a single chi-square variate and
    the normal approximation to its far upper tail is badly anti-conservative.
    The exact comparison of two AUROCs is the two-sample z on the difference.
    """
    th = [0.8856, 0.8328]
    v = [7.2712e-05, 7.2712e-05]
    p_dc = lum.dc_ztest(th, v)["p_value"]
    from scipy.stats import norm as _norm

    p_exact = 2 * _norm.sf(abs(th[0] - th[1]) / math.sqrt(sum(v)))
    assert p_dc < p_exact / 1e10          # 1e-37 against 1e-05
    # Cochran's Q on the same inputs is the exactly calibrated reference.
    assert lum.cochran_q(th, v)["p_value"] == pytest.approx(p_exact, rel=0.05)


def test_dc_ztest_z_is_the_estimate_over_the_null_se():
    th = [0.6, 0.7, 0.8, 0.75, 0.65]
    v = [0.002, 0.003, 0.001, 0.004, 0.002]
    out = lum.dc_ztest(th, v)
    est = lum.double_corrected_variance(th, v)["V_dc"]
    assert out["z"] == pytest.approx(est / lum.dc_variance_se(v, 0.0))


def test_lum_run_cohort_emits_all_three_readings():
    data = load_cohort("german_credit")
    block = lum.run_cohort(data, n_boot=200)
    assert set(block["variants"]) == {"lum2022_cochranQ", "lum2022_bootstrapCI"}
    for rule in RULES:
        assert rule in block["results"]
        for v in block["variants"].values():
            assert rule in v


# ── 6. Four-fifths and the fixed threshold ───────────────────────────────────
def test_four_fifths_ratio_hand_computed():
    lv = [core.Level(100, 30, 70, 0.90, 0.01),
          core.Level(100, 30, 70, 0.60, 0.01)]
    assert four_fifths.ratio(lv) == pytest.approx(0.60 / 0.90)
    assert four_fifths.ratio(lv) < four_fifths.THRESHOLD


def test_four_fifths_does_not_fire_on_a_large_absolute_gap():
    """The finding this rule has to be reported for.

    A 0.15 AUROC gap -- three times the naive threshold -- passes the 0.80 rule.
    """
    lv = [core.Level(100, 30, 70, 0.85, 0.01),
          core.Level(100, 30, 70, 0.70, 0.01)]
    assert max_min_gap(lv) == pytest.approx(0.15)
    assert four_fifths.ratio(lv) > four_fifths.THRESHOLD


@pytest.mark.parametrize("cohort", SMOKE_COHORTS)
def test_four_fifths_matches_fairlearn_on_every_cohort(cohort):
    """The published implementation and our loop-friendly one must agree.

    ``run_cohort`` reports fairlearn's number; ``ratio`` is used inside the Type
    I simulation for speed. If they ever diverge the reported and simulated
    behaviour of the rule would be different things.
    """
    data = load_cohort(cohort)
    for rule in RULES:
        for col, codes in data.codes_by_col.items():
            keep = admissible(level_stats(data.y, data.s, codes), rule)
            fl = four_fifths.ratio_fairlearn(data.y, data.s, codes, rule)
            if len(keep) < 2:
                assert math.isnan(fl["ratio"])
                continue
            assert fl["ratio"] == pytest.approx(four_fifths.ratio(keep),
                                                abs=1e-12)
            assert fl["difference"] == pytest.approx(max_min_gap(keep),
                                                     abs=1e-12)


def test_four_fifths_min_detectable_gap():
    lv = [core.Level(100, 30, 70, 0.80, 0.01),
          core.Level(100, 30, 70, 0.75, 0.01)]
    assert four_fifths.min_detectable_gap(lv) == pytest.approx(0.16)


def test_fixed_threshold_boundary_is_inclusive():
    data = load_cohort("german_credit")
    out = naive.run_cohort(data, rules=["m30"])
    res = out["results"]["m30"]
    if res.statistic is not None:
        assert (res.conclusion == "flag") == (res.statistic >= naive.THRESHOLD)


def test_fixed_threshold_equals_the_incumbents_statistic():
    """The naive baseline must be the incumbent's point estimate, exactly."""
    data = load_cohort("german_credit")
    for rule in RULES:
        mine = naive.cohort_gap(data, rule)
        theirs = partition_gaps_by_rule(data.y, data.s, data.codes_by_col,
                                        [rule])[rule]
        if math.isnan(theirs):
            assert math.isnan(mine)
        else:
            assert mine == pytest.approx(theirs, abs=1e-12)


# ── 7. Multiplicity and simulation plumbing ──────────────────────────────────
def test_holm_is_monotone_and_bounded():
    p = [0.001, 0.02, 0.04, 0.7]
    adj = holm(p)
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj <= 1.0)
    assert adj[0] == pytest.approx(0.004)


def test_holm_ignores_nan():
    adj = holm([0.01, np.nan, 0.5])
    assert math.isnan(adj[1])
    assert adj[0] == pytest.approx(0.02)


@pytest.mark.parametrize("geom_name", [g.name for g in
                                       __import__("recompute.comparators.simulate",
                                                  fromlist=["GEOMETRIES"]).GEOMETRIES])
def test_simulated_geometries_really_are_null(geom_name):
    """No DGP may smuggle in a real subgroup effect.

    Checked on the studentized scale, which is asymptotically standard normal
    under the null in every geometry. Any real planted effect would grow without
    bound as n grows; sampling noise would not. The raw gap is checked far more
    loosely because in the skewed geometries the smallest level holds 3% of the
    rows and its AUROC standard error alone is over a percentage point.
    """
    from recompute.comparators.simulate import GEOMETRY_BY_NAME, verify_null

    out = verify_null(GEOMETRY_BY_NAME[geom_name], n_check=200_000)
    assert out["max_studentized"] < 5.0, out
    assert out["max_gap"] < 0.05, out


@pytest.mark.slow
@pytest.mark.parametrize("geom_name", ["skewed_5", "rare_outcome",
                                       "composite_shift_skewed"])
def test_null_geometries_do_not_grow_an_effect_with_n(geom_name):
    """A planted effect would survive n; noise must shrink like 1/sqrt(n).

    Averaged over several draws -- a single realisation of a maximum is far too
    noisy to be monotone in n.
    """
    from recompute.comparators.simulate import GEOMETRY_BY_NAME, verify_null

    geom = GEOMETRY_BY_NAME[geom_name]
    small = np.mean([verify_null(geom, n_check=50_000, seed=s)["max_gap"]
                     for s in range(11, 17)])
    big = np.mean([verify_null(geom, n_check=800_000, seed=s)["max_gap"]
                   for s in range(11, 17)])
    # 16x the rows should cut a pure-noise gap by roughly 4x.
    assert big < 0.5 * small, (small, big)


def test_composite_geometry_really_does_change_the_score_distributions():
    """Otherwise the composite-null cells would be testing nothing."""
    from recompute.comparators.simulate import GEOMETRY_BY_NAME, make_dataset

    geom = GEOMETRY_BY_NAME["composite_shift_4"]
    y, s, codes = make_dataset(geom, rep=0, seed=42)
    c = codes["p0"]
    means = [s[c == k].mean() for k in np.unique(c)]
    assert max(means) - min(means) > 0.2


def test_simulation_is_reproducible():
    from recompute.comparators.simulate import GEOMETRY_BY_NAME, make_dataset

    g = GEOMETRY_BY_NAME["skewed_5"]
    a = make_dataset(g, 3, 42)
    b = make_dataset(g, 3, 42)
    assert np.array_equal(a[0], b[0])
    assert np.allclose(a[1], b[1])
    assert np.array_equal(a[2]["p0"], b[2]["p0"])
    c = make_dataset(g, 4, 42)
    assert not np.allclose(a[1], c[1])


def test_one_sim_returns_every_method():
    from recompute.comparators.simulate import GEOMETRY_BY_NAME
    from recompute.comparators.type1 import METHODS, _one_sim

    out = _one_sim(GEOMETRY_BY_NAME["balanced_3x1000"], 0, "m30", 49, 42, 0.05)
    for m in METHODS:
        assert m in out
        assert math.isnan(out[m]) or out[m] in (0.0, 1.0)


# ── 8. End-to-end shape of the deliverables ──────────────────────────────────
def test_run_cohorts_produces_every_cell(tmp_path):
    from recompute.comparators.run import run_cohorts

    df = run_cohorts(["german_credit"], n_perm=60, seed=42, alpha=0.05, jobs=1,
                     out=tmp_path / "c.csv")
    assert len(df) == 7 * len(RULES)
    assert set(df["method"]) == {
        "permutation_null", "diciccio2020", "lum2022", "lum2022_cochranQ",
        "lum2022_bootstrapCI", "four_fifths", "fixed_threshold_005"}
    assert set(df["conclusion"]) <= {"flag", "no_flag", "not_evaluable"}
    assert (df["runtime_s"] >= 0).all()


def test_every_method_reports_not_evaluable_rather_than_crashing():
    """UCI Heart has a 61-row test split and is not estimable at any rule."""
    data = load_cohort("uci_heart")
    for block in (diciccio.run_cohort(data, n_perm=30),
                  lum.run_cohort(data),
                  four_fifths.run_cohort(data),
                  naive.run_cohort(data)):
        for rule, res in block["results"].items():
            assert res.conclusion in ("flag", "no_flag", "not_evaluable")


@pytest.mark.slow
def test_comparison_csv_is_internally_consistent():
    """Guards the published artefact itself."""
    path = core.REPO / "recompute" / "results" / "comparator_comparison.csv"
    if not path.exists():
        pytest.skip("comparator_comparison.csv not generated yet")
    import pandas as pd

    df = pd.read_csv(path)
    assert len(df) == 10 * len(RULES) * 7
    # A p-value, where present, must agree with the conclusion.
    has_p = df["p_value"].notna() & (df["conclusion"] != "not_evaluable")
    sub = df[has_p]
    assert ((sub["p_value"] < 0.05) == (sub["conclusion"] == "flag")).all()
    # The naive baseline and the incumbent must share the observed statistic.
    piv = df.pivot_table(index=["cohort", "rule"], columns="method",
                         values="statistic")
    both = piv[["permutation_null", "fixed_threshold_005"]].dropna()
    assert np.allclose(both["permutation_null"], both["fixed_threshold_005"],
                       atol=1e-9)
