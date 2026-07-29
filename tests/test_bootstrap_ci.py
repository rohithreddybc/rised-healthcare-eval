"""Tests for the bootstrap / jackknife machinery, including clustering (F3)."""

import numpy as np
import pytest

from rised.bootstrap_ci import (
    MIN_RELIABLE_BOOTSTRAP,
    ResamplingPlan,
    bca_interval,
    bootstrap_replicates,
    holm_bonferroni,
    jackknife_from_plan,
    jackknife_replicates,
)


# ── row-level behaviour is unchanged ─────────────────────────────────────────
def test_unclustered_plan_matches_row_level_resampling():
    plan = ResamplingPlan(50)
    assert plan.clustered is False
    assert plan.n_units == 50
    rng = np.random.default_rng(0)
    idx = plan.bootstrap_index(rng)
    assert len(idx) == 50
    assert idx.min() >= 0 and idx.max() < 50


def test_unclustered_jackknife_leaves_out_one_row():
    plan = ResamplingPlan(10)
    replicates = list(plan.jackknife_index_iter())
    assert len(replicates) == 10
    for i, idx in enumerate(replicates):
        assert len(idx) == 9
        assert i not in set(idx.tolist())


def test_jackknife_replicates_default_is_unchanged():
    stat = lambda idx: float(idx.sum())
    legacy = jackknife_replicates(stat, 8)
    assert len(legacy) == 8
    assert legacy[0] == pytest.approx(sum(range(8)) - 0)
    assert legacy[7] == pytest.approx(sum(range(8)) - 7)


# ── F3: clustered resampling ─────────────────────────────────────────────────
def test_clustered_bootstrap_takes_whole_groups():
    groups = np.array([0, 0, 0, 1, 1, 2])
    plan = ResamplingPlan(6, groups)
    assert plan.clustered is True
    assert plan.n_units == 3
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx = plan.bootstrap_index(rng)
        picked = groups[idx]
        # Every sampled group contributes all of its rows.
        for g in np.unique(picked):
            count = int((picked == g).sum())
            assert count % int((groups == g).sum()) == 0


def test_clustered_jackknife_deletes_whole_groups():
    groups = np.array([0, 0, 0, 1, 1, 2])
    plan = ResamplingPlan(6, groups)
    replicates = list(plan.jackknife_index_iter())
    assert len(replicates) == 3
    left_out = []
    for idx in replicates:
        present = set(groups[idx].tolist())
        missing = {0, 1, 2} - present
        assert len(missing) == 1
        left_out.append(missing.pop())
    assert sorted(left_out) == [0, 1, 2]


def test_clustered_plan_describe_reports_structure():
    groups = np.repeat(np.arange(4), [1, 2, 3, 4])
    plan = ResamplingPlan(10, groups)
    info = plan.describe()
    assert info == {
        "clustered": True,
        "n_rows": 10,
        "n_units": 4,
        "mean_rows_per_group": pytest.approx(2.5),
        "max_rows_per_group": 4,
    }


def test_grouped_jackknife_via_legacy_entry_point():
    groups = np.array([0, 0, 1, 1])
    out = jackknife_replicates(lambda idx: float(len(idx)), 4, groups=groups)
    assert list(out) == [2.0, 2.0]


def test_plan_rejects_mismatched_groups_length():
    with pytest.raises(ValueError, match="length n"):
        ResamplingPlan(5, np.array([0, 1]))


def test_clustered_interval_is_wider_when_rows_are_correlated():
    """Ignoring clustering understates uncertainty."""
    rng = np.random.default_rng(4)
    n_groups, per = 40, 5
    groups = np.repeat(np.arange(n_groups), per)
    group_mean = rng.normal(0, 1.0, n_groups)
    values = group_mean[groups] + rng.normal(0, 0.1, n_groups * per)

    def stat(idx):
        return float(values[idx].mean())

    theta = stat(np.arange(len(values)))
    rng_b = np.random.default_rng(0)
    row_plan = ResamplingPlan(len(values))
    clus_plan = ResamplingPlan(len(values), groups)

    row_ci = bca_interval(
        theta,
        bootstrap_replicates(stat, row_plan, 400, rng_b),
        jackknife_from_plan(stat, row_plan),
    )
    rng_b = np.random.default_rng(0)
    clus_ci = bca_interval(
        theta,
        bootstrap_replicates(stat, clus_plan, 400, rng_b),
        jackknife_from_plan(stat, clus_plan),
    )
    assert (clus_ci[1] - clus_ci[0]) > (row_ci[1] - row_ci[0])


# ── BCa behaviour ────────────────────────────────────────────────────────────
def test_bca_drops_nan_replicates():
    boot = np.array([0.1, 0.2, np.nan, 0.3] * 100)
    jack = np.array([0.2, np.nan, 0.21])
    lo, hi = bca_interval(0.2, boot, jack)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_bca_returns_nan_without_usable_replicates():
    lo, hi = bca_interval(0.5, np.array([np.nan, np.nan]), np.array([0.5]))
    assert np.isnan(lo) and np.isnan(hi)


def test_bca_warns_at_small_bootstrap_counts():
    boot = np.linspace(0.0, 1.0, 30)
    with pytest.warns(UserWarning, match="usable bootstrap replicates"):
        bca_interval(0.5, boot, np.linspace(0.4, 0.6, 20))


def test_bca_does_not_warn_at_adequate_counts():
    import warnings

    boot = np.linspace(0.0, 1.0, MIN_RELIABLE_BOOTSTRAP)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bca_interval(0.5, boot, np.linspace(0.4, 0.6, 50))


def test_bca_contains_point_estimate_for_a_smooth_statistic():
    rng = np.random.default_rng(7)
    values = rng.normal(0.0, 1.0, 300)

    def stat(idx):
        return float(values[idx].mean())

    plan = ResamplingPlan(len(values))
    theta = stat(np.arange(len(values)))
    lo, hi = bca_interval(
        theta,
        bootstrap_replicates(stat, plan, 500, rng),
        jackknife_from_plan(stat, plan),
    )
    assert lo <= theta <= hi


# ── Holm-Bonferroni ──────────────────────────────────────────────────────────
def test_holm_bonferroni_rejects_only_the_smallest():
    rejected, alphas = holm_bonferroni([0.001, 0.20, 0.30])
    assert rejected[0] and not rejected[1] and not rejected[2]
    assert alphas[0] == pytest.approx(0.05 / 3)
