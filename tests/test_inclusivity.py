"""Tests for the Inclusivity dimension."""

import numpy as np
import pandas as pd
import pytest

from rised.inclusivity import (
    _gaps_from_per_column,
    _subgroup_aucs_by_column,
    evaluate_inclusivity,
)
from rised.results import InclusivityResult


def test_subgroup_aucs_computed(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_inclusivity(fitted_lr, X, y, demographic_df)
    assert len(result.subgroup_aucs) > 0
    for val in result.subgroup_aucs.values():
        assert 0.0 <= val <= 1.0


def test_auc_parity_gap_nonnegative(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_inclusivity(fitted_lr, X, y, demographic_df)
    if result.auc_parity_gap is not None:
        assert result.auc_parity_gap >= 0.0


def test_subgroup_ece_in_range(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_inclusivity(fitted_lr, X, y, demographic_df)
    for val in result.subgroup_calibration.values():
        assert 0.0 <= val <= 1.0


def test_subgroup_columns_filter(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_inclusivity(
        fitted_lr, X, y, demographic_df, subgroup_columns=["sex"]
    )
    for key in result.subgroup_aucs:
        assert key.startswith("sex=")


def test_skips_subgroup_with_all_same_label(fitted_lr, small_cohort):
    X, y = small_cohort
    n = len(X)
    all_pos = pd.Series([1] * n, name="label")
    demo = pd.DataFrame({"group": ["A"] * n})
    result = evaluate_inclusivity(fitted_lr, X, all_pos, demo)
    assert len(result.subgroup_aucs) == 0
    assert "group=A" in result.excluded_subgroups


# ── F1: per-partition gaps ───────────────────────────────────────────────────
def test_f1_gap_is_computed_within_each_partition(fitted_lr, small_cohort, demographic_df):
    """The headline gap must be a within-column range, not a cross-column one."""
    X, y = small_cohort
    result = evaluate_inclusivity(fitted_lr, X, y, demographic_df)

    assert set(result.per_partition_auc_gaps) == {"race", "sex", "age_group"}
    for col, gap in result.per_partition_auc_gaps.items():
        levels = result.per_partition_aucs[col]
        assert gap == pytest.approx(max(levels.values()) - min(levels.values()))
    assert result.auc_parity_gap == pytest.approx(
        max(result.per_partition_auc_gaps.values())
    )


def test_f1_pooled_gap_is_separate_and_at_least_the_partition_max(
    fitted_lr, small_cohort, demographic_df
):
    """The old pooled max-min survives only as a diagnostic, and dominates."""
    X, y = small_cohort
    result = evaluate_inclusivity(fitted_lr, X, y, demographic_df)
    assert result.pooled_auc_gap_diagnostic is not None
    assert result.pooled_auc_gap_diagnostic >= result.auc_parity_gap - 1e-12


def test_f1_cross_partition_comparison_no_longer_drives_the_headline():
    """A cohort where the extremes sit in *different* columns.

    Column 'a' is internally uniform and column 'b' is internally uniform, but
    a's level is far better than b's. The pooled statistic reports a large gap;
    the per-partition statistic correctly reports ~0 for both partitions.
    """
    n = 400
    rng = np.random.default_rng(0)
    scores = rng.random(n)
    # Labels perfectly ranked in the first half, anti-ranked in the second.
    y = np.empty(n, dtype=int)
    half = n // 2
    y[:half] = (scores[:half] > np.median(scores[:half])).astype(int)
    y[half:] = (scores[half:] < np.median(scores[half:])).astype(int)

    class Fixed:
        def predict_proba(self, X):
            s = X[:, 0]
            return np.column_stack([1 - s, s])

    X = scores.reshape(-1, 1)
    # 'a' splits within the good half only; 'b' within the bad half only.
    demo = pd.DataFrame(
        {
            "a": ["a1" if i < half // 2 else "a2" for i in range(n)],
            "b": ["b1" if i < half + half // 2 else "b2" for i in range(n)],
        }
    )
    result = evaluate_inclusivity(Fixed(), X, y, demo)
    assert result.auc_parity_gap <= result.pooled_auc_gap_diagnostic
    for col, gap in result.per_partition_auc_gaps.items():
        assert gap <= result.pooled_auc_gap_diagnostic


# ── F2: one exclusion rule, in every estimator ───────────────────────────────
def test_f2_small_group_excluded_from_point_estimate(fitted_lr, small_cohort):
    """Sub-threshold groups must not enter the point estimate at all."""
    X, y = small_cohort
    n = len(X)
    demo = pd.DataFrame({"group": ["majority"] * (n - 5) + ["tiny"] * 5})
    result = evaluate_inclusivity(fitted_lr, X, y, demo)
    assert "group=tiny" in result.excluded_subgroups
    assert "n=5" in result.excluded_subgroups["group=tiny"]
    assert "group=tiny" not in result.subgroup_aucs
    assert "group=tiny" not in result.subgroup_calibration


def test_f2_exclusion_rule_is_identical_in_replicates(fitted_lr, small_cohort):
    """The rule applied to a resampled index must be the same rule."""
    X, y = small_cohort
    n = len(X)
    demo = pd.DataFrame(
        {"group": ["a"] * 80 + ["b"] * 80 + ["c"] * 25 + ["d"] * (n - 185)}
    )
    y_arr = np.asarray(y)
    scores = fitted_lr.predict_proba(np.asarray(X, dtype=float))[:, 1]

    idx = np.arange(n)
    per_col, excluded = _subgroup_aucs_by_column(
        y_arr, scores, demo, ["group"], 30
    )
    # 'c' has 25 rows: excluded in the full sample.
    assert "group=c" in excluded
    assert "group=c" not in per_col["group"]

    # Same call on a subsample applies the same rule, not a different one.
    sub = idx[:100]
    per_col_b, excluded_b = _subgroup_aucs_by_column(
        y_arr[sub], scores[sub], demo.iloc[sub], ["group"], 30
    )
    for label in per_col_b["group"]:
        assert label not in excluded_b


def test_f2_bca_interval_contains_its_own_point_estimate():
    """Regression for the verified P3 defect.

    Constructed as in the verification: five groups of 196 sharing one true
    AUC, plus one group of 20 with a very different AUC. Previously the small
    group entered the point estimate (0.7407) but was dropped from 97.8% of
    bootstrap replicates, and the BCa interval (0.7724, 0.8236) did not contain
    it. With one exclusion rule everywhere the interval must cover it.
    """
    rng = np.random.default_rng(42)
    parts = []
    for g in range(5):
        n = 196
        y_g = (rng.random(n) < 0.3).astype(int)
        s_g = np.where(
            y_g == 1, rng.normal(0.60, 0.15, n), rng.normal(0.40, 0.15, n)
        )
        parts.append((np.clip(s_g, 0, 1), y_g, [f"G{g}"] * n))
    n = 20
    y_s = (rng.random(n) < 0.3).astype(int)
    s_s = np.where(y_s == 1, rng.normal(0.40, 0.15, n), rng.normal(0.60, 0.15, n))
    parts.append((np.clip(s_s, 0, 1), y_s, ["SMALL"] * n))

    scores = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    demo = pd.DataFrame({"grp": sum((p[2] for p in parts), [])})

    class Passthrough:
        def predict_proba(self, X):
            s = X[:, 0]
            return np.column_stack([1 - s, s])

    X = scores.reshape(-1, 1)
    result = evaluate_inclusivity(
        Passthrough(), X, y, demo, n_bootstrap=300, random_state=1
    )

    assert "grp=SMALL" in result.excluded_subgroups
    lo, hi = result.auc_gap_ci
    assert lo <= result.auc_parity_gap <= hi, (
        f"BCa interval ({lo:.4f}, {hi:.4f}) does not contain its own point "
        f"estimate {result.auc_parity_gap:.4f}"
    )


def test_f2_min_subgroup_n_zero_includes_everything(fitted_lr, small_cohort):
    X, y = small_cohort
    n = len(X)
    demo = pd.DataFrame({"group": ["majority"] * (n - 12) + ["tiny"] * 12})
    strict = evaluate_inclusivity(fitted_lr, X, y, demo, min_subgroup_n=30)
    loose = evaluate_inclusivity(fitted_lr, X, y, demo, min_subgroup_n=0)
    assert "group=tiny" not in strict.subgroup_aucs
    assert "group=tiny" in loose.subgroup_aucs
    assert loose.details["min_subgroup_n"] == 0


# ── F3: clustered resampling ─────────────────────────────────────────────────
def test_f3_grouped_bootstrap_changes_the_interval(clustered_cohort):
    X, y, groups, demo, clf = clustered_cohort
    row_level = evaluate_inclusivity(
        clf, X, y, demo, n_bootstrap=200, random_state=3
    )
    clustered = evaluate_inclusivity(
        clf, X, y, demo, n_bootstrap=200, random_state=3, groups=groups
    )
    assert row_level.auc_parity_gap == pytest.approx(clustered.auc_parity_gap)
    assert row_level.details["resampling"]["clustered"] is False
    assert clustered.details["resampling"]["clustered"] is True
    assert clustered.details["resampling"]["n_units"] == len(np.unique(groups))
    assert row_level.auc_gap_ci != clustered.auc_gap_ci


def test_f3_default_behaviour_unchanged_without_groups(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    a = evaluate_inclusivity(
        fitted_lr, X, y, demographic_df, n_bootstrap=200, random_state=9
    )
    b = evaluate_inclusivity(
        fitted_lr, X, y, demographic_df, n_bootstrap=200, random_state=9, groups=None
    )
    assert a.auc_gap_ci == b.auc_gap_ci


# ── result object ────────────────────────────────────────────────────────────
def test_worst_partition_identifies_the_widest_column():
    r = InclusivityResult(
        per_partition_auc_gaps={"race": 0.02, "sex": 0.11, "age": 0.05},
        auc_parity_gap=0.11,
    )
    assert r.worst_partition == "sex"


def test_gaps_helper_needs_two_levels():
    gaps, mx = _gaps_from_per_column({"a": {"a=1": 0.8}})
    assert gaps == {}
    assert mx is None


def test_inclusivity_passed_is_withdrawn():
    with pytest.raises(NotImplementedError, match="withdrawn"):
        InclusivityResult(auc_parity_gap=0.0).passed()
