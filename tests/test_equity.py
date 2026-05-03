"""Tests for the Equity dimension."""

import numpy as np
import pandas as pd
import pytest

from rised.equity import evaluate_equity
from rised.results import EquityResult


def test_equity_runs(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_equity(fitted_lr, X, y, demographic_df)
    assert isinstance(result, EquityResult)


def test_need_prediction_correlation_in_range(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_equity(fitted_lr, X, y, demographic_df)
    assert -1.0 <= result.need_prediction_correlation <= 1.0


def test_group_need_gaps_cover_all_subgroups(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_equity(fitted_lr, X, y, demographic_df, subgroup_columns=["sex"])
    assert len(result.group_need_gaps) >= 1
    for key in result.group_need_gaps:
        assert key.startswith("sex=")


def test_need_column_used_when_provided(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    n = len(X)
    demo_with_need = demographic_df.copy()
    rng = np.random.default_rng(7)
    demo_with_need["comorbidity"] = rng.integers(0, 10, size=n).astype(float)
    result = evaluate_equity(
        fitted_lr, X, y, demo_with_need,
        need_column="comorbidity",
        subgroup_columns=["sex"],
    )
    assert result.details["need_source"] == "comorbidity"
    assert -1.0 <= result.need_prediction_correlation <= 1.0


def test_need_source_is_y_true_when_no_column(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_equity(fitted_lr, X, y, demographic_df)
    assert result.details["need_source"] == "y_true"


def test_proxy_bias_flags_are_subset_of_gap_keys(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_equity(fitted_lr, X, y, demographic_df)
    for flag in result.proxy_bias_flags:
        assert flag in result.group_need_gaps


def test_equity_passed_high_correlation():
    r = EquityResult(
        need_prediction_correlation=0.80,
        group_need_gaps={"race=A": 0.05, "race=B": -0.03},
        proxy_bias_flags=[],
    )
    assert r.passed(min_correlation=0.70) is True


def test_equity_failed_low_correlation():
    r = EquityResult(
        need_prediction_correlation=0.50,
        group_need_gaps={"race=A": 0.05},
        proxy_bias_flags=[],
    )
    assert r.passed(min_correlation=0.70) is False


def test_equity_result_passed_none():
    r = EquityResult()
    assert r.passed() is False


def test_small_groups_below_5_skipped(fitted_lr, small_cohort):
    X, y = small_cohort
    n = len(X)
    demo = pd.DataFrame({
        "group": ["majority"] * (n - 3) + ["tiny"] * 3,
    })
    result = evaluate_equity(fitted_lr, X, y, demo)
    for key in result.group_need_gaps:
        assert "tiny" not in key
