"""Tests for the Inclusivity dimension."""

import numpy as np
import pandas as pd
import pytest

from rised.inclusivity import evaluate_inclusivity
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


def test_small_group_flagged(fitted_lr, small_cohort):
    X, y = small_cohort
    n = len(X)
    rng = np.random.default_rng(99)
    demo = pd.DataFrame({
        "group": ["majority"] * (n - 5) + ["tiny"] * 5,
    })
    result = evaluate_inclusivity(fitted_lr, X, y, demo)
    assert any("tiny" in flag for flag in result.details["small_group_flags"])


def test_subgroup_columns_filter(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    result = evaluate_inclusivity(
        fitted_lr, X, y, demographic_df, subgroup_columns=["sex"]
    )
    for key in result.subgroup_aucs:
        assert key.startswith("sex=")


def test_passed_with_zero_gap():
    r = InclusivityResult(
        subgroup_aucs={"race=A": 0.75, "race=B": 0.75},
        auc_parity_gap=0.0,
        subgroup_calibration={"race=A": 0.05, "race=B": 0.04},
    )
    assert r.passed() is True


def test_failed_large_auc_gap():
    r = InclusivityResult(
        subgroup_aucs={"race=A": 0.85, "race=B": 0.70},
        auc_parity_gap=0.15,
        subgroup_calibration={"race=A": 0.05, "race=B": 0.04},
    )
    assert r.passed() is False


def test_failed_high_ece():
    r = InclusivityResult(
        subgroup_aucs={"race=A": 0.80, "race=B": 0.82},
        auc_parity_gap=0.02,
        subgroup_calibration={"race=A": 0.05, "race=B": 0.15},
    )
    assert r.passed() is False


def test_inclusivity_result_passed_none():
    r = InclusivityResult()
    assert r.passed() is False


def test_skips_subgroup_with_all_same_label(fitted_lr, small_cohort):
    X, y = small_cohort
    n = len(X)
    all_pos = pd.Series([1] * n, name="label")
    demo = pd.DataFrame({"group": ["A"] * n})
    result = evaluate_inclusivity(fitted_lr, X, all_pos, demo)
    assert len(result.subgroup_aucs) == 0
