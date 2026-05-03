"""Tests for the Sensitivity dimension."""

import numpy as np
import pytest

from rised.results import SensitivityResult
from rised.sensitivity import evaluate_sensitivity


def test_default_threshold_range_runs(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert isinstance(result, SensitivityResult)
    assert len(result.threshold_flip_rates) == 17


def test_custom_threshold_range(fitted_lr, small_cohort):
    X, y = small_cohort
    thresholds = np.array([0.3, 0.5, 0.7])
    result = evaluate_sensitivity(fitted_lr, X, y, threshold_range=thresholds)
    assert len(result.threshold_flip_rates) == 3


def test_flip_rates_nonnegative(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    for rate in result.threshold_flip_rates.values():
        assert rate >= 0.0


def test_flip_rate_at_reference_threshold_is_zero(fitted_lr, small_cohort):
    X, y = small_cohort
    thresholds = np.array([0.5])
    result = evaluate_sensitivity(fitted_lr, X, y, threshold_range=thresholds)
    assert result.threshold_flip_rates[0.5] == pytest.approx(0.0)


def test_flip_rate_increases_away_from_reference(fitted_lr, small_cohort):
    X, y = small_cohort
    thresholds = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
    result = evaluate_sensitivity(fitted_lr, X, y, threshold_range=thresholds)
    flip_at_01 = result.threshold_flip_rates[0.1]
    flip_at_049 = result.threshold_flip_rates[0.49]
    assert flip_at_01 >= flip_at_049


def test_rank_stability_in_range(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert 0.0 <= result.rank_stability_score <= 1.0


def test_decision_boundary_width_in_range(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert 0.0 <= result.decision_boundary_width <= 1.0


def test_details_contains_expected_keys(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert "reference_threshold" in result.details
    assert "boundary_delta" in result.details


def test_sensitivity_passed_low_flip():
    r = SensitivityResult(
        threshold_flip_rates={0.45: 0.02, 0.50: 0.0, 0.55: 0.03},
        rank_stability_score=0.98,
        decision_boundary_width=0.05,
    )
    assert r.passed(max_flip_rate=0.10) is True


def test_sensitivity_failed_high_flip():
    r = SensitivityResult(
        threshold_flip_rates={0.45: 0.15, 0.50: 0.0, 0.55: 0.18},
        rank_stability_score=0.82,
        decision_boundary_width=0.20,
    )
    assert r.passed(max_flip_rate=0.10) is False


def test_sensitivity_result_passed_no_data():
    r = SensitivityResult()
    assert r.passed() is False
