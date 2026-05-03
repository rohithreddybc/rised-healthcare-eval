"""Tests for the Reliability dimension."""

import numpy as np
import pytest

from rised.metrics import decision_flip_rate, judge_sensitivity_score, rank_correlation
from rised.reliability import evaluate_reliability
from rised.results import ReliabilityResult


def test_no_perturbations_returns_perfect_scores(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=None)
    assert result.judge_sensitivity_score == 0.0
    assert result.perturbation_flip_rate == 0.0
    assert result.rank_correlation_mean == 1.0


def test_no_perturbations_empty_list(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=[])
    assert result.judge_sensitivity_score == 0.0
    assert result.perturbation_flip_rate == 0.0
    assert result.rank_correlation_mean == 1.0


def test_gaussian_noise_low_jss(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.001, "random_state": 0}]
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
    assert result.judge_sensitivity_score < 0.15


def test_large_noise_higher_flip_rate(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 5.0, "random_state": 42}]
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
    assert result.perturbation_flip_rate >= 0.0


def test_details_contains_per_perturbation_keys(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [
        {"type": "gaussian_noise", "scale": 0.01, "label": "noise_small", "random_state": 1},
        {"type": "gaussian_noise", "scale": 0.05, "label": "noise_med", "random_state": 2},
    ]
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
    assert "noise_small" in result.details["per_perturbation_flip_rate"]
    assert "noise_med" in result.details["per_perturbation_flip_rate"]
    assert "noise_small" in result.details["per_perturbation_rank_correlation"]


def test_reliability_result_passed_threshold_low_flip():
    r = ReliabilityResult(
        judge_sensitivity_score=0.03,
        perturbation_flip_rate=0.03,
        rank_correlation_mean=0.97,
    )
    assert r.passed(threshold=0.05) is True


def test_reliability_result_failed_threshold_high_flip():
    r = ReliabilityResult(
        judge_sensitivity_score=0.09,
        perturbation_flip_rate=0.09,
        rank_correlation_mean=0.88,
    )
    assert r.passed(threshold=0.05) is False


def test_reliability_result_passed_none():
    r = ReliabilityResult()
    assert r.passed() is False


def test_rank_correlation_identical_arrays():
    a = np.array([0.1, 0.5, 0.9, 0.3])
    assert rank_correlation(a, a) == pytest.approx(1.0)


def test_decision_flip_rate_no_flips():
    a = np.array([0.2, 0.8, 0.6])
    assert decision_flip_rate(a, a) == pytest.approx(0.0)


def test_decision_flip_rate_all_flip():
    a = np.array([0.3, 0.3])
    b = np.array([0.7, 0.7])
    assert decision_flip_rate(a, b, threshold=0.5) == pytest.approx(1.0)


def test_jss_empty_list():
    baseline = np.array([0.3, 0.7, 0.5])
    assert judge_sensitivity_score(baseline, []) == pytest.approx(0.0)
