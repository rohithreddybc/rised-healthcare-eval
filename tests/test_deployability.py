"""Tests for the Deployability dimension."""

import pytest

from rised.deployability import evaluate_deployability
from rised.results import DeployabilityResult


def test_deployability_runs(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(fitted_lr, X, n_latency_trials=5, n_shap_samples=20)
    assert isinstance(result, DeployabilityResult)


def test_latency_positive(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(fitted_lr, X, n_latency_trials=5)
    assert result.mean_inference_latency_ms > 0.0


def test_latency_std_in_details(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(fitted_lr, X, n_latency_trials=5)
    assert "latency_std_ms" in result.details
    assert result.details["latency_std_ms"] >= 0.0


def test_shap_runs_or_records_error(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20
    )
    shap_ran = result.explanation_faithfulness is not None
    shap_errored = "shap_error" in result.details
    assert shap_ran or shap_errored


def test_shap_faithfulness_in_range(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20
    )
    if result.explanation_faithfulness is not None:
        assert 0.0 <= result.explanation_faithfulness <= 1.0


def test_top_feature_stability_in_range(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20
    )
    if result.top_feature_stability is not None:
        assert 0.0 <= result.top_feature_stability <= 1.0


def test_feature_names_stored_in_details(fitted_lr, small_cohort):
    X, _ = small_cohort
    names = [f"feature_{i}" for i in range(10)]
    result = evaluate_deployability(
        fitted_lr, X, feature_names=names, n_latency_trials=3, n_shap_samples=20
    )
    if result.explanation_faithfulness is not None:
        assert "global_top_features" in result.details
        assert all(n.startswith("feature_") for n in result.details["global_top_features"])


def test_deployability_passed_within_latency():
    r = DeployabilityResult(
        mean_inference_latency_ms=120.0,
        explanation_faithfulness=0.70,
        top_feature_stability=0.65,
    )
    assert r.passed(max_latency_ms=500.0) is True


def test_deployability_failed_high_latency():
    r = DeployabilityResult(
        mean_inference_latency_ms=750.0,
        explanation_faithfulness=0.70,
        top_feature_stability=0.65,
    )
    assert r.passed(max_latency_ms=500.0) is False


def test_deployability_result_passed_none():
    r = DeployabilityResult()
    assert r.passed() is False
