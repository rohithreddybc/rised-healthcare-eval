"""
Integration tests for the RISED evaluate_all() entry point.
"""

import pytest

import rised
from rised import evaluate_all
from rised.results import FrameworkReport


def test_evaluate_all_returns_framework_report(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert isinstance(report, FrameworkReport)


def test_evaluate_all_all_dimensions_populated(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert report.reliability is not None
    assert report.inclusivity is not None
    assert report.sensitivity is not None
    assert report.equity is not None
    assert report.deployability is not None


def test_evaluate_all_summary_has_five_keys(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    s = report.summary()
    assert set(s.keys()) == {"reliability", "inclusivity", "sensitivity", "equity", "deployability"}


def test_evaluate_all_metadata_contains_shape(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert report.metadata["n_samples"] == 200
    assert report.metadata["n_features"] == 10


def test_evaluate_all_with_perturbation_specs(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.01, "random_state": 0}]
    report = evaluate_all(fitted_lr, X, y, demographic_df, perturbation_specs=specs)
    assert report.reliability.judge_sensitivity_score is not None


def test_framework_report_summary_all_none():
    report = FrameworkReport()
    s = report.summary()
    assert all(v is None for v in s.values())
    assert report.all_passed() is False


def test_framework_report_all_passed_requires_all_true():
    from rised.results import (
        DeployabilityResult,
        EquityResult,
        InclusivityResult,
        ReliabilityResult,
        SensitivityResult,
    )

    report = FrameworkReport(
        reliability=ReliabilityResult(judge_sensitivity_score=0.02),
        inclusivity=InclusivityResult(auc_parity_gap=0.03, subgroup_calibration={"g=A": 0.05}),
        sensitivity=SensitivityResult(threshold_flip_rates={0.5: 0.0, 0.6: 0.04}),
        equity=EquityResult(need_prediction_correlation=0.75),
        deployability=DeployabilityResult(mean_inference_latency_ms=200.0),
    )
    assert report.all_passed() is True


def test_evaluate_all_reproducible_with_random_state(fitted_lr, small_cohort, demographic_df):
    """evaluate_all() with same random_state must produce identical CI values."""
    X, y = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.05, "random_state": 0, "label": "n5"}]
    r1 = evaluate_all(
        fitted_lr, X, y, demographic_df,
        perturbation_specs=specs, random_state=42, n_bootstrap=20,
    )
    r2 = evaluate_all(
        fitted_lr, X, y, demographic_df,
        perturbation_specs=specs, random_state=42, n_bootstrap=20,
    )
    assert r1.reliability.judge_sensitivity_score == r2.reliability.judge_sensitivity_score
    assert r1.reliability.jss_ci == r2.reliability.jss_ci
    assert r1.sensitivity.max_tfr_ci == r2.sensitivity.max_tfr_ci
    assert r1.inclusivity.auc_gap_ci == r2.inclusivity.auc_gap_ci


def test_deployability_per_patient_latency(fitted_lr, small_cohort):
    """per-patient latency must equal batch latency / n_samples."""
    from rised.deployability import evaluate_deployability
    X, _ = small_cohort
    result = evaluate_deployability(fitted_lr, X, n_latency_trials=5, n_shap_samples=10)
    assert result.mean_latency_per_patient_ms is not None
    expected = result.mean_inference_latency_ms / X.shape[0]
    assert abs(result.mean_latency_per_patient_ms - expected) < 1e-10


def test_package_version():
    assert rised.__version__ == "0.1.0"
