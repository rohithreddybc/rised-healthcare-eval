"""Tests for the Deployability dimension."""

import numpy as np
import pytest

from rised.deployability import evaluate_deployability, explanation_chance_level
from rised.results import DeployabilityResult


# ── F9: timing named for what it measures ────────────────────────────────────
def test_f9_batch_time_and_amortised_time_are_distinct_fields(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=5, n_shap_samples=20
    )
    assert result.batch_scoring_time_ms > 0.0
    assert result.amortised_time_per_row_ms == pytest.approx(
        result.batch_scoring_time_ms / X.shape[0]
    )
    assert not hasattr(result, "mean_inference_latency_ms")


def test_f9_single_row_latency_is_measured_separately(fitted_lr, small_cohort):
    """The per-request quantity is not the batch time divided by n."""
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=5, n_shap_samples=20, n_single_row_trials=10
    )
    assert result.single_row_latency_ms is not None
    assert result.single_row_latency_ms > 0.0
    # Scoring one row costs far more than the amortised per-row share.
    assert result.single_row_latency_ms > result.amortised_time_per_row_ms


def test_f9_timing_note_documents_the_distinction(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(fitted_lr, X, n_latency_trials=3)
    note = result.details["timing_note"]
    assert "not" in note and "per-request" in note
    assert "batch_scoring_time_std_ms" in result.details


def test_f9_single_row_trials_can_be_disabled(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_single_row_trials=0
    )
    assert result.single_row_latency_ms is None


# ── F9: the explanation statistic is no longer degenerate or misnamed ────────
def test_f9_low_dimensional_cohort_no_longer_yields_one(low_dimensional_cohort):
    """Regression: d <= 3 previously returned exactly 1.0 in 40/40 runs."""
    X, y, clf = low_dimensional_cohort
    result = evaluate_deployability(
        clf, X, n_latency_trials=3, n_shap_samples=20, top_k=3
    )
    assert result.local_global_topk_agreement is None
    assert result.global_top1_in_local_topk is None
    reason = result.details["explanation_metrics_undefined_reason"]
    assert "d=3" in reason and "top_k=3" in reason


def test_f9_faithfulness_name_is_gone(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20
    )
    assert not hasattr(result, "explanation_faithfulness")
    assert not hasattr(result, "top_feature_stability")
    assert hasattr(result, "local_global_topk_agreement")
    assert hasattr(result, "global_top1_in_local_topk")


def test_f9_global_reference_sample_is_disjoint_from_scored_sample(
    fitted_lr, small_cohort
):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=40
    )
    if result.local_global_topk_agreement is not None:
        assert result.details["explanation_reference"] == "disjoint sample"
        n_ref = result.details["n_reference_rows"]
        n_scored = result.details["n_scored_rows"]
        assert n_ref >= 2 and n_scored >= 2
        assert n_ref + n_scored <= X.shape[0]


def test_f9_chance_level_is_reported(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20, top_k=3
    )
    assert result.details["top_k"] == 3
    assert result.details["explanation_chance_level"] == pytest.approx(3 / 10)
    assert "0.394" in result.details["explanation_null_note"]


def test_f9_chance_level_helper():
    assert explanation_chance_level(10, 3) == pytest.approx(0.3)
    assert explanation_chance_level(3, 3) == pytest.approx(1.0)
    assert explanation_chance_level(2, 3) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        explanation_chance_level(0)


def test_f9_tiny_cohort_cannot_be_split(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X[:3], n_latency_trials=2, n_single_row_trials=1
    )
    assert result.local_global_topk_agreement is None
    assert "disjoint" in result.details["explanation_metrics_undefined_reason"]


# ── general behaviour ────────────────────────────────────────────────────────
def test_deployability_runs(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=5, n_shap_samples=20
    )
    assert isinstance(result, DeployabilityResult)


def test_shap_runs_or_records_error(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20
    )
    ran = result.local_global_topk_agreement is not None
    errored = "shap_error" in result.details
    undefined = "explanation_metrics_undefined_reason" in result.details
    assert ran or errored or undefined


def test_agreement_metrics_in_range(fitted_lr, small_cohort):
    X, _ = small_cohort
    result = evaluate_deployability(
        fitted_lr, X, n_latency_trials=3, n_shap_samples=20
    )
    for value in (
        result.local_global_topk_agreement,
        result.global_top1_in_local_topk,
    ):
        if value is not None:
            assert 0.0 <= value <= 1.0


def test_feature_names_stored_in_details(fitted_lr, small_cohort):
    X, _ = small_cohort
    names = [f"feature_{i}" for i in range(10)]
    result = evaluate_deployability(
        fitted_lr, X, feature_names=names, n_latency_trials=3, n_shap_samples=20
    )
    if result.local_global_topk_agreement is not None:
        assert "global_top_features" in result.details
        assert all(n.startswith("feature_") for n in result.details["global_top_features"])


def test_deployability_passed_is_withdrawn():
    with pytest.raises(NotImplementedError, match="withdrawn"):
        DeployabilityResult(batch_scoring_time_ms=1.0).passed()
