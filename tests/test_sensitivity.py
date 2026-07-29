"""Tests for the Sensitivity dimension."""

import numpy as np
import pytest

from rised.results import SensitivityResult
from rised.sensitivity import (
    NARROW_THRESHOLD_BAND,
    WIDE_THRESHOLD_BAND,
    evaluate_sensitivity,
    prevalence_matched_threshold,
    suggest_tau_ref,
    youden_j_threshold,
)


def test_default_band_is_the_narrow_band(fitted_lr, small_cohort):
    """F5: the primary report is the narrow [0.30, 0.70] band, not [0.1, 0.9]."""
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert isinstance(result, SensitivityResult)
    assert len(result.threshold_flip_rates) == len(NARROW_THRESHOLD_BAND)
    assert min(result.threshold_flip_rates) == pytest.approx(0.30)
    assert max(result.threshold_flip_rates) == pytest.approx(0.70)
    assert result.details["threshold_band_name"] == "narrow[0.30,0.70]"


def test_wide_band_also_computed_and_reported(fitted_lr, small_cohort):
    """F5: both bands are computable; the wide one is secondary."""
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert len(result.wide_band_flip_rates) == len(WIDE_THRESHOLD_BAND)
    assert min(result.wide_band_flip_rates) == pytest.approx(0.10)
    assert max(result.wide_band_flip_rates) == pytest.approx(0.90)
    assert result.wide_band_max_tfr >= result.max_threshold_flip_rate - 1e-12


def test_custom_threshold_range(fitted_lr, small_cohort):
    X, y = small_cohort
    thresholds = np.array([0.3, 0.5, 0.7])
    result = evaluate_sensitivity(fitted_lr, X, y, threshold_range=thresholds)
    assert len(result.threshold_flip_rates) == 3
    assert result.details["threshold_band_name"] == "custom"


def test_flip_rates_nonnegative(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    for rate in result.threshold_flip_rates.values():
        assert rate >= 0.0


def test_flip_rate_at_reference_threshold_is_zero(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(
        fitted_lr, X, y, threshold_range=np.array([0.5]), tau_ref=0.5
    )
    assert result.threshold_flip_rates[0.5] == pytest.approx(0.0)


def test_flip_rate_increases_away_from_reference(fitted_lr, small_cohort):
    X, y = small_cohort
    thresholds = np.array([0.1, 0.49, 0.5, 0.51, 0.9])
    result = evaluate_sensitivity(fitted_lr, X, y, threshold_range=thresholds)
    assert result.threshold_flip_rates[0.1] >= result.threshold_flip_rates[0.49]


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
    assert result.details["uses_y_true"] is False


# ── F4: per-cohort reference threshold ───────────────────────────────────────
def test_f4_tau_ref_is_honoured(fitted_lr, small_cohort):
    """tau_ref must actually move the reference point, not be ignored."""
    X, y = small_cohort
    band = np.array([0.2, 0.35, 0.5, 0.65])
    at_half = evaluate_sensitivity(fitted_lr, X, y, threshold_range=band, tau_ref=0.5)
    at_035 = evaluate_sensitivity(fitted_lr, X, y, threshold_range=band, tau_ref=0.35)
    assert at_half.threshold_flip_rates[0.5] == pytest.approx(0.0)
    assert at_035.threshold_flip_rates[0.35] == pytest.approx(0.0)
    assert at_half.details["reference_threshold"] == 0.5
    assert at_035.details["reference_threshold"] == 0.35
    assert at_half.threshold_flip_rates != at_035.threshold_flip_rates


def test_f4_evaluate_all_threads_tau_ref(fitted_lr, small_cohort, demographic_df):
    """F4: evaluate_all previously had no tau_ref parameter at all."""
    from rised import evaluate_all

    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df, tau_ref=0.35)
    assert report.sensitivity.details["reference_threshold"] == 0.35
    assert report.reliability.details["reference_threshold"] == 0.35
    assert report.metadata["tau_ref"] == 0.35


def test_f4_prevalence_matched_threshold_matches_positive_rate():
    scores = np.linspace(0.0, 1.0, 1000)
    tau = prevalence_matched_threshold(scores, 0.20)
    assert np.mean(scores >= tau) == pytest.approx(0.20, abs=0.01)


def test_f4_youden_threshold_on_separable_scores():
    y = np.array([0] * 50 + [1] * 50)
    scores = np.concatenate([np.full(50, 0.2), np.full(50, 0.8)])
    tau = youden_j_threshold(y, scores)
    assert 0.2 < tau <= 0.8


def test_f4_suggest_tau_ref_returns_candidates_without_applying(fitted_lr, small_cohort):
    """The helper offers thresholds; nothing applies them silently."""
    X, y = small_cohort
    scores = fitted_lr.predict_proba(np.asarray(X, dtype=float))[:, 1]
    suggestions = suggest_tau_ref(scores, y)
    assert set(suggestions) == {"default", "prevalence_matched", "youden_j"}
    assert suggestions["default"] == 0.5
    assert suggestions["prevalence_matched"] is not None

    # Running the evaluation without passing one keeps the 0.5 default.
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert result.details["reference_threshold"] == 0.5


def test_f4_suggest_tau_ref_without_labels():
    assert suggest_tau_ref(np.linspace(0, 1, 50))["prevalence_matched"] is None


def test_prevalence_matched_rejects_degenerate_prevalence():
    with pytest.raises(ValueError):
        prevalence_matched_threshold(np.linspace(0, 1, 10), 0.0)


# ── TFR is outcome-free ──────────────────────────────────────────────────────
def test_tfr_is_invariant_to_label_permutation(fitted_lr, small_cohort):
    """Verified property P4a: y_true has no effect on any Sensitivity output."""
    X, y = small_cohort
    rng = np.random.default_rng(0)
    permuted = np.asarray(y)[rng.permutation(len(y))]
    a = evaluate_sensitivity(fitted_lr, X, y)
    b = evaluate_sensitivity(fitted_lr, X, permuted)
    c = evaluate_sensitivity(fitted_lr, X, None)
    assert a.threshold_flip_rates == b.threshold_flip_rates == c.threshold_flip_rates
    assert a.rank_stability_score == c.rank_stability_score


def test_constant_model_attains_perfect_tfr(small_cohort):
    """A useless constant predictor scores perfectly; TFR must not be read alone."""
    X, y = small_cohort

    class Constant:
        def predict_proba(self, X):
            p = np.full(len(X), 0.05)
            return np.column_stack([1 - p, p])

    result = evaluate_sensitivity(Constant(), X, y)
    assert result.max_threshold_flip_rate == pytest.approx(0.0)
    assert result.rank_stability_score == pytest.approx(1.0)


# ── result object ────────────────────────────────────────────────────────────
def test_max_threshold_flip_rate_field_matches_dict(fitted_lr, small_cohort):
    X, y = small_cohort
    result = evaluate_sensitivity(fitted_lr, X, y)
    assert result.max_threshold_flip_rate == pytest.approx(
        max(result.threshold_flip_rates.values())
    )


def test_sensitivity_passed_is_withdrawn():
    with pytest.raises(NotImplementedError, match="withdrawn"):
        SensitivityResult(threshold_flip_rates={0.5: 0.0}).passed()
