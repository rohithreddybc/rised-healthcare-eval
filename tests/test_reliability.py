"""Tests for the Reliability dimension and the perturbation semantics."""

import numpy as np
import pytest

from rised.metrics import decision_flip_rate, judge_sensitivity_score, rank_correlation
from rised.perturbations import (
    BINARY,
    CATEGORICAL,
    CONTINUOUS,
    COVARIATE_SHIFT,
    ORDINAL,
    SEMANTICS_PRESERVING,
    FeatureSchema,
    apply_perturbation,
    gaussian_noise,
    ordinal_jitter,
    perturbation_semantics,
)
from rised.reliability import evaluate_reliability
from rised.results import ReliabilityResult


# ── F6: typed feature schema ─────────────────────────────────────────────────
def test_f6_schema_inference_types_columns(mixed_type_cohort):
    X, y, names, clf = mixed_type_cohort
    schema = FeatureSchema.infer(X, names=names)
    assert schema.type_of(0) == CONTINUOUS
    assert schema.type_of(1) == CONTINUOUS
    assert schema.type_of(2) == BINARY
    assert schema.type_of(3) == CATEGORICAL
    assert schema.indices_of(CONTINUOUS) == [0, 1]


def test_f6_binary_and_categorical_never_receive_continuous_noise(mixed_type_cohort):
    """The core F6 requirement: no impossible patients."""
    X, y, names, clf = mixed_type_cohort
    perturbed = gaussian_noise(X, scale=0.5, random_state=0)

    binary_before, binary_after = X[:, 2], perturbed[:, 2]
    cat_before, cat_after = X[:, 3], perturbed[:, 3]
    assert np.array_equal(binary_before, binary_after)
    assert np.array_equal(cat_before, cat_after)
    assert set(np.unique(binary_after)).issubset({0.0, 1.0})
    assert set(np.unique(cat_after)).issubset(set(np.unique(cat_before)))
    # Continuous columns did move.
    assert not np.array_equal(X[:, 0], perturbed[:, 0])


def test_f6_naming_a_binary_column_explicitly_raises(mixed_type_cohort):
    X, y, names, clf = mixed_type_cohort
    schema = FeatureSchema.infer(X, names=names)
    with pytest.raises(ValueError, match="Refusing to add continuous"):
        gaussian_noise(X, feature_indices=[2], scale=0.1, schema=schema)


def test_f6_legacy_all_column_noise_requires_explicit_optin(mixed_type_cohort):
    X, y, names, clf = mixed_type_cohort
    with pytest.warns(UserWarning, match="cannot occur"):
        perturbed = gaussian_noise(
            X, scale=0.5, random_state=0, respect_schema=False
        )
    assert not np.array_equal(X[:, 2], perturbed[:, 2])


def test_f6_ordinal_jitter_stays_on_the_level_grid(mixed_type_cohort):
    X, y, names, clf = mixed_type_cohort
    schema = FeatureSchema(
        types=[CONTINUOUS, CONTINUOUS, BINARY, ORDINAL], names=names
    )
    perturbed = ordinal_jitter(X, feature_indices=[3], random_state=1, schema=schema)
    levels = set(np.unique(X[:, 3]))
    assert set(np.unique(perturbed[:, 3])).issubset(levels)
    assert np.abs(perturbed[:, 3] - X[:, 3]).max() <= 1.0
    assert not np.array_equal(X[:, 3], perturbed[:, 3])


def test_f6_ordinal_jitter_refuses_non_ordinal_columns(mixed_type_cohort):
    X, y, names, clf = mixed_type_cohort
    schema = FeatureSchema.infer(X, names=names)
    with pytest.raises(ValueError, match="requires ordinal columns"):
        ordinal_jitter(X, feature_indices=[3], schema=schema)


def test_f6_multiplicative_rescaling_is_covariate_shift_not_reliability():
    """age * 1.05 changes the patient, not the encoding."""
    assert perturbation_semantics(
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05}
    ) == COVARIATE_SHIFT
    assert perturbation_semantics(
        {"type": "gaussian_noise", "scale": 0.01}
    ) == SEMANTICS_PRESERVING
    # A documented unit conversion may opt in explicitly.
    assert perturbation_semantics(
        {
            "type": "unit_rescaling",
            "feature_index": 0,
            "factor": 2.20462,
            "semantics": SEMANTICS_PRESERVING,
        }
    ) == SEMANTICS_PRESERVING


def test_f6_covariate_shift_excluded_from_jss(mixed_type_cohort):
    X, y, names, clf = mixed_type_cohort
    specs = [
        {"type": "gaussian_noise", "scale": 0.02, "random_state": 0, "label": "noise"},
        {
            "type": "unit_rescaling",
            "feature_index": 0,
            "factor": 1.05,
            "label": "lab_a +5%",
        },
    ]
    result = evaluate_reliability(clf, X, perturbation_specs=specs, feature_names=names)

    assert result.details["perturbation_classes"] == {
        "noise": SEMANTICS_PRESERVING,
        "lab_a +5%": COVARIATE_SHIFT,
    }
    assert set(result.details["per_perturbation_flip_rate"]) == {"noise"}
    assert set(result.details["covariate_shift_flip_rate"]) == {"lab_a +5%"}
    assert result.details["n_semantics_preserving"] == 1
    assert result.details["n_covariate_shift"] == 1
    # JSS equals the flip rate of the single semantics-preserving perturbation.
    assert result.judge_sensitivity_score == pytest.approx(
        result.details["per_perturbation_flip_rate"]["noise"]
    )


def test_f6_apply_perturbation_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown perturbation type"):
        apply_perturbation(np.zeros((4, 2)), {"type": "teleport"})


def test_f6_schema_rejects_unknown_type_names():
    with pytest.raises(ValueError, match="Unknown feature type"):
        FeatureSchema(types=["continuous", "quantum"])


# ── F7: minimum rank correlation ─────────────────────────────────────────────
def test_f7_min_rank_correlation_is_exposed(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [
        {"type": "gaussian_noise", "scale": 0.001, "random_state": 1, "label": "tiny"},
        {"type": "gaussian_noise", "scale": 2.0, "random_state": 2, "label": "huge"},
    ]
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
    per = result.details["per_perturbation_rank_correlation"]
    assert result.rank_correlation_min == pytest.approx(min(per.values()))
    assert result.rank_correlation_min <= result.rank_correlation_mean
    assert result.details["min_rank_correlation_perturbation"] == "huge"
    assert result.per_perturbation_rank_correlation == per


def test_f7_mean_can_hide_a_failing_perturbation():
    """The documented R2 is rho >= 0.95 for EVERY perturbation."""
    r = ReliabilityResult(
        rank_correlation_mean=float(np.mean([1.0, 1.0, 1.0, 1.0, 1.0, 0.75])),
        rank_correlation_min=0.75,
    )
    assert r.rank_correlation_mean > 0.95
    assert r.rank_correlation_min < 0.95


def test_f7_policy_evaluates_the_minimum():
    from rised.policy import PolicyThresholds, Verdict, evaluate_policy
    from rised.results import FrameworkReport

    report = FrameworkReport(
        reliability=ReliabilityResult(
            judge_sensitivity_score=0.01,
            rank_correlation_mean=0.958,
            rank_correlation_min=0.75,
        )
    )
    policy = evaluate_policy(
        report, PolicyThresholds(min_rank_correlation=0.95)
    )
    assert policy.dimensions["reliability"].verdict is Verdict.DOES_NOT_MEET


# ── empty perturbation set is not a pass ─────────────────────────────────────
def test_no_perturbations_is_not_evaluated_rather_than_perfect(fitted_lr, small_cohort):
    X, _ = small_cohort
    for specs in (None, []):
        result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
        assert result.judge_sensitivity_score is None
        assert result.perturbation_flip_rate is None
        assert result.rank_correlation_mean is None
        assert result.rank_correlation_min is None
        assert result.details["status"] == "not_evaluated"


def test_only_covariate_shift_specs_is_not_evaluated(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [{"type": "unit_rescaling", "feature_index": 0, "factor": 1.05}]
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
    assert result.judge_sensitivity_score is None
    assert result.details["status"] == "not_evaluated"
    assert result.details["n_covariate_shift"] == 1


# ── general behaviour ────────────────────────────────────────────────────────
def test_gaussian_noise_low_jss(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.001, "random_state": 0}]
    result = evaluate_reliability(fitted_lr, X, perturbation_specs=specs)
    assert result.judge_sensitivity_score < 0.15


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


def test_tau_ref_changes_flip_rates(fitted_lr, small_cohort):
    X, _ = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.5, "random_state": 4}]
    at_half = evaluate_reliability(fitted_lr, X, perturbation_specs=specs, tau_ref=0.5)
    at_third = evaluate_reliability(fitted_lr, X, perturbation_specs=specs, tau_ref=0.3)
    assert at_half.details["reference_threshold"] == 0.5
    assert at_third.details["reference_threshold"] == 0.3
    assert at_half.judge_sensitivity_score != at_third.judge_sensitivity_score


def test_grouped_bootstrap_for_jss(clustered_cohort):
    X, y, groups, demo, clf = clustered_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.2, "random_state": 0}]
    clustered = evaluate_reliability(
        clf, X, perturbation_specs=specs, n_bootstrap=200,
        random_state=1, groups=groups,
    )
    assert clustered.details["resampling"]["clustered"] is True
    assert clustered.jss_ci is not None


def test_reliability_passed_is_withdrawn():
    with pytest.raises(NotImplementedError, match="withdrawn"):
        ReliabilityResult(judge_sensitivity_score=0.0).passed()


# ── metrics ──────────────────────────────────────────────────────────────────
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
