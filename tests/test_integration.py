"""
Integration tests for the RISED evaluate_all() entry point.
"""

import numpy as np
import pytest

import rised
from rised import evaluate_all
from rised.policy import PolicyThresholds, Verdict, evaluate_policy
from rised.results import FrameworkReport


def test_evaluate_all_returns_framework_report(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert isinstance(report, FrameworkReport)


def test_evaluate_all_populates_every_dimension_with_a_need_proxy(
    fitted_lr, small_cohort, demographic_with_need
):
    X, y = small_cohort
    report = evaluate_all(
        fitted_lr, X, y, demographic_with_need, need_column="comorbidity"
    )
    assert report.is_complete()
    assert report.missing_dimensions() == []


def test_evaluate_all_skips_equity_without_a_need_proxy(
    fitted_lr, small_cohort, demographic_df
):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert report.equity is None
    assert report.missing_dimensions() == ["equity"]
    assert report.is_complete() is False


def test_measurement_summary_has_five_keys(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert set(report.measurement_summary()) == {
        "reliability", "inclusivity", "sensitivity", "equity", "deployability"
    }


def test_evaluate_all_metadata_contains_shape(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert report.metadata["n_samples"] == 200
    assert report.metadata["n_features"] == 10
    assert report.metadata["tau_ref"] == 0.5
    assert report.metadata["clustered_resampling"] is False
    assert report.metadata["rised_version"] == rised.__version__


def test_evaluate_all_with_perturbation_specs(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.01, "random_state": 0}]
    report = evaluate_all(fitted_lr, X, y, demographic_df, perturbation_specs=specs)
    assert report.reliability.judge_sensitivity_score is not None
    assert report.reliability.rank_correlation_min is not None


def test_evaluate_all_reproducible_with_random_state(
    fitted_lr, small_cohort, demographic_df
):
    """evaluate_all() with same random_state must produce identical CI values."""
    X, y = small_cohort
    specs = [{"type": "gaussian_noise", "scale": 0.05, "random_state": 0, "label": "n5"}]
    kwargs = dict(
        perturbation_specs=specs, random_state=42, n_bootstrap=200,
    )
    r1 = evaluate_all(fitted_lr, X, y, demographic_df, **kwargs)
    r2 = evaluate_all(fitted_lr, X, y, demographic_df, **kwargs)
    assert r1.reliability.judge_sensitivity_score == r2.reliability.judge_sensitivity_score
    assert r1.reliability.jss_ci == r2.reliability.jss_ci
    assert r1.sensitivity.max_tfr_ci == r2.sensitivity.max_tfr_ci
    assert r1.inclusivity.auc_gap_ci == r2.inclusivity.auc_gap_ci


def test_evaluate_all_threads_groups_to_every_dimension(clustered_cohort):
    X, y, groups, demo, clf = clustered_cohort
    report = evaluate_all(
        clf, X, y, demo, n_bootstrap=200, random_state=0, groups=groups,
        perturbation_specs=[{"type": "gaussian_noise", "scale": 0.1, "random_state": 0}],
    )
    assert report.metadata["clustered_resampling"] is True
    n_units = len(np.unique(groups))
    assert report.inclusivity.details["resampling"]["n_units"] == n_units
    assert report.sensitivity.details["resampling"]["n_units"] == n_units
    assert report.reliability.details["resampling"]["n_units"] == n_units


def test_evaluate_all_threads_min_subgroup_n(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df, min_subgroup_n=60)
    assert report.inclusivity.details["min_subgroup_n"] == 60
    assert report.inclusivity.excluded_subgroups


def test_evaluate_all_narrow_band_is_primary(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert report.sensitivity.details["threshold_band"] == (0.30, 0.70)
    assert report.sensitivity.wide_band_max_tfr is not None


def test_full_pipeline_measurement_then_policy(
    fitted_lr, small_cohort, demographic_with_need
):
    """The documented two-step workflow."""
    X, y = small_cohort
    report = evaluate_all(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity",
        perturbation_specs=[
            {"type": "gaussian_noise", "scale": 0.01, "random_state": 0}
        ],
    )
    policy = evaluate_policy(
        report,
        PolicyThresholds(
            max_judge_sensitivity_score=0.05,
            min_rank_correlation=0.95,
            max_partition_auc_gap=0.50,
            max_threshold_flip_rate=1.0,
            max_single_row_latency_ms=5000.0,
        ),
    )
    assert policy.overall_verdict() in (
        Verdict.MEETS, Verdict.DOES_NOT_MEET, Verdict.INDETERMINATE
    )
    assert policy.dimensions["equity"].verdict is Verdict.DIAGNOSTIC
    assert "ADVISORY" in policy.explain()


def test_public_api_exports_both_layers():
    for name in (
        "evaluate_all", "FrameworkReport", "PolicyThresholds", "PolicyReport",
        "Verdict", "evaluate_policy", "ADVISORY_NOTICE",
    ):
        assert hasattr(rised, name), name
        assert name in rised.__all__


def test_package_version():
    assert rised.__version__ == "0.2.0"
