"""Tests for the policy layer and the withdrawal of the PASS/FAIL gate."""

import pytest

from rised.policy import (
    ADVISORY_NOTICE,
    ROLLUP_DIMENSIONS,
    PolicyThresholds,
    Verdict,
    evaluate_policy,
)
from rised.results import (
    DeployabilityResult,
    EquityResult,
    FrameworkReport,
    InclusivityResult,
    ReliabilityResult,
    SensitivityResult,
)


def _complete_report(**overrides):
    base = dict(
        reliability=ReliabilityResult(
            judge_sensitivity_score=0.02, rank_correlation_min=0.99
        ),
        inclusivity=InclusivityResult(
            auc_parity_gap=0.03,
            per_partition_auc_gaps={"race": 0.03, "sex": 0.01},
            subgroup_calibration={"race=A": 0.05},
        ),
        sensitivity=SensitivityResult(
            threshold_flip_rates={0.3: 0.04, 0.5: 0.0, 0.7: 0.05},
            max_threshold_flip_rate=0.05,
        ),
        equity=EquityResult(need_prediction_correlation=0.4),
        deployability=DeployabilityResult(
            batch_scoring_time_ms=20.0, single_row_latency_ms=0.9
        ),
    )
    base.update(overrides)
    return FrameworkReport(**base)


def _permissive_thresholds():
    return PolicyThresholds(
        max_judge_sensitivity_score=0.05,
        min_rank_correlation=0.95,
        max_partition_auc_gap=0.05,
        max_subgroup_ece=0.10,
        max_threshold_flip_rate=0.10,
        max_single_row_latency_ms=500.0,
    )


# ── the gate is withdrawn ────────────────────────────────────────────────────
def test_gate_methods_raise_rather_than_return_a_boolean():
    report = _complete_report()
    for method in ("all_passed", "summary", "diagnostic_status"):
        with pytest.raises(NotImplementedError, match="withdrawn"):
            getattr(report, method)()


def test_gate_error_points_at_the_replacement():
    with pytest.raises(NotImplementedError, match="rised.policy.evaluate_policy"):
        FrameworkReport().all_passed()


def test_advisory_notice_is_attached_and_says_it_is_not_clearance():
    policy = evaluate_policy(_complete_report(), _permissive_thresholds())
    assert policy.notice == ADVISORY_NOTICE
    assert "not" in ADVISORY_NOTICE and "clearance" in ADVISORY_NOTICE
    assert "ADVISORY" in policy.explain()


# ── F10: partial reports are never positive ──────────────────────────────────
def test_f10_report_with_only_reliability_is_indeterminate():
    """The exact defect: a report containing only Reliability=PASS cleared it."""
    report = FrameworkReport(
        reliability=ReliabilityResult(
            judge_sensitivity_score=0.01, rank_correlation_min=1.0
        )
    )
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions["reliability"].verdict is Verdict.MEETS
    assert policy.overall_verdict() is Verdict.INDETERMINATE


@pytest.mark.parametrize("missing", ROLLUP_DIMENSIONS)
def test_f10_any_missing_rolled_up_dimension_is_indeterminate(missing):
    report = _complete_report(**{missing: None})
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions[missing].verdict is Verdict.NOT_EVALUATED
    assert policy.overall_verdict() is Verdict.INDETERMINATE


def test_f10_empty_report_is_indeterminate_not_false():
    policy = evaluate_policy(FrameworkReport(), _permissive_thresholds())
    assert policy.overall_verdict() is Verdict.INDETERMINATE
    for dim in ROLLUP_DIMENSIONS:
        assert policy.dimensions[dim].verdict is Verdict.NOT_EVALUATED


def test_f10_complete_report_can_reach_a_positive_verdict():
    policy = evaluate_policy(_complete_report(), _permissive_thresholds())
    assert policy.overall_verdict() is Verdict.MEETS


def test_f10_one_failing_dimension_fails_the_rollup():
    report = _complete_report(
        sensitivity=SensitivityResult(
            threshold_flip_rates={0.3: 0.4}, max_threshold_flip_rate=0.4
        )
    )
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions["sensitivity"].verdict is Verdict.DOES_NOT_MEET
    assert policy.overall_verdict() is Verdict.DOES_NOT_MEET


def test_missing_dimension_helpers():
    report = _complete_report(equity=None)
    assert report.missing_dimensions() == ["equity"]
    assert "reliability" in report.evaluated_dimensions()
    assert report.is_complete() is False
    assert _complete_report().is_complete() is True


# ── no thresholds means no verdict ───────────────────────────────────────────
def test_unconfigured_thresholds_do_not_default_to_passing():
    policy = evaluate_policy(_complete_report())
    for dim in ROLLUP_DIMENSIONS:
        assert policy.dimensions[dim].verdict is Verdict.NOT_CONFIGURED
    assert policy.overall_verdict() is Verdict.INDETERMINATE


def test_partially_configured_dimension_uses_only_configured_criteria():
    policy = evaluate_policy(
        _complete_report(),
        PolicyThresholds(
            max_judge_sensitivity_score=0.05,
            max_partition_auc_gap=0.05,
            max_threshold_flip_rate=0.10,
            max_single_row_latency_ms=500.0,
        ),
    )
    assert policy.dimensions["reliability"].verdict is Verdict.MEETS
    assert policy.overall_verdict() is Verdict.MEETS


def test_configured_threshold_with_missing_measurement_is_indeterminate():
    report = _complete_report(
        reliability=ReliabilityResult(judge_sensitivity_score=None)
    )
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions["reliability"].verdict is Verdict.INDETERMINATE
    assert policy.overall_verdict() is Verdict.INDETERMINATE


# ── CI-aware verdicts ────────────────────────────────────────────────────────
def test_ci_straddling_the_threshold_is_indeterminate():
    report = _complete_report(
        inclusivity=InclusivityResult(
            auc_parity_gap=0.04,
            auc_gap_ci=(0.01, 0.09),
            subgroup_calibration={"race=A": 0.02},
        )
    )
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions["inclusivity"].verdict is Verdict.INDETERMINATE
    assert policy.overall_verdict() is Verdict.INDETERMINATE


def test_ci_entirely_below_the_threshold_meets():
    report = _complete_report(
        inclusivity=InclusivityResult(
            auc_parity_gap=0.02,
            auc_gap_ci=(0.01, 0.03),
            subgroup_calibration={"race=A": 0.02},
        )
    )
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions["inclusivity"].verdict is Verdict.MEETS


def test_ci_entirely_above_the_threshold_does_not_meet():
    report = _complete_report(
        inclusivity=InclusivityResult(
            auc_parity_gap=0.20,
            auc_gap_ci=(0.15, 0.25),
            subgroup_calibration={"race=A": 0.02},
        )
    )
    policy = evaluate_policy(report, _permissive_thresholds())
    assert policy.dimensions["inclusivity"].verdict is Verdict.DOES_NOT_MEET


def test_ci_can_be_disabled():
    report = _complete_report(
        inclusivity=InclusivityResult(
            auc_parity_gap=0.04,
            auc_gap_ci=(0.01, 0.09),
            subgroup_calibration={"race=A": 0.02},
        )
    )
    thresholds = _permissive_thresholds()
    thresholds.use_confidence_intervals = False
    policy = evaluate_policy(report, thresholds)
    assert policy.dimensions["inclusivity"].verdict is Verdict.MEETS


# ── reporting surface ────────────────────────────────────────────────────────
def test_verdict_table_includes_overall():
    policy = evaluate_policy(_complete_report(), _permissive_thresholds())
    table = policy.verdict_table()
    assert table["overall"] == Verdict.MEETS.value
    assert table["equity"] == Verdict.DIAGNOSTIC.value


def test_explain_renders_every_dimension():
    text = evaluate_policy(_complete_report(), _permissive_thresholds()).explain()
    for dim in FrameworkReport.DIMENSIONS:
        assert dim in text
    assert "OVERALL (advisory)" in text


def test_measurement_summary_carries_no_verdicts():
    summary = _complete_report().measurement_summary()
    assert set(summary) == set(FrameworkReport.DIMENSIONS)
    assert summary["inclusivity"]["max_partition_auc_gap"] == 0.03
    assert summary["deployability"]["single_row_latency_ms"] == 0.9
    flat = str(summary)
    assert "PASS" not in flat and "passed" not in flat
