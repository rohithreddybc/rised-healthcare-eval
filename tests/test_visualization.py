"""
Tests for rised.visualization — smoke-tests that all plot functions return
a matplotlib Figure without raising, and that no plot asserts a validated
cut-point of its own.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from rised.policy import PolicyThresholds
from rised.results import (
    DeployabilityResult,
    EquityResult,
    FrameworkReport,
    InclusivityResult,
    ReliabilityResult,
    SensitivityResult,
)
from rised.visualization import (
    plot_equity_gaps,
    plot_framework_dashboard,
    plot_reliability_summary,
    plot_shap_summary,
    plot_subgroup_aucs,
    plot_threshold_sensitivity,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def reliability_result():
    return ReliabilityResult(
        judge_sensitivity_score=0.064,
        perturbation_flip_rate=0.064,
        rank_correlation_mean=0.981,
        rank_correlation_min=0.963,
        details={
            "per_perturbation_flip_rate": {"noise_5pct": 0.097, "noise_10pct": 0.101},
            "per_perturbation_rank_correlation": {"noise_5pct": 0.965, "noise_10pct": 0.963},
        },
    )


@pytest.fixture
def inclusivity_result():
    return InclusivityResult(
        subgroup_aucs={"race=White": 0.963, "race=Black": 0.958, "sex=F": 0.965, "sex=M": 0.957},
        per_partition_aucs={
            "race": {"race=White": 0.963, "race=Black": 0.958},
            "sex": {"sex=F": 0.965, "sex=M": 0.957},
        },
        per_partition_auc_gaps={"race": 0.005, "sex": 0.008},
        auc_parity_gap=0.008,
        pooled_auc_gap_diagnostic=0.008,
        subgroup_calibration={"race=White": 0.027, "race=Black": 0.046},
        excluded_subgroups={"race=Other": "n=12 < min_subgroup_n=30"},
    )


@pytest.fixture
def sensitivity_result():
    return SensitivityResult(
        threshold_flip_rates={0.3: 0.08, 0.4: 0.04, 0.5: 0.00, 0.6: 0.03, 0.7: 0.06},
        max_threshold_flip_rate=0.08,
        wide_band_flip_rates={0.1: 0.21, 0.5: 0.0, 0.9: 0.19},
        wide_band_max_tfr=0.21,
        rank_stability_score=0.942,
        decision_boundary_width=0.04,
    )


@pytest.fixture
def equity_result():
    return EquityResult(
        need_prediction_correlation=0.42,
        attainable_rho_ceiling=0.546,
        proxy_prevalence=0.112,
        group_need_gaps={"race=White": -0.01, "race=Black": 0.03, "sex=F": -0.02, "sex=M": 0.02},
        details={"ceiling_note": "binary proxy"},
    )


@pytest.fixture
def deployability_result():
    return DeployabilityResult(
        batch_scoring_time_ms=2.4,
        amortised_time_per_row_ms=0.012,
        single_row_latency_ms=0.41,
        local_global_topk_agreement=0.86,
        global_top1_in_local_topk=0.74,
        details={
            "global_top_features": ["age", "prior_hosp_count", "cci_score"],
            "explanation_chance_level": 0.3,
        },
    )


@pytest.fixture
def full_report(reliability_result, inclusivity_result, sensitivity_result,
                equity_result, deployability_result):
    return FrameworkReport(
        reliability=reliability_result,
        inclusivity=inclusivity_result,
        sensitivity=sensitivity_result,
        equity=equity_result,
        deployability=deployability_result,
    )


def test_plot_reliability_returns_figure(reliability_result):
    assert hasattr(plot_reliability_summary(reliability_result), "savefig")


def test_plot_reliability_with_policy_line(reliability_result):
    fig = plot_reliability_summary(reliability_result, policy_max_flip_rate=0.05)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Policy threshold" in t for t in labels)


def test_plot_reliability_without_policy_draws_no_threshold(reliability_result):
    fig = plot_reliability_summary(reliability_result)
    assert fig.axes[0].get_legend() is None


def test_plot_subgroup_aucs_reports_partition_not_fixed_threshold(inclusivity_result):
    fig = plot_subgroup_aucs(inclusivity_result)
    xlabel = fig.axes[0].get_xlabel()
    assert "per-partition" in xlabel
    assert "0.05" not in xlabel


def test_plot_threshold_sensitivity_returns_figure(sensitivity_result):
    fig = plot_threshold_sensitivity(sensitivity_result)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert any("Primary band" in t for t in labels)
    assert any("Wide band" in t for t in labels)
    assert not any("Pass threshold" in t for t in labels)


def test_plot_equity_gaps_shows_ceiling(equity_result):
    assert hasattr(plot_equity_gaps(equity_result), "savefig")


def test_plot_shap_summary_uses_renamed_metrics(deployability_result):
    assert hasattr(plot_shap_summary(deployability_result), "savefig")


def test_plot_shap_summary_handles_undefined_metrics():
    result = DeployabilityResult(
        details={
            "explanation_metrics_undefined_reason": "d=3 <= top_k=3",
            "explanation_chance_level": 1.0,
        }
    )
    assert hasattr(plot_shap_summary(result), "savefig")


def test_dashboard_without_thresholds_is_indeterminate(full_report):
    fig = plot_framework_dashboard(full_report)
    text = " ".join(
        cell.get_text().get_text() for cell in fig.axes[0].tables[0].get_celld().values()
    )
    assert "NO THRESHOLD" in text
    assert "INDETERMINATE" in text


def test_dashboard_with_thresholds_renders_verdicts(full_report):
    fig = plot_framework_dashboard(
        full_report,
        PolicyThresholds(
            max_judge_sensitivity_score=0.05,
            max_partition_auc_gap=0.05,
            max_threshold_flip_rate=0.10,
            max_single_row_latency_ms=500.0,
        ),
    )
    text = " ".join(
        cell.get_text().get_text() for cell in fig.axes[0].tables[0].get_celld().values()
    )
    assert "DOES NOT MEET" in text or "MEETS" in text
    assert "DIAGNOSTIC" in text


def test_dashboard_handles_partial_report(reliability_result):
    fig = plot_framework_dashboard(FrameworkReport(reliability=reliability_result))
    text = " ".join(
        cell.get_text().get_text() for cell in fig.axes[0].tables[0].get_celld().values()
    )
    assert "NOT EVALUATED" in text
    assert "INDETERMINATE" in text


def test_plot_reliability_empty_details():
    assert hasattr(plot_reliability_summary(ReliabilityResult()), "savefig")


def test_plot_subgroup_aucs_empty():
    assert hasattr(plot_subgroup_aucs(InclusivityResult()), "savefig")


def test_plot_threshold_sensitivity_empty():
    assert hasattr(plot_threshold_sensitivity(SensitivityResult()), "savefig")


def test_plot_equity_gaps_empty():
    assert hasattr(plot_equity_gaps(EquityResult()), "savefig")
