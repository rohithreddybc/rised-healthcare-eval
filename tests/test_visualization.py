"""
Tests for rised.visualization — smoke-tests that all plot functions return
a matplotlib Figure without raising.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

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
        details={
            "per_perturbation_flip_rate": {"noise_5pct": 0.097, "noise_10pct": 0.101},
            "per_perturbation_rank_correlation": {"noise_5pct": 0.965, "noise_10pct": 0.963},
        },
    )


@pytest.fixture
def inclusivity_result():
    return InclusivityResult(
        subgroup_aucs={"race=White": 0.963, "race=Black": 0.958, "sex=F": 0.965, "sex=M": 0.957},
        auc_parity_gap=0.008,
        subgroup_calibration={"race=White": 0.027, "race=Black": 0.046},
        details={"small_group_flags": []},
    )


@pytest.fixture
def sensitivity_result():
    return SensitivityResult(
        threshold_flip_rates={0.3: 0.08, 0.4: 0.04, 0.5: 0.00, 0.6: 0.03, 0.7: 0.06},
        rank_stability_score=0.942,
        decision_boundary_width=0.04,
    )


@pytest.fixture
def equity_result():
    return EquityResult(
        need_prediction_correlation=0.73,
        group_need_gaps={"race=White": -0.01, "race=Black": 0.03, "sex=F": -0.02, "sex=M": 0.02},
    )


@pytest.fixture
def deployability_result():
    return DeployabilityResult(
        mean_inference_latency_ms=2.4,
        explanation_faithfulness=0.86,
        top_feature_stability=0.74,
        details={"global_top_features": ["age", "prior_hosp_count", "cci_score"]},
    )


def test_plot_reliability_returns_figure(reliability_result):
    fig = plot_reliability_summary(reliability_result)
    assert hasattr(fig, "savefig")


def test_plot_subgroup_aucs_returns_figure(inclusivity_result):
    fig = plot_subgroup_aucs(inclusivity_result)
    assert hasattr(fig, "savefig")


def test_plot_threshold_sensitivity_returns_figure(sensitivity_result):
    fig = plot_threshold_sensitivity(sensitivity_result)
    assert hasattr(fig, "savefig")


def test_plot_equity_gaps_returns_figure(equity_result):
    fig = plot_equity_gaps(equity_result)
    assert hasattr(fig, "savefig")


def test_plot_shap_summary_returns_figure(deployability_result):
    fig = plot_shap_summary(deployability_result)
    assert hasattr(fig, "savefig")


def test_plot_framework_dashboard_returns_figure(
    reliability_result, inclusivity_result, sensitivity_result,
    equity_result, deployability_result
):
    report = FrameworkReport(
        reliability=reliability_result,
        inclusivity=inclusivity_result,
        sensitivity=sensitivity_result,
        equity=equity_result,
        deployability=deployability_result,
    )
    fig = plot_framework_dashboard(report)
    assert hasattr(fig, "savefig")


def test_plot_reliability_empty_details():
    result = ReliabilityResult(perturbation_flip_rate=0.0)
    fig = plot_reliability_summary(result)
    assert hasattr(fig, "savefig")


def test_plot_subgroup_aucs_empty():
    result = InclusivityResult()
    fig = plot_subgroup_aucs(result)
    assert hasattr(fig, "savefig")


def test_plot_equity_gaps_empty():
    result = EquityResult()
    fig = plot_equity_gaps(result)
    assert hasattr(fig, "savefig")
