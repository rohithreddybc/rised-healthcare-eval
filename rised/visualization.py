"""
Visualization utilities for RISED framework outputs.

All plot functions return a matplotlib Figure so callers can save or embed
in notebooks without side effects.
"""

from __future__ import annotations

from typing import Optional

from rised.results import (
    DeployabilityResult,
    EquityResult,
    FrameworkReport,
    InclusivityResult,
    ReliabilityResult,
    SensitivityResult,
)


def plot_reliability_summary(result: ReliabilityResult, title: Optional[str] = None):
    """Bar chart of perturbation flip rates per perturbation type."""
    raise NotImplementedError("plot_reliability_summary() will be implemented in Session 5.")


def plot_subgroup_aucs(result: InclusivityResult, title: Optional[str] = None):
    """Horizontal bar chart comparing subgroup AUCs with overall AUC baseline."""
    raise NotImplementedError("plot_subgroup_aucs() will be implemented in Session 5.")


def plot_threshold_sensitivity(result: SensitivityResult, title: Optional[str] = None):
    """Line plot of decision flip rate vs. decision threshold."""
    raise NotImplementedError("plot_threshold_sensitivity() will be implemented in Session 5.")


def plot_equity_gaps(result: EquityResult, title: Optional[str] = None):
    """Bar chart of group-level need–prediction gaps."""
    raise NotImplementedError("plot_equity_gaps() will be implemented in Session 5.")


def plot_shap_summary(result: DeployabilityResult, title: Optional[str] = None):
    """SHAP beeswarm summary plot (wraps shap.summary_plot)."""
    raise NotImplementedError("plot_shap_summary() will be implemented in Session 5.")


def plot_framework_dashboard(report: FrameworkReport, title: Optional[str] = None):
    """Five-panel dashboard summarizing all RISED dimensions."""
    raise NotImplementedError("plot_framework_dashboard() will be implemented in Session 5.")
