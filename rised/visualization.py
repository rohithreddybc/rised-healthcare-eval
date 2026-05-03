"""
Visualization utilities for RISED framework outputs.

All plot functions return a matplotlib Figure so callers can save or embed
in notebooks without side effects.
"""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rised.results import (
    DeployabilityResult,
    EquityResult,
    FrameworkReport,
    InclusivityResult,
    ReliabilityResult,
    SensitivityResult,
)

_PASS_COLOR = "#2e7d32"   # dark green
_FAIL_COLOR = "#c62828"   # dark red
_NEUTRAL    = "#1565c0"   # dark blue for bars


def plot_reliability_summary(
    result: ReliabilityResult,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of perturbation flip rates per perturbation type."""
    flips = result.details.get("per_perturbation_flip_rate", {})
    if not flips:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No perturbation data", ha="center", va="center")
        return fig

    labels = list(flips.keys())
    values = [flips[k] * 100 for k in labels]
    threshold_pct = 5.0

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = [_FAIL_COLOR if v >= threshold_pct else _PASS_COLOR for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    ax.axvline(threshold_pct, color="black", linestyle="--", linewidth=1.2,
               label=f"Pass threshold ({threshold_pct:.0f}%)")
    ax.set_xlabel("Decision Flip Rate (%)", fontsize=11)
    ax.set_title(title or "Reliability: Decision Flip Rate by Perturbation", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(values) * 1.25 + 2)
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_subgroup_aucs(
    result: InclusivityResult,
    title: Optional[str] = None,
) -> plt.Figure:
    """Horizontal bar chart comparing subgroup AUCs with AUC parity gap annotation."""
    aucs = result.subgroup_aucs
    if not aucs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No subgroup AUC data", ha="center", va="center")
        return fig

    labels = sorted(aucs.keys())
    values = [aucs[k] for k in labels]
    overall_mean = float(np.mean(values))

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    colors = [_NEUTRAL] * len(values)
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.65)
    ax.axvline(overall_mean, color="black", linestyle="--", linewidth=1.2,
               label=f"Mean AUC ({overall_mean:.3f})")
    ax.set_xlabel("AUC-ROC", fontsize=11)
    ax.set_title(title or "Inclusivity: Subgroup AUC-ROC", fontsize=12)
    ax.legend(fontsize=9)
    x_min = min(values) - 0.02
    ax.set_xlim(x_min, 1.02)
    for bar, val in zip(bars, values):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    if result.auc_parity_gap is not None:
        ax.set_xlabel(
            f"AUC-ROC  (parity gap = {result.auc_parity_gap:.3f}, "
            f"threshold = 0.05)", fontsize=11
        )
    fig.tight_layout()
    return fig


def plot_threshold_sensitivity(
    result: SensitivityResult,
    title: Optional[str] = None,
) -> plt.Figure:
    """Line plot of decision flip rate vs. decision threshold."""
    tfr = result.threshold_flip_rates
    if not tfr:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No threshold flip rate data", ha="center", va="center")
        return fig

    thresholds = sorted(tfr.keys())
    rates = [tfr[t] * 100 for t in thresholds]
    threshold_pct = 10.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, rates, marker="o", color=_NEUTRAL, linewidth=2, markersize=5)
    ax.axhline(threshold_pct, color="black", linestyle="--", linewidth=1.2,
               label=f"Pass threshold ({threshold_pct:.0f}%)")
    ax.fill_between(thresholds, rates, threshold_pct,
                    where=[r > threshold_pct for r in rates],
                    alpha=0.15, color=_FAIL_COLOR, label="Exceeds threshold")
    ax.set_xlabel("Decision Threshold τ", fontsize=11)
    ax.set_ylabel("Threshold Flip Rate (%)", fontsize=11)
    ax.set_title(title or "Sensitivity: Threshold Flip Rate Sweep", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(min(thresholds) - 0.02, max(thresholds) + 0.02)
    ax.set_ylim(0, max(rates) * 1.15 + 2)
    if result.rank_stability_score is not None:
        ax.text(0.98, 0.96, f"Rank stability: {result.rank_stability_score:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    return fig


def plot_equity_gaps(
    result: EquityResult,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of group-level need–prediction gaps."""
    gaps = result.group_need_gaps
    if not gaps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No equity gap data", ha="center", va="center")
        return fig

    labels = sorted(gaps.keys())
    values = [gaps[k] for k in labels]
    flag_thresh = 0.10

    colors = [_FAIL_COLOR if abs(v) > flag_thresh else _PASS_COLOR for v in values]

    fig, ax = plt.subplots(figsize=(9, max(4, len(labels) * 0.45)))
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.axvline(flag_thresh, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    ax.axvline(-flag_thresh, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
               label=f"Flag threshold (±{flag_thresh})")
    ax.set_xlabel("Need–Prediction Gap (score − need)", fontsize=11)
    ax.set_title(title or "Equity: Group-Level Need–Prediction Gaps", fontsize=12)
    ax.legend(fontsize=9)
    if result.need_prediction_correlation is not None:
        ax.text(0.98, 0.04, f"ρ_need = {result.need_prediction_correlation:.3f}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    return fig


def plot_shap_summary(
    result: DeployabilityResult,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of global top SHAP features and faithfulness metrics."""
    top_features = result.details.get("global_top_features", [])
    faithfulness = result.explanation_faithfulness
    stability = result.top_feature_stability

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if top_features:
        n = len(top_features)
        # Rank bars (importance implied by rank order)
        y_vals = list(range(n, 0, -1))
        ax.barh(y_vals, [n - i for i in range(n)], color=_NEUTRAL,
                edgecolor="white", height=0.6)
        ax.set_yticks(y_vals)
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel("Global Rank Score (higher = more important)", fontsize=10)
    ax.set_title(title or "Deployability: Global SHAP Feature Importance (Top Features)", fontsize=12)
    info_text = []
    if faithfulness is not None:
        info_text.append(f"Explanation faithfulness: {faithfulness:.2f}")
    if stability is not None:
        info_text.append(f"Top feature stability: {stability:.2f}")
    if info_text:
        ax.text(0.98, 0.05, "\n".join(info_text), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    return fig


def plot_framework_dashboard(
    report: FrameworkReport,
    title: Optional[str] = None,
) -> plt.Figure:
    """Five-panel dashboard summarizing all RISED dimensions as a scorecard."""
    dims = ["Reliability", "Inclusivity", "Sensitivity", "Equity", "Deployability"]
    results_objs = [
        report.reliability,
        report.inclusivity,
        report.sensitivity,
        report.equity,
        report.deployability,
    ]

    # Build summary rows: [dimension, primary metric label, value, passed]
    rows = []
    for dim, res in zip(dims, results_objs):
        if res is None:
            rows.append((dim, "—", "N/A", None))
            continue
        if dim == "Reliability":
            label = "JSS"
            val = f"{res.judge_sensitivity_score:.4f}" if res.judge_sensitivity_score is not None else "—"
        elif dim == "Inclusivity":
            label = "AUC parity gap"
            val = f"{res.auc_parity_gap:.4f}" if res.auc_parity_gap is not None else "—"
        elif dim == "Sensitivity":
            label = "Rank stability"
            val = f"{res.rank_stability_score:.4f}" if res.rank_stability_score is not None else "—"
        elif dim == "Equity":
            label = "ρ_need"
            val = f"{res.need_prediction_correlation:.4f}" if res.need_prediction_correlation is not None else "—"
        else:  # Deployability
            label = "Latency (ms)"
            val = f"{res.mean_inference_latency_ms:.1f}" if res.mean_inference_latency_ms is not None else "—"
        rows.append((dim, label, val, res.passed()))

    fig, ax = plt.subplots(figsize=(8, len(dims) * 0.8 + 1.2))
    ax.axis("off")
    table_data = [["Dimension", "Primary metric", "Value", "Status"]]
    for dim, label, val, passed in rows:
        status = ("PASS" if passed else "FAIL") if passed is not None else "N/A"
        table_data.append([dim, label, val, status])

    col_widths = [0.28, 0.32, 0.20, 0.20]
    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 3 and row > 0:
            text = cell.get_text().get_text()
            cell.set_facecolor(_PASS_COLOR if text == "PASS" else
                               (_FAIL_COLOR if text == "FAIL" else "#bdbdbd"))
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")

    ax.set_title(title or "RISED Framework Scorecard", fontsize=14, pad=20, fontweight="bold")
    fig.tight_layout()
    return fig
