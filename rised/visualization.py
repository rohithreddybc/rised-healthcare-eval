"""
Visualization utilities for RISED framework outputs.

All plot functions return a matplotlib Figure so callers can save or embed
in notebooks without side effects.
"""

from __future__ import annotations

from typing import Optional, Tuple

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


_INCONCLUSIVE_COLOR = "#f9a825"  # amber


def _ci_status(point: Optional[float], ci: Optional[Tuple[float, float]],
               threshold: float, lower_better: bool) -> str:
    """Return PASS / FAIL / INCONCLUSIVE using a CI-based decision rule.

    lower_better=True : pass when value < threshold (e.g., JSS, gap, TFR)
    lower_better=False: pass when value >= threshold (e.g., rho_need)
    Falls back to point estimate if CI not available.
    """
    if point is None:
        return "N/A"
    if ci is None or ci[0] is None or ci[1] is None:
        if lower_better:
            return "PASS" if point < threshold else "FAIL"
        return "PASS" if point >= threshold else "FAIL"
    lo, hi = ci
    if lower_better:
        if hi < threshold:
            return "PASS"
        if lo > threshold:
            return "FAIL"
        return "INCONCLUSIVE"
    if lo >= threshold:
        return "PASS"
    if hi < threshold:
        return "FAIL"
    return "INCONCLUSIVE"


def plot_framework_dashboard(
    report: FrameworkReport,
    title: Optional[str] = None,
) -> plt.Figure:
    """RISED scorecard with point estimate, 95% CI, and CI-based status."""
    dims = ["Reliability", "Inclusivity", "Sensitivity", "Equity", "Deployability"]

    rows = []  # (dim, metric, value, ci_str, status)

    # ── Reliability (JSS < 0.05) ──
    rel = report.reliability
    if rel is None or rel.judge_sensitivity_score is None:
        rows.append(("Reliability", "JSS", "—", "—", "N/A"))
    else:
        ci = rel.jss_ci
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        status = _ci_status(rel.judge_sensitivity_score, ci, 0.05, lower_better=True)
        rows.append(("Reliability", "JSS",
                     f"{rel.judge_sensitivity_score:.4f}", ci_str, status))

    # ── Inclusivity (AUC gap <= 0.05) ──
    inc = report.inclusivity
    if inc is None or inc.auc_parity_gap is None:
        rows.append(("Inclusivity", "AUC parity gap", "—", "—", "N/A"))
    else:
        ci = inc.auc_gap_ci
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
        status = _ci_status(inc.auc_parity_gap, ci, 0.05, lower_better=True)
        rows.append(("Inclusivity", "AUC parity gap",
                     f"{inc.auc_parity_gap:.4f}", ci_str, status))

    # ── Sensitivity (max TFR <= 0.10) ──
    sen = report.sensitivity
    if sen is None or not sen.threshold_flip_rates:
        rows.append(("Sensitivity", "Max TFR (%)", "—", "—", "N/A"))
    else:
        max_tfr = max(sen.threshold_flip_rates.values())
        ci = sen.max_tfr_ci
        ci_str = f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]" if ci else "—"
        status = _ci_status(max_tfr, ci, 0.10, lower_better=True)
        rows.append(("Sensitivity", "Max TFR (%)",
                     f"{max_tfr*100:.1f}%", ci_str, status))

    # ── Equity (rho_need >= 0.70) ──
    eq = report.equity
    if eq is None or eq.need_prediction_correlation is None:
        rows.append(("Equity", "ρ_need", "—", "—", "N/A"))
    else:
        # No CI computed for rho_need; fall back to point estimate
        status = _ci_status(eq.need_prediction_correlation, None, 0.70, lower_better=False)
        rows.append(("Equity", "ρ_need",
                     f"{eq.need_prediction_correlation:.4f}", "—", status))

    # ── Deployability (latency <= 500 ms) ──
    dep = report.deployability
    if dep is None or dep.mean_inference_latency_ms is None:
        rows.append(("Deployability", "Latency (ms)", "—", "—", "N/A"))
    else:
        status = _ci_status(dep.mean_inference_latency_ms, None, 500.0, lower_better=True)
        rows.append(("Deployability", "Latency (ms)",
                     f"{dep.mean_inference_latency_ms:.1f}", "—", status))

    # ── Render compactly to avoid an empty page in the PDF ──
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.axis("off")
    headers = ["Dimension", "Primary metric", "Value", "95% CI", "Status"]
    col_widths = [0.18, 0.22, 0.14, 0.28, 0.18]
    tbl = ax.table(
        cellText=[r for r in rows],
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)

    color_map = {"PASS": _PASS_COLOR, "FAIL": _FAIL_COLOR,
                 "INCONCLUSIVE": _INCONCLUSIVE_COLOR, "N/A": "#bdbdbd"}
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 4:
            text = cell.get_text().get_text()
            cell.set_facecolor(color_map.get(text, "#bdbdbd"))
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")

    ax.set_title(title or "RISED Framework Scorecard (CI-based decisions)",
                 fontsize=12, pad=8, fontweight="bold")
    fig.tight_layout()
    return fig
