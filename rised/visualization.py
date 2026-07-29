"""
Visualization utilities for RISED framework outputs.

All plot functions return a matplotlib Figure so callers can save or embed
in notebooks without side effects.

Plots draw the *measurement* layer. Where a reference line is drawn it comes
from a caller-supplied policy threshold, is labelled "policy", and is omitted
entirely when no threshold is given — no plot asserts a validated cut-point.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from rised.policy import PolicyThresholds, Verdict, evaluate_policy
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
    policy_max_flip_rate: Optional[float] = None,
) -> plt.Figure:
    """Bar chart of perturbation flip rates per semantics-preserving perturbation.

    ``policy_max_flip_rate`` draws an advisory reference line when supplied.
    """
    flips = result.details.get("per_perturbation_flip_rate", {})
    if not flips:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.text(0.5, 0.5, "No perturbation data", ha="center", va="center")
        return fig

    labels = list(flips.keys())
    values = [flips[k] * 100 for k in labels]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    if policy_max_flip_rate is None:
        colors = [_NEUTRAL] * len(values)
    else:
        thr_pct = policy_max_flip_rate * 100.0
        colors = [_FAIL_COLOR if v > thr_pct else _PASS_COLOR for v in values]
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.55)
    if policy_max_flip_rate is not None:
        ax.axvline(policy_max_flip_rate * 100.0, color="black", linestyle="--",
                   linewidth=1.2,
                   label=f"Policy threshold ({policy_max_flip_rate * 100:.0f}%)")
        ax.legend(fontsize=9)
    ax.set_xlabel("Decision Flip Rate (%)", fontsize=11)
    ax.set_title(title or "Reliability: Decision Flip Rate by Perturbation", fontsize=12)
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
        worst = result.worst_partition
        detail = f" in partition '{worst}'" if worst else ""
        ax.set_xlabel(
            f"AUC-ROC  (max per-partition parity gap = "
            f"{result.auc_parity_gap:.3f}{detail})",
            fontsize=11,
        )
    if result.excluded_subgroups:
        ax.text(0.02, 0.02,
                f"{len(result.excluded_subgroups)} subgroup(s) excluded from the "
                "estimand",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                color="#616161")
    fig.tight_layout()
    return fig


def plot_threshold_sensitivity(
    result: SensitivityResult,
    title: Optional[str] = None,
    policy_max_tfr: Optional[float] = None,
    show_wide_band: bool = True,
) -> plt.Figure:
    """Line plot of threshold flip rate vs. decision threshold.

    The primary (narrow) band is drawn solid; the wide band, when present, is
    drawn faint for context. TFR is a functional of the score CDF alone and
    never reads ``y_true`` — it must be read alongside a discrimination metric.
    """
    tfr = result.threshold_flip_rates
    if not tfr:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No threshold flip rate data", ha="center", va="center")
        return fig

    thresholds = sorted(tfr.keys())
    rates = [tfr[t] * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(7, 4))
    if show_wide_band and result.wide_band_flip_rates:
        wide_t = sorted(result.wide_band_flip_rates.keys())
        wide_r = [result.wide_band_flip_rates[t] * 100 for t in wide_t]
        ax.plot(wide_t, wide_r, marker=".", color="#90a4ae", linewidth=1,
                markersize=4, label="Wide band (secondary)")
        all_rates = rates + wide_r
        all_t = thresholds + wide_t
    else:
        all_rates, all_t = rates, thresholds
    ax.plot(thresholds, rates, marker="o", color=_NEUTRAL, linewidth=2,
            markersize=5, label="Primary band")
    if policy_max_tfr is not None:
        ax.axhline(policy_max_tfr * 100.0, color="black", linestyle="--",
                   linewidth=1.2,
                   label=f"Policy threshold ({policy_max_tfr * 100:.0f}%)")
    ax.set_xlabel("Decision Threshold τ", fontsize=11)
    ax.set_ylabel("Threshold Flip Rate (%)", fontsize=11)
    ax.set_title(title or "Sensitivity: Threshold Flip Rate Sweep", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_xlim(min(all_t) - 0.02, max(all_t) + 0.02)
    ax.set_ylim(0, max(all_rates) * 1.15 + 2)
    if result.rank_stability_score is not None:
        ax.text(0.98, 0.96,
                f"Rank stability: {result.rank_stability_score:.3f}\n"
                "(TFR does not use y_true)",
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
        txt = f"ρ_need = {result.need_prediction_correlation:.3f}"
        if result.attainable_rho_ceiling is not None:
            txt += f"\nceiling √(3p(1−p)) = {result.attainable_rho_ceiling:.3f}"
        ax.text(0.98, 0.04, txt,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    return fig


def plot_shap_summary(
    result: DeployabilityResult,
    title: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of global top SHAP features and explanation-agreement metrics."""
    top_features = result.details.get("global_top_features", [])
    faithfulness = result.local_global_topk_agreement
    stability = result.global_top1_in_local_topk
    chance = result.details.get("explanation_chance_level")
    undefined = result.details.get("explanation_metrics_undefined_reason")

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
        info_text.append(f"Local top-1 ∈ global top-k: {faithfulness:.2f}")
    if stability is not None:
        info_text.append(f"Global top-1 ∈ local top-k: {stability:.2f}")
    if info_text and chance is not None:
        info_text.append(f"chance level k/d = {chance:.2f}")
    if undefined:
        info_text.append("agreement metrics undefined (d ≤ k)")
    if info_text:
        ax.text(0.98, 0.05, "\n".join(info_text), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.3"))
    fig.tight_layout()
    return fig


_INDETERMINATE_COLOR = "#f9a825"  # amber
_DIAGNOSTIC_COLOR = "#546e7a"     # slate
_UNSET_COLOR = "#bdbdbd"          # grey

_VERDICT_COLORS = {
    Verdict.MEETS.value: _PASS_COLOR,
    Verdict.DOES_NOT_MEET.value: _FAIL_COLOR,
    Verdict.INDETERMINATE.value: _INDETERMINATE_COLOR,
    Verdict.DIAGNOSTIC.value: _DIAGNOSTIC_COLOR,
    Verdict.NOT_CONFIGURED.value: _UNSET_COLOR,
    Verdict.NOT_EVALUATED.value: _UNSET_COLOR,
}

_SHORT_VERDICT = {
    Verdict.MEETS.value: "MEETS",
    Verdict.DOES_NOT_MEET.value: "DOES NOT MEET",
    Verdict.INDETERMINATE.value: "INDETERMINATE",
    Verdict.DIAGNOSTIC.value: "DIAGNOSTIC",
    Verdict.NOT_CONFIGURED.value: "NO THRESHOLD",
    Verdict.NOT_EVALUATED.value: "NOT EVALUATED",
}

#: Headline measurement rendered per dimension in the scorecard.
_HEADLINE = {
    "reliability": ("JSS", lambda r: r.reliability.judge_sensitivity_score,
                    lambda r: r.reliability.jss_ci),
    "inclusivity": ("Max per-partition AUC gap",
                    lambda r: r.inclusivity.auc_parity_gap,
                    lambda r: r.inclusivity.auc_gap_ci),
    "sensitivity": ("Max TFR (primary band)",
                    lambda r: r.sensitivity.max_threshold_flip_rate,
                    lambda r: r.sensitivity.max_tfr_ci),
    "equity": ("ρ_need", lambda r: r.equity.need_prediction_correlation,
               lambda r: None),
    "deployability": ("Single-row latency (ms)",
                      lambda r: r.deployability.single_row_latency_ms,
                      lambda r: None),
}


def plot_framework_dashboard(
    report: FrameworkReport,
    thresholds: Optional[PolicyThresholds] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """RISED scorecard: measurement, 95% CI, and advisory policy verdict.

    The verdict column comes from :func:`rised.policy.evaluate_policy` under the
    caller's ``thresholds``. With no thresholds supplied every rolled-up
    dimension reads ``NO THRESHOLD`` and the overall verdict is
    ``INDETERMINATE`` — the framework does not supply cut-points of its own.
    """
    policy = evaluate_policy(report, thresholds)

    rows = []  # (dim, metric, value, ci_str, verdict)
    for dim in FrameworkReport.DIMENSIONS:
        label, get_value, get_ci = _HEADLINE[dim]
        verdict = policy.dimensions[dim].verdict.value
        if getattr(report, dim) is None:
            rows.append((dim.capitalize(), label, "—", "—", verdict))
            continue
        value = get_value(report)
        ci = get_ci(report)
        value_str = "—" if value is None else f"{value:.4f}"
        ci_str = (
            f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            if ci is not None and ci[0] is not None
            else "—"
        )
        rows.append((dim.capitalize(), label, value_str, ci_str, verdict))

    overall = policy.overall_verdict().value
    rows.append(("OVERALL", "advisory roll-up", "—", "—", overall))

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    headers = ["Dimension", "Primary measurement", "Value", "95% CI", "Policy verdict"]
    col_widths = [0.16, 0.30, 0.13, 0.24, 0.22]
    tbl = ax.table(
        cellText=[
            (a, b, c, d, _SHORT_VERDICT.get(e, e)) for a, b, c, d, e in rows
        ],
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#37474f")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 4:
            raw = rows[row - 1][4]
            cell.set_facecolor(_VERDICT_COLORS.get(raw, _UNSET_COLOR))
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "white")

    ax.set_title(title or "RISED Scorecard — measurements with advisory policy verdicts",
                 fontsize=12, pad=8, fontweight="bold")
    fig.text(0.5, 0.01,
             "Advisory only: thresholds are institutional configuration, not "
             "validated against deployment outcomes.",
             ha="center", fontsize=8, color="#616161")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
