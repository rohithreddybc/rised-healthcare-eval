"""
Result dataclasses for each RISED dimension and the combined FrameworkReport.

These objects are the **measurement layer**: metrics and their confidence
intervals, with no thresholds and no verdicts. Nothing here decides whether a
model may be deployed.

The previous ``passed()`` / ``all_passed()`` deployment gate has been withdrawn.
It could not be validated: no observed deployment outcomes exist against which
the cut-points (JSS < 0.05, gap <= 0.05, TFR <= 0.10, rho >= 0.70, latency
<= 500 ms) were ever calibrated, so "PASS" asserted a clearance the evidence did
not support. Thresholds now live in :mod:`rised.policy`, are supplied by the
caller, and yield advisory verdicts explicitly labelled as institutional
configuration.

Calling the withdrawn methods raises :class:`NotImplementedError` with a pointer
to the replacement, rather than silently returning a boolean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_GATE_WITHDRAWN = (
    "The RISED PASS/FAIL deployment gate has been withdrawn: its cut-points "
    "were never calibrated against observed deployment outcomes, so a boolean "
    "'passed' asserted a clearance the evidence does not support. Use "
    "rised.policy.evaluate_policy(report, PolicyThresholds(...)) to apply your "
    "own institutional thresholds and obtain an advisory, clearly-labelled "
    "verdict (including INDETERMINATE when a dimension is missing or its "
    "confidence interval straddles the threshold)."
)


def _gate_withdrawn(*_args, **_kwargs):
    raise NotImplementedError(_GATE_WITHDRAWN)


@dataclass
class ReliabilityResult:
    """Outputs from the Reliability dimension evaluation (measurement only).

    ``rank_correlation_min`` is the statistic the documented R2 criterion
    actually refers to (rho >= 0.95 for *every* perturbation). The mean cannot
    express that criterion and is retained for continuity only.
    """

    judge_sensitivity_score: Optional[float] = None
    perturbation_flip_rate: Optional[float] = None
    rank_correlation_mean: Optional[float] = None
    rank_correlation_min: Optional[float] = None
    jss_ci: Optional[Tuple[float, float]] = None
    flip_rate_ci: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def per_perturbation_rank_correlation(self) -> Dict[str, float]:
        """Rank correlation for each semantics-preserving perturbation."""
        return dict(self.details.get("per_perturbation_rank_correlation", {}))

    passed = _gate_withdrawn


@dataclass
class InclusivityResult:
    """Outputs from the Inclusivity dimension evaluation (measurement only).

    ``auc_parity_gap`` is the maximum **per-partition** gap: the AUC range is
    taken within each demographic column and then maximised over columns.
    ``pooled_auc_gap_diagnostic`` is the old single ``max - min`` over every
    level of every column pooled together; it compares levels from different
    partitions of the cohort and is retained as a diagnostic only.
    """

    subgroup_aucs: Dict[str, float] = field(default_factory=dict)
    per_partition_aucs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_partition_auc_gaps: Dict[str, float] = field(default_factory=dict)
    auc_parity_gap: Optional[float] = None
    auc_gap_ci: Optional[Tuple[float, float]] = None
    pooled_auc_gap_diagnostic: Optional[float] = None
    subgroup_calibration: Dict[str, float] = field(default_factory=dict)
    excluded_subgroups: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def worst_partition(self) -> Optional[str]:
        """Demographic column carrying the maximum within-partition gap."""
        if not self.per_partition_auc_gaps:
            return None
        return max(self.per_partition_auc_gaps, key=self.per_partition_auc_gaps.get)

    passed = _gate_withdrawn


@dataclass
class SensitivityResult:
    """Outputs from the Sensitivity dimension evaluation (measurement only).

    ``threshold_flip_rates`` covers the primary (narrow) band; the wide band is
    reported separately. Threshold flip rate is a functional of the score CDF
    alone and never reads ``y_true``.
    """

    threshold_flip_rates: Dict[float, float] = field(default_factory=dict)
    max_threshold_flip_rate: Optional[float] = None
    rank_stability_score: Optional[float] = None
    decision_boundary_width: Optional[float] = None
    max_tfr_ci: Optional[Tuple[float, float]] = None
    wide_band_flip_rates: Dict[float, float] = field(default_factory=dict)
    wide_band_max_tfr: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    passed = _gate_withdrawn


@dataclass
class EquityResult:
    """Outputs from the Equity dimension evaluation (measurement only).

    ``need_prediction_correlation`` is bounded above by
    ``attainable_rho_ceiling`` when the proxy is binary; comparing it to a fixed
    target without that ceiling is not meaningful across cohorts of differing
    prevalence.
    """

    need_prediction_correlation: Optional[float] = None
    attainable_rho_ceiling: Optional[float] = None
    proxy_prevalence: Optional[float] = None
    group_need_gaps: Dict[str, float] = field(default_factory=dict)
    proxy_bias_flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    passed = _gate_withdrawn


@dataclass
class DeployabilityResult:
    """Outputs from the Deployability dimension evaluation (measurement only).

    Timing fields are named for what they measure: ``batch_scoring_time_ms`` is
    whole-cohort throughput, ``single_row_latency_ms`` is the per-request
    figure. The explanation fields measure agreement between local attributions
    and a global ranking computed on a disjoint sample; they are ``None`` when
    the comparison is structurally degenerate (``d <= top_k``).
    """

    batch_scoring_time_ms: Optional[float] = None
    amortised_time_per_row_ms: Optional[float] = None
    single_row_latency_ms: Optional[float] = None
    local_global_topk_agreement: Optional[float] = None
    global_top1_in_local_topk: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    passed = _gate_withdrawn


@dataclass
class FrameworkReport:
    """Combined RISED measurements across all five dimensions.

    This object reports numbers, not verdicts. To obtain advisory verdicts pass
    it to :func:`rised.policy.evaluate_policy` together with explicit
    thresholds.
    """

    reliability: Optional[ReliabilityResult] = None
    inclusivity: Optional[InclusivityResult] = None
    sensitivity: Optional[SensitivityResult] = None
    equity: Optional[EquityResult] = None
    deployability: Optional[DeployabilityResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    #: Dimensions that must be present for any overall roll-up to be meaningful.
    DIMENSIONS = (
        "reliability",
        "inclusivity",
        "sensitivity",
        "equity",
        "deployability",
    )

    def evaluated_dimensions(self) -> List[str]:
        """Names of dimensions that were actually evaluated."""
        return [d for d in self.DIMENSIONS if getattr(self, d) is not None]

    def missing_dimensions(self) -> List[str]:
        """Names of dimensions absent from this report."""
        return [d for d in self.DIMENSIONS if getattr(self, d) is None]

    def is_complete(self) -> bool:
        """True only when every dimension was evaluated."""
        return not self.missing_dimensions()

    def measurement_summary(self) -> Dict[str, Dict[str, Any]]:
        """Headline measurement per dimension, with no verdict attached."""
        out: Dict[str, Dict[str, Any]] = {}
        rel = self.reliability
        out["reliability"] = (
            {
                "judge_sensitivity_score": rel.judge_sensitivity_score,
                "jss_ci": rel.jss_ci,
                "rank_correlation_min": rel.rank_correlation_min,
            }
            if rel is not None
            else {}
        )
        inc = self.inclusivity
        out["inclusivity"] = (
            {
                "max_partition_auc_gap": inc.auc_parity_gap,
                "auc_gap_ci": inc.auc_gap_ci,
                "per_partition_auc_gaps": dict(inc.per_partition_auc_gaps),
                "max_subgroup_ece": (
                    max(inc.subgroup_calibration.values())
                    if inc.subgroup_calibration
                    else None
                ),
                "n_excluded_subgroups": len(inc.excluded_subgroups),
            }
            if inc is not None
            else {}
        )
        sen = self.sensitivity
        out["sensitivity"] = (
            {
                "max_threshold_flip_rate": sen.max_threshold_flip_rate,
                "max_tfr_ci": sen.max_tfr_ci,
                "wide_band_max_tfr": sen.wide_band_max_tfr,
                "threshold_band": sen.details.get("threshold_band"),
            }
            if sen is not None
            else {}
        )
        eq = self.equity
        out["equity"] = (
            {
                "need_prediction_correlation": eq.need_prediction_correlation,
                "attainable_rho_ceiling": eq.attainable_rho_ceiling,
                "need_source": eq.details.get("need_source"),
            }
            if eq is not None
            else {}
        )
        dep = self.deployability
        out["deployability"] = (
            {
                "batch_scoring_time_ms": dep.batch_scoring_time_ms,
                "single_row_latency_ms": dep.single_row_latency_ms,
                "local_global_topk_agreement": dep.local_global_topk_agreement,
                "explanation_chance_level": dep.details.get(
                    "explanation_chance_level"
                ),
            }
            if dep is not None
            else {}
        )
        return out

    # ── withdrawn gate ────────────────────────────────────────────────────────
    summary = _gate_withdrawn
    diagnostic_status = _gate_withdrawn
    all_passed = _gate_withdrawn
