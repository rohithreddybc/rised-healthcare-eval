"""
Policy layer: user-supplied thresholds producing **advisory** verdicts.

This module is deliberately separate from the measurement layer
(:mod:`rised.results`). Everything here is institutional configuration, not
validated science:

* No cut-point in RISED has been calibrated against observed deployment
  outcomes. A verdict of ``MEETS_POLICY`` means "this measurement satisfies the
  threshold you supplied", never "this model is cleared for deployment".
* Thresholds default to ``None``. A dimension with no configured threshold is
  ``NOT_CONFIGURED``, which propagates to an ``INDETERMINATE`` overall verdict.
  Silence is never treated as consent.
* A dimension that was not evaluated is ``NOT_EVALUATED``, which also propagates
  to ``INDETERMINATE``. A partial report can never roll up to a positive
  verdict: a report containing only Reliability cannot clear the framework.
* When a confidence interval is available it is used: the verdict is
  ``INDETERMINATE`` whenever the interval straddles the threshold, because the
  data do not distinguish the two conclusions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from rised.results import FrameworkReport

ADVISORY_NOTICE = (
    "ADVISORY ONLY. RISED policy verdicts apply thresholds chosen by the "
    "operating institution. They are not validated against deployment outcomes "
    "and do not constitute deployment clearance."
)

#: Dimensions rolled up into the overall advisory verdict. Equity is excluded
#: because its statistic is proxy-dependent and bounded by a prevalence-specific
#: ceiling, so a fixed cut-point is not interpretable across cohorts; it is
#: always reported as DIAGNOSTIC.
ROLLUP_DIMENSIONS = ("reliability", "inclusivity", "sensitivity", "deployability")


class Verdict(str, Enum):
    """Advisory verdict for one dimension or for the roll-up."""

    MEETS = "MEETS_POLICY"
    DOES_NOT_MEET = "DOES_NOT_MEET_POLICY"
    INDETERMINATE = "INDETERMINATE"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    NOT_EVALUATED = "NOT_EVALUATED"
    DIAGNOSTIC = "DIAGNOSTIC"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass
class PolicyThresholds:
    """Institution-supplied thresholds. Every field defaults to ``None``.

    A ``None`` threshold means "not configured", which makes the corresponding
    dimension ``NOT_CONFIGURED`` and the overall verdict ``INDETERMINATE``.
    """

    # Reliability
    max_judge_sensitivity_score: Optional[float] = None
    min_rank_correlation: Optional[float] = None  # applied to the MINIMUM (R2)
    # Inclusivity
    max_partition_auc_gap: Optional[float] = None
    max_subgroup_ece: Optional[float] = None
    # Sensitivity
    max_threshold_flip_rate: Optional[float] = None
    # Deployability
    max_single_row_latency_ms: Optional[float] = None
    max_batch_scoring_time_ms: Optional[float] = None
    #: Use confidence intervals when available; INDETERMINATE if they straddle.
    use_confidence_intervals: bool = True


@dataclass
class CriterionResult:
    """One threshold comparison."""

    name: str
    value: Optional[float]
    threshold: Optional[float]
    ci: Optional[Tuple[float, float]]
    verdict: Verdict
    note: str = ""


@dataclass
class DimensionVerdict:
    """Advisory verdict for one dimension, with its constituent criteria."""

    dimension: str
    verdict: Verdict
    criteria: List[CriterionResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class PolicyReport:
    """Advisory verdicts under a given :class:`PolicyThresholds`."""

    dimensions: Dict[str, DimensionVerdict]
    thresholds: PolicyThresholds
    notice: str = ADVISORY_NOTICE

    def overall_verdict(self) -> Verdict:
        """Roll-up over :data:`ROLLUP_DIMENSIONS`.

        ``INDETERMINATE`` whenever any rolled-up dimension is missing,
        unconfigured or itself indeterminate. Never ``MEETS`` on a partial
        report.
        """
        verdicts = [self.dimensions[d].verdict for d in ROLLUP_DIMENSIONS]
        if any(
            v in (Verdict.NOT_EVALUATED, Verdict.NOT_CONFIGURED, Verdict.INDETERMINATE)
            for v in verdicts
        ):
            return Verdict.INDETERMINATE
        if all(v is Verdict.MEETS for v in verdicts):
            return Verdict.MEETS
        return Verdict.DOES_NOT_MEET

    def verdict_table(self) -> Dict[str, str]:
        """Dimension -> verdict string, plus ``overall``."""
        out = {d: v.verdict.value for d, v in self.dimensions.items()}
        out["overall"] = self.overall_verdict().value
        return out

    def explain(self) -> str:
        """Human-readable rendering, including the advisory notice."""
        lines = [self.notice, ""]
        for name in FrameworkReport.DIMENSIONS:
            dv = self.dimensions[name]
            lines.append(f"{name}: {dv.verdict.value}")
            for c in dv.criteria:
                val = "—" if c.value is None else f"{c.value:.4f}"
                thr = "—" if c.threshold is None else f"{c.threshold:g}"
                ci = "—" if not c.ci else f"[{c.ci[0]:.4f}, {c.ci[1]:.4f}]"
                lines.append(
                    f"    {c.name}: value={val} ci={ci} threshold={thr} "
                    f"-> {c.verdict.value}"
                    + (f"  ({c.note})" if c.note else "")
                )
            for note in dv.notes:
                lines.append(f"    note: {note}")
        lines.append("")
        lines.append(f"OVERALL (advisory): {self.overall_verdict().value}")
        return "\n".join(lines)


def _compare(
    name: str,
    value: Optional[float],
    threshold: Optional[float],
    ci: Optional[Tuple[float, float]],
    lower_is_better: bool,
    use_ci: bool,
) -> CriterionResult:
    """Compare one measurement to one threshold, CI-aware."""
    if threshold is None:
        return CriterionResult(
            name, value, None, ci, Verdict.NOT_CONFIGURED,
            "no threshold supplied",
        )
    if value is None:
        return CriterionResult(
            name, None, threshold, ci, Verdict.NOT_EVALUATED,
            "measurement unavailable",
        )
    if use_ci and ci is not None and all(c is not None and c == c for c in ci):
        lo, hi = float(ci[0]), float(ci[1])
        if lower_is_better:
            if hi <= threshold:
                return CriterionResult(name, value, threshold, (lo, hi), Verdict.MEETS)
            if lo > threshold:
                return CriterionResult(
                    name, value, threshold, (lo, hi), Verdict.DOES_NOT_MEET
                )
        else:
            if lo >= threshold:
                return CriterionResult(name, value, threshold, (lo, hi), Verdict.MEETS)
            if hi < threshold:
                return CriterionResult(
                    name, value, threshold, (lo, hi), Verdict.DOES_NOT_MEET
                )
        return CriterionResult(
            name, value, threshold, (lo, hi), Verdict.INDETERMINATE,
            "95% CI straddles the threshold",
        )
    ok = value <= threshold if lower_is_better else value >= threshold
    return CriterionResult(
        name, value, threshold, ci,
        Verdict.MEETS if ok else Verdict.DOES_NOT_MEET,
        "point estimate only; no interval available",
    )


def _combine(criteria: List[CriterionResult]) -> Verdict:
    """Combine criteria within a dimension (conjunction, indeterminacy wins).

    Criteria the institution did not configure are not part of its policy and
    are dropped. If *nothing* in the dimension was configured, the dimension is
    ``NOT_CONFIGURED`` — which makes the overall roll-up ``INDETERMINATE``.
    A configured criterion whose measurement is missing stays indeterminate.
    """
    configured = [c for c in criteria if c.verdict is not Verdict.NOT_CONFIGURED]
    if not configured:
        return Verdict.NOT_CONFIGURED
    verdicts = [c.verdict for c in configured]
    if any(v is Verdict.DOES_NOT_MEET for v in verdicts):
        return Verdict.DOES_NOT_MEET
    if any(v in (Verdict.INDETERMINATE, Verdict.NOT_EVALUATED) for v in verdicts):
        return Verdict.INDETERMINATE
    return Verdict.MEETS


def evaluate_policy(
    report: FrameworkReport,
    thresholds: Optional[PolicyThresholds] = None,
) -> PolicyReport:
    """Apply institution-supplied thresholds to a measured report.

    Parameters
    ----------
    report : FrameworkReport
        Measurements from :func:`rised.evaluate_all` or the individual
        ``evaluate_*`` functions.
    thresholds : PolicyThresholds, optional
        Institutional configuration. When omitted, nothing is configured and
        every rolled-up dimension is ``NOT_CONFIGURED``, giving an
        ``INDETERMINATE`` overall verdict.

    Returns
    -------
    PolicyReport
    """
    thresholds = thresholds or PolicyThresholds()
    use_ci = thresholds.use_confidence_intervals
    dims: Dict[str, DimensionVerdict] = {}

    # ── Reliability ───────────────────────────────────────────────────────────
    rel = report.reliability
    if rel is None:
        dims["reliability"] = DimensionVerdict(
            "reliability", Verdict.NOT_EVALUATED, notes=["dimension not evaluated"]
        )
    else:
        criteria = [
            _compare(
                "judge_sensitivity_score",
                rel.judge_sensitivity_score,
                thresholds.max_judge_sensitivity_score,
                rel.jss_ci,
                lower_is_better=True,
                use_ci=use_ci,
            ),
            # R2 is "rho >= threshold for EVERY perturbation", i.e. the minimum.
            _compare(
                "rank_correlation_min",
                rel.rank_correlation_min,
                thresholds.min_rank_correlation,
                None,
                lower_is_better=False,
                use_ci=use_ci,
            ),
        ]
        notes = []
        if rel.details.get("status") == "not_evaluated":
            notes.append(str(rel.details.get("reason", "")))
        if rel.details.get("n_covariate_shift"):
            notes.append(
                f"{rel.details['n_covariate_shift']} covariate-shift "
                "perturbation(s) excluded from JSS"
            )
        dims["reliability"] = DimensionVerdict(
            "reliability", _combine(criteria), criteria, notes
        )

    # ── Inclusivity ───────────────────────────────────────────────────────────
    inc = report.inclusivity
    if inc is None:
        dims["inclusivity"] = DimensionVerdict(
            "inclusivity", Verdict.NOT_EVALUATED, notes=["dimension not evaluated"]
        )
    else:
        max_ece = (
            max(inc.subgroup_calibration.values())
            if inc.subgroup_calibration
            else None
        )
        criteria = [
            _compare(
                "max_partition_auc_gap",
                inc.auc_parity_gap,
                thresholds.max_partition_auc_gap,
                inc.auc_gap_ci,
                lower_is_better=True,
                use_ci=use_ci,
            ),
            _compare(
                "max_subgroup_ece",
                max_ece,
                thresholds.max_subgroup_ece,
                None,
                lower_is_better=True,
                use_ci=use_ci,
            ),
        ]
        notes = []
        if inc.excluded_subgroups:
            notes.append(
                f"{len(inc.excluded_subgroups)} subgroup(s) excluded from the "
                "estimand: " + ", ".join(sorted(inc.excluded_subgroups))
            )
        if inc.worst_partition:
            notes.append(f"widest partition: {inc.worst_partition}")
        dims["inclusivity"] = DimensionVerdict(
            "inclusivity", _combine(criteria), criteria, notes
        )

    # ── Sensitivity ───────────────────────────────────────────────────────────
    sen = report.sensitivity
    if sen is None:
        dims["sensitivity"] = DimensionVerdict(
            "sensitivity", Verdict.NOT_EVALUATED, notes=["dimension not evaluated"]
        )
    else:
        criteria = [
            _compare(
                "max_threshold_flip_rate",
                sen.max_threshold_flip_rate,
                thresholds.max_threshold_flip_rate,
                sen.max_tfr_ci,
                lower_is_better=True,
                use_ci=use_ci,
            )
        ]
        notes = [
            "threshold flip rate never reads y_true; read alongside a "
            "discrimination metric",
        ]
        band = sen.details.get("threshold_band")
        if band:
            notes.append(f"primary band {band[0]:g}-{band[1]:g}")
        dims["sensitivity"] = DimensionVerdict(
            "sensitivity", _combine(criteria), criteria, notes
        )

    # ── Equity (always diagnostic) ────────────────────────────────────────────
    eq = report.equity
    if eq is None:
        dims["equity"] = DimensionVerdict(
            "equity", Verdict.NOT_EVALUATED, notes=["dimension not evaluated"]
        )
    else:
        notes = [
            "Equity is reported as a diagnostic. No fixed correlation target is "
            "applied: the statistic is bounded by a prevalence-specific ceiling "
            "and is not comparable across cohorts.",
            str(eq.details.get("ceiling_note", "")),
        ]
        dims["equity"] = DimensionVerdict("equity", Verdict.DIAGNOSTIC, [], notes)

    # ── Deployability ─────────────────────────────────────────────────────────
    dep = report.deployability
    if dep is None:
        dims["deployability"] = DimensionVerdict(
            "deployability", Verdict.NOT_EVALUATED, notes=["dimension not evaluated"]
        )
    else:
        criteria = [
            _compare(
                "single_row_latency_ms",
                dep.single_row_latency_ms,
                thresholds.max_single_row_latency_ms,
                None,
                lower_is_better=True,
                use_ci=use_ci,
            ),
            _compare(
                "batch_scoring_time_ms",
                dep.batch_scoring_time_ms,
                thresholds.max_batch_scoring_time_ms,
                None,
                lower_is_better=True,
                use_ci=use_ci,
            ),
        ]
        notes = [str(dep.details.get("timing_note", ""))]
        if dep.details.get("explanation_metrics_undefined_reason"):
            notes.append(str(dep.details["explanation_metrics_undefined_reason"]))
        dims["deployability"] = DimensionVerdict(
            "deployability", _combine(criteria), criteria, notes
        )

    return PolicyReport(dimensions=dims, thresholds=thresholds)
