"""
Result dataclasses for each RISED dimension and the combined FrameworkReport.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReliabilityResult:
    """Outputs from the Reliability dimension evaluation."""

    judge_sensitivity_score: Optional[float] = None
    perturbation_flip_rate: Optional[float] = None
    rank_correlation_mean: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self, threshold: float = 0.05) -> bool:
        """Return True if flip rate is below threshold."""
        if self.perturbation_flip_rate is None:
            return False
        return self.perturbation_flip_rate < threshold


@dataclass
class InclusivityResult:
    """Outputs from the Inclusivity dimension evaluation."""

    subgroup_aucs: Dict[str, float] = field(default_factory=dict)
    auc_parity_gap: Optional[float] = None
    subgroup_calibration: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self, max_gap: float = 0.05) -> bool:
        """Return True if AUC parity gap is within tolerance."""
        if self.auc_parity_gap is None:
            return False
        return self.auc_parity_gap <= max_gap


@dataclass
class SensitivityResult:
    """Outputs from the Sensitivity dimension evaluation."""

    threshold_flip_rates: Dict[float, float] = field(default_factory=dict)
    rank_stability_score: Optional[float] = None
    decision_boundary_width: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self, max_flip_rate: float = 0.10) -> bool:
        """Return True if no threshold produces flip rate above tolerance."""
        if not self.threshold_flip_rates:
            return False
        return max(self.threshold_flip_rates.values()) <= max_flip_rate


@dataclass
class EquityResult:
    """Outputs from the Equity dimension evaluation."""

    need_prediction_correlation: Optional[float] = None
    group_need_gaps: Dict[str, float] = field(default_factory=dict)
    proxy_bias_flags: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self, min_correlation: float = 0.70) -> bool:
        """Return True if need-prediction correlation meets threshold."""
        if self.need_prediction_correlation is None:
            return False
        return self.need_prediction_correlation >= min_correlation


@dataclass
class DeployabilityResult:
    """Outputs from the Deployability dimension evaluation."""

    mean_inference_latency_ms: Optional[float] = None
    explanation_faithfulness: Optional[float] = None
    top_feature_stability: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def passed(self, max_latency_ms: float = 500.0) -> bool:
        """Return True if mean latency is within operational limit."""
        if self.mean_inference_latency_ms is None:
            return False
        return self.mean_inference_latency_ms <= max_latency_ms


@dataclass
class FrameworkReport:
    """Combined RISED evaluation report across all five dimensions."""

    reliability: Optional[ReliabilityResult] = None
    inclusivity: Optional[InclusivityResult] = None
    sensitivity: Optional[SensitivityResult] = None
    equity: Optional[EquityResult] = None
    deployability: Optional[DeployabilityResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Optional[bool]]:
        """Return pass/fail status for each completed dimension."""
        return {
            "reliability": self.reliability.passed() if self.reliability else None,
            "inclusivity": self.inclusivity.passed() if self.inclusivity else None,
            "sensitivity": self.sensitivity.passed() if self.sensitivity else None,
            "equity": self.equity.passed() if self.equity else None,
            "deployability": self.deployability.passed() if self.deployability else None,
        }

    def all_passed(self) -> bool:
        """Return True only if all evaluated dimensions passed."""
        return all(v for v in self.summary().values() if v is not None)
