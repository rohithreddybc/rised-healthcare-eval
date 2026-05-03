"""
Core metric functions shared across RISED dimensions.

All functions follow a consistent interface:
  metric(y_true, y_score, **kwargs) -> float
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def roc_auc(y_true, y_score) -> float:
    """Compute ROC-AUC. Thin wrapper around sklearn for consistent imports."""
    raise NotImplementedError("roc_auc() will be implemented in Session 3.")


def brier_score(y_true, y_prob) -> float:
    """Compute the Brier score (mean squared calibration error)."""
    raise NotImplementedError("brier_score() will be implemented in Session 3.")


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width binning.

    References
    ----------
    Guo et al. (2017) "On calibration of modern neural networks."
    """
    raise NotImplementedError("expected_calibration_error() will be implemented in Session 3.")


def rank_correlation(scores_a, scores_b) -> float:
    """Spearman rank correlation between two score vectors."""
    raise NotImplementedError("rank_correlation() will be implemented in Session 3.")


def decision_flip_rate(scores_a, scores_b, threshold: float = 0.5) -> float:
    """
    Fraction of patients whose binary decision (score >= threshold) changes
    between scores_a and scores_b.
    """
    raise NotImplementedError("decision_flip_rate() will be implemented in Session 3.")


def judge_sensitivity_score(baseline_scores, perturbed_scores_list) -> float:
    """
    Compute the Judge Sensitivity Score (JSS) across a set of perturbations.

    Extends the JSS definition from JudgeSense (2025) to the clinical
    decision-support context.
    """
    raise NotImplementedError("judge_sensitivity_score() will be implemented in Session 3.")
