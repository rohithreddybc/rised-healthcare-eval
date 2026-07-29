"""
Core metric functions shared across RISED dimensions.

All functions follow a consistent interface:
  metric(y_true, y_score, **kwargs) -> float
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def roc_auc(y_true, y_score) -> float:
    """Compute ROC-AUC. Thin wrapper around sklearn for consistent imports."""
    return float(roc_auc_score(np.asarray(y_true), np.asarray(y_score)))


def brier_score(y_true, y_prob) -> float:
    """Compute the Brier score (mean squared error between labels and probabilities)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_true - y_prob) ** 2))


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE) using equal-width binning.

    References
    ----------
    Guo et al. (2017) "On calibration of modern neural networks."
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= bins[i]) & (y_prob <= bins[i + 1])
        else:
            mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def rank_correlation(scores_a, scores_b) -> float:
    """Spearman rank correlation between two score vectors."""
    return float(spearmanr(scores_a, scores_b).statistic)


def decision_flip_rate(scores_a, scores_b, threshold: float = 0.5) -> float:
    """
    Fraction of patients whose binary decision (score >= threshold) changes
    between scores_a and scores_b.
    """
    a = np.asarray(scores_a) >= threshold
    b = np.asarray(scores_b) >= threshold
    return float(np.mean(a != b))


def judge_sensitivity_score(
    baseline_scores, perturbed_scores_list, threshold: float = 0.5
) -> float:
    """
    Compute the Judge Sensitivity Score (JSS) across a set of perturbations.

    JSS is the mean perturbation flip rate across all perturbations in Phi.
    Extends the JSS definition from JudgeSense (2025) to the clinical
    decision-support context.
    """
    rates = [decision_flip_rate(baseline_scores, p, threshold) for p in perturbed_scores_list]
    return float(np.mean(rates)) if rates else 0.0
