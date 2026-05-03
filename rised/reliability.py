"""
Reliability dimension: output stability under semantically equivalent
but differently encoded inputs, perturbation variants, and temporal re-encodings.

Formally extends the Judge Sensitivity Score (JudgeSense, 2025) to the
clinical decision-support context.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from rised.metrics import decision_flip_rate, judge_sensitivity_score, rank_correlation
from rised.perturbations import apply_perturbation
from rised.results import ReliabilityResult


def evaluate_reliability(
    model,
    X,
    perturbation_specs: Optional[List[Dict[str, Any]]] = None,
    feature_names: Optional[List[str]] = None,
) -> ReliabilityResult:
    """
    Evaluate the Reliability dimension.

    Computes the Judge Sensitivity Score (JSS), mean perturbation flip rate,
    and mean rank correlation across all perturbation variants.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Original (unperturbed) feature matrix.
    perturbation_specs : list of dict, optional
        Each dict specifies one perturbation: ``{"type": str, ...}``.
        If None or empty, returns perfect-stability scores (JSS=0, rho=1).
    feature_names : list of str, optional
        Feature names corresponding to columns of X. Not used in computation
        but stored in details for downstream reporting.

    Returns
    -------
    ReliabilityResult
    """
    X_arr = np.asarray(X, dtype=float)
    baseline = model.predict_proba(X_arr)[:, 1]

    if not perturbation_specs:
        return ReliabilityResult(
            judge_sensitivity_score=0.0,
            perturbation_flip_rate=0.0,
            rank_correlation_mean=1.0,
            details={"feature_names": feature_names},
        )

    per_perturbation_flip: Dict[str, float] = {}
    per_perturbation_rho: Dict[str, float] = {}
    perturbed_scores: List[np.ndarray] = []

    for spec in perturbation_specs:
        label = spec.get("label", spec["type"])
        X_pert = apply_perturbation(X_arr, spec)
        scores = model.predict_proba(X_pert)[:, 1]
        perturbed_scores.append(scores)
        per_perturbation_flip[label] = decision_flip_rate(baseline, scores)
        per_perturbation_rho[label] = rank_correlation(baseline, scores)

    jss = judge_sensitivity_score(baseline, perturbed_scores)
    mean_flip = float(np.mean(list(per_perturbation_flip.values())))
    mean_rho = float(np.mean(list(per_perturbation_rho.values())))

    return ReliabilityResult(
        judge_sensitivity_score=jss,
        perturbation_flip_rate=mean_flip,
        rank_correlation_mean=mean_rho,
        details={
            "per_perturbation_flip_rate": per_perturbation_flip,
            "per_perturbation_rank_correlation": per_perturbation_rho,
            "feature_names": feature_names,
        },
    )
