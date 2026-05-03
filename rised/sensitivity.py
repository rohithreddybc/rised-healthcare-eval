"""
Sensitivity dimension: behavioral stability under small perturbations to
decision thresholds, measured through threshold sweep flip rates and the
fraction of patients in the borderline decision zone.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from rised.results import SensitivityResult


def evaluate_sensitivity(
    model,
    X,
    y_true,
    threshold_range: Optional[np.ndarray] = None,
    tau_ref: float = 0.5,
    boundary_delta: float = 0.05,
) -> SensitivityResult:
    """
    Evaluate the Sensitivity dimension.

    For each threshold in ``threshold_range``, computes the fraction of patients
    whose binary decision differs from the decision at ``tau_ref`` (Threshold
    Flip Rate). Also computes the fraction of patients with scores within
    ``boundary_delta`` of ``tau_ref`` (decision boundary width).

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (not used in computation; kept for API
        symmetry with other evaluate_* functions).
    threshold_range : array-like, optional
        Decision thresholds to sweep. Defaults to np.linspace(0.1, 0.9, 17).
    tau_ref : float
        Reference (operational) threshold. Flip rates are computed relative
        to this threshold. Default 0.5.
    boundary_delta : float
        Half-width of the borderline zone around tau_ref. Patients with
        |score - tau_ref| <= boundary_delta are borderline-sensitive.
        Default 0.05.

    Returns
    -------
    SensitivityResult
    """
    X_arr = np.asarray(X, dtype=float)
    scores = model.predict_proba(X_arr)[:, 1]

    if threshold_range is None:
        threshold_range = np.linspace(0.1, 0.9, 17)

    ref_decisions = scores >= tau_ref

    threshold_flip_rates: Dict[float, float] = {}
    for tau in threshold_range:
        tau_f = float(round(float(tau), 8))
        decisions = scores >= tau_f
        threshold_flip_rates[tau_f] = float(np.mean(ref_decisions != decisions))

    mean_flip = float(np.mean(list(threshold_flip_rates.values())))
    rank_stability_score = 1.0 - mean_flip

    decision_boundary_width = float(np.mean(np.abs(scores - tau_ref) <= boundary_delta))

    return SensitivityResult(
        threshold_flip_rates=threshold_flip_rates,
        rank_stability_score=rank_stability_score,
        decision_boundary_width=decision_boundary_width,
        details={
            "reference_threshold": tau_ref,
            "boundary_delta": boundary_delta,
            "n_thresholds_evaluated": len(threshold_flip_rates),
        },
    )
