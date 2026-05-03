"""
Sensitivity dimension: behavioral stability under small perturbations to
input data or decision thresholds, measured through rank stability and
decision flip rates across a threshold sweep.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from rised.results import SensitivityResult


def evaluate_sensitivity(
    model,
    X,
    y_true,
    threshold_range: Optional[np.ndarray] = None,
) -> SensitivityResult:
    """
    Evaluate the Sensitivity dimension.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    threshold_range : array-like, optional
        Decision thresholds to sweep. Defaults to np.linspace(0.1, 0.9, 17).

    Returns
    -------
    SensitivityResult
    """
    raise NotImplementedError("evaluate_sensitivity() will be implemented in Session 3.")
