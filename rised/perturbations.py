"""
Perturbation generators for the Reliability dimension.

Each function takes a feature matrix X and returns a perturbed copy that
is semantically equivalent to the original (same clinical meaning, different
encoding or measurement variant).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def apply_perturbation(X, spec: Dict[str, Any]):
    """
    Dispatch to the appropriate perturbation function given a spec dict.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix.
    spec : dict
        Must contain ``"type"`` key. Additional keys are perturbation-specific.

    Returns
    -------
    X_perturbed : same type as X
    """
    raise NotImplementedError("apply_perturbation() will be implemented in Session 3.")


def gaussian_noise(X, feature_indices: Optional[List[int]] = None, scale: float = 0.01):
    """Add small Gaussian noise to continuous features."""
    raise NotImplementedError("gaussian_noise() will be implemented in Session 3.")


def unit_rescaling(X, feature_index: int, factor: float):
    """Rescale a single feature by a constant factor (e.g., kg → lb conversion)."""
    raise NotImplementedError("unit_rescaling() will be implemented in Session 3.")


def temporal_jitter(X, date_feature_index: int, max_days: int = 3):
    """Add ±max_days jitter to a date-encoded feature."""
    raise NotImplementedError("temporal_jitter() will be implemented in Session 3.")
