"""
Perturbation generators for the Reliability dimension.

Each function takes a feature matrix X and returns a perturbed copy that
is semantically equivalent to the original (same clinical meaning, different
encoding or measurement variant).

Perturbation spec format: {"type": str, ...kwargs}
Supported types: "gaussian_noise", "unit_rescaling", "temporal_jitter"
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def gaussian_noise(
    X,
    feature_indices: Optional[List[int]] = None,
    scale: float = 0.01,
    random_state: Optional[int] = None,
):
    """
    Add small Gaussian noise scaled by each feature's standard deviation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    feature_indices : list of int, optional
        Columns to perturb. If None, all columns are perturbed.
    scale : float
        Noise magnitude as a fraction of each feature's std. Default 0.01 = 1%.
    random_state : int, optional
        Seed for reproducibility.
    """
    X_arr = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(random_state)
    indices = feature_indices if feature_indices is not None else list(range(X_arr.shape[1]))
    for idx in indices:
        col_std = X_arr[:, idx].std()
        if col_std == 0:
            continue
        X_arr[:, idx] += rng.normal(0.0, scale * col_std, size=X_arr.shape[0])
    return X_arr


def unit_rescaling(X, feature_index: int, factor: float):
    """
    Rescale a single feature column by a constant factor.

    Simulates unit-of-measurement encoding differences (e.g., kg vs. lb).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    feature_index : int
        Column index of the feature to rescale.
    factor : float
        Multiplicative rescaling factor.
    """
    X_arr = np.array(X, dtype=float, copy=True)
    X_arr[:, feature_index] *= factor
    return X_arr


def temporal_jitter(
    X,
    date_feature_index: int,
    max_days: int = 3,
    random_state: Optional[int] = None,
):
    """
    Add uniform integer noise in [-max_days, max_days] to a date-encoded feature.

    Simulates minor encoding differences in date-derived features (e.g.,
    age in days, days-since-last-visit).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    date_feature_index : int
        Column index of the date-encoded feature.
    max_days : int
        Maximum absolute jitter in days. Default 3.
    random_state : int, optional
        Seed for reproducibility.
    """
    X_arr = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(random_state)
    jitter = rng.integers(-max_days, max_days + 1, size=X_arr.shape[0]).astype(float)
    X_arr[:, date_feature_index] += jitter
    return X_arr


def apply_perturbation(X, spec: Dict[str, Any]):
    """
    Dispatch to the appropriate perturbation function given a spec dict.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix.
    spec : dict
        Must contain ``"type"`` key. Additional keys are perturbation-specific:
        - gaussian_noise: feature_indices (optional), scale (default 0.01),
          random_state (optional)
        - unit_rescaling: feature_index (required), factor (required)
        - temporal_jitter: date_feature_index (required), max_days (default 3),
          random_state (optional)

    Returns
    -------
    np.ndarray : perturbed copy of X
    """
    ptype = spec["type"]
    if ptype == "gaussian_noise":
        return gaussian_noise(
            X,
            feature_indices=spec.get("feature_indices"),
            scale=spec.get("scale", 0.01),
            random_state=spec.get("random_state"),
        )
    elif ptype == "unit_rescaling":
        return unit_rescaling(
            X,
            feature_index=spec["feature_index"],
            factor=spec["factor"],
        )
    elif ptype == "temporal_jitter":
        return temporal_jitter(
            X,
            date_feature_index=spec["date_feature_index"],
            max_days=spec.get("max_days", 3),
            random_state=spec.get("random_state"),
        )
    else:
        raise ValueError(f"Unknown perturbation type: {ptype!r}. "
                         f"Supported: 'gaussian_noise', 'unit_rescaling', 'temporal_jitter'.")
