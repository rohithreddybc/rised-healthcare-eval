"""
Equity dimension: alignment between model-predicted need and observed
clinical need across patient groups, extending beyond demographic parity
to need-based fairness.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from rised.results import EquityResult


def evaluate_equity(
    model,
    X,
    y_true,
    demographic_df: pd.DataFrame,
    need_column: Optional[str] = None,
    subgroup_columns: Optional[List[str]] = None,
) -> EquityResult:
    """
    Evaluate the Equity dimension.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels (proxy for clinical need).
    demographic_df : pd.DataFrame
        Demographic/subgroup columns aligned to X.
    need_column : str, optional
        Column in demographic_df encoding an independent measure of clinical
        need (e.g., comorbidity count). If None, y_true is used as a proxy.
    subgroup_columns : list of str, optional
        Columns to evaluate for group need gaps.

    Returns
    -------
    EquityResult
    """
    raise NotImplementedError("evaluate_equity() will be implemented in Session 3.")
