"""
Inclusivity dimension: performance consistency across clinically and
demographically distinct subpopulations, assessed through subgroup AUC
parity and calibration comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from rised.results import InclusivityResult


def evaluate_inclusivity(
    model,
    X,
    y_true,
    demographic_df: pd.DataFrame,
    subgroup_columns: Optional[List[str]] = None,
) -> InclusivityResult:
    """
    Evaluate the Inclusivity dimension.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    demographic_df : pd.DataFrame
        Demographic/subgroup columns aligned to X.
    subgroup_columns : list of str, optional
        Columns in demographic_df to evaluate. If None, uses all columns.

    Returns
    -------
    InclusivityResult
    """
    raise NotImplementedError("evaluate_inclusivity() will be implemented in Session 3.")
