"""
Equity dimension: alignment between model-predicted need and observed
clinical need across patient groups, extending beyond demographic parity
to need-based fairness.

Key reference: Paulus & Kent (2020), "Predictably unequal", npj Digital Medicine.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rised.metrics import rank_correlation
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

    Computes the Spearman correlation between predicted scores and clinical
    need (need-prediction correlation), and the per-group need gap (mean
    predicted score minus mean clinical need) for each subgroup.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels used as a proxy for clinical need when
        ``need_column`` is not provided.
    demographic_df : pd.DataFrame
        Demographic/subgroup columns aligned to X.
    need_column : str, optional
        Column in ``demographic_df`` with an independent clinical need measure
        (e.g., comorbidity count). If provided, this column is normalized to
        [0, 1] before computing gaps. If None, ``y_true`` is used as the
        need proxy.
    subgroup_columns : list of str, optional
        Columns in ``demographic_df`` to evaluate for group need gaps.
        Defaults to all columns except ``need_column``.

    Returns
    -------
    EquityResult
    """
    X_arr = np.asarray(X, dtype=float)
    y = np.asarray(y_true, dtype=float)
    scores = model.predict_proba(X_arr)[:, 1]

    # Resolve clinical need measure
    if need_column is not None and need_column in demographic_df.columns:
        need_raw = np.asarray(demographic_df[need_column], dtype=float)
        need_min, need_max = need_raw.min(), need_raw.max()
        need = (need_raw - need_min) / (need_max - need_min) if need_max > need_min else need_raw
        need_source = need_column
    else:
        need = y
        need_source = "y_true"
        warnings.warn(
            "Using y_true as the clinical need proxy. If the model was trained on y_true, "
            "need_prediction_correlation may be inflated. Provide need_column for an "
            "independent measure of clinical need (e.g., comorbidity count, subsequent "
            "hospitalization).",
            UserWarning,
            stacklevel=2,
        )

    # Need-prediction correlation (Spearman)
    need_pred_corr = rank_correlation(scores, need)

    # Determine subgroup columns (exclude need_column)
    candidate_cols = subgroup_columns if subgroup_columns is not None else list(demographic_df.columns)
    eval_cols = [c for c in candidate_cols if c != need_column]

    group_need_gaps: Dict[str, float] = {}
    proxy_bias_flags: List[str] = []

    for col in eval_cols:
        for grp_val in demographic_df[col].unique():
            mask = (demographic_df[col] == grp_val).values
            label = f"{col}={grp_val}"
            if mask.sum() < 5:
                continue
            mean_score = float(scores[mask].mean())
            mean_need = float(need[mask].mean())
            gap = mean_score - mean_need
            group_need_gaps[label] = gap
            if abs(gap) > 0.10:
                proxy_bias_flags.append(label)

    return EquityResult(
        need_prediction_correlation=need_pred_corr,
        group_need_gaps=group_need_gaps,
        proxy_bias_flags=proxy_bias_flags,
        details={"need_source": need_source},
    )
