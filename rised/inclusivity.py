"""
Inclusivity dimension: performance consistency across clinically and
demographically distinct subpopulations, assessed through subgroup AUC
parity and calibration comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rised.metrics import expected_calibration_error, roc_auc
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

    Computes subgroup AUC and ECE for every unique value of each demographic
    column. Flags subgroups with fewer than 30 patients as informationally
    unreliable (sub-criterion I3). Skips subgroups where AUC is undefined
    (fewer than 2 positive or 2 negative labels).

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
    X_arr = np.asarray(X, dtype=float)
    y = np.asarray(y_true)
    scores = model.predict_proba(X_arr)[:, 1]

    cols = subgroup_columns if subgroup_columns is not None else list(demographic_df.columns)
    subgroup_aucs: Dict[str, float] = {}
    subgroup_ece: Dict[str, float] = {}
    small_groups: List[str] = []

    for col in cols:
        for grp_val in demographic_df[col].unique():
            mask = (demographic_df[col] == grp_val).values
            label = f"{col}={grp_val}"
            n_grp = int(mask.sum())
            if n_grp < 30:
                small_groups.append(label)
            n_pos = int(y[mask].sum())
            n_neg = n_grp - n_pos
            if n_pos < 2 or n_neg < 2:
                continue
            subgroup_aucs[label] = roc_auc(y[mask], scores[mask])
            subgroup_ece[label] = expected_calibration_error(y[mask], scores[mask])

    auc_gap: Optional[float] = None
    if len(subgroup_aucs) >= 2:
        auc_gap = max(subgroup_aucs.values()) - min(subgroup_aucs.values())

    return InclusivityResult(
        subgroup_aucs=subgroup_aucs,
        auc_parity_gap=auc_gap,
        subgroup_calibration=subgroup_ece,
        details={"small_group_flags": small_groups},
    )
