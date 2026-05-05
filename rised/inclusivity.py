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
    n_bootstrap: int = 0,
    random_state: Optional[int] = None,
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

    # Bootstrap 95% CI for AUC parity gap and per-subgroup AUCs
    auc_gap_ci = None
    subgroup_auc_cis: Dict[str, tuple] = {}
    if n_bootstrap > 0 and auc_gap is not None:
        rng = np.random.default_rng(random_state)
        n = len(X_arr)
        gap_boot = []
        per_label_boot: Dict[str, List[float]] = {label: [] for label in subgroup_aucs}
        demo_arr = demographic_df.reset_index(drop=True)
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            scores_b = model.predict_proba(X_arr[idx])[:, 1]
            y_b = y[idx]
            demo_b = demo_arr.iloc[idx]
            boot_aucs: List[float] = []
            for col in cols:
                for grp_val in demo_b[col].unique():
                    mask_b = (demo_b[col] == grp_val).values
                    if mask_b.sum() < 30:
                        continue
                    n_pos_b = int(y_b[mask_b].sum())
                    n_neg_b = int(mask_b.sum()) - n_pos_b
                    if n_pos_b < 2 or n_neg_b < 2:
                        continue
                    a = roc_auc(y_b[mask_b], scores_b[mask_b])
                    boot_aucs.append(a)
                    label = f"{col}={grp_val}"
                    if label in per_label_boot:
                        per_label_boot[label].append(a)
            if len(boot_aucs) >= 2:
                gap_boot.append(max(boot_aucs) - min(boot_aucs))
        if gap_boot:
            auc_gap_ci = (
                float(np.percentile(gap_boot, 2.5)),
                float(np.percentile(gap_boot, 97.5)),
            )
        for label, samples in per_label_boot.items():
            if len(samples) >= 50:  # require enough valid resamples
                subgroup_auc_cis[label] = (
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                )

    return InclusivityResult(
        subgroup_aucs=subgroup_aucs,
        auc_parity_gap=auc_gap,
        auc_gap_ci=auc_gap_ci,
        subgroup_calibration=subgroup_ece,
        details={
            "small_group_flags": small_groups,
            "subgroup_auc_cis": subgroup_auc_cis,
        },
    )
