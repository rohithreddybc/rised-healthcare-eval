"""
Inclusivity dimension: performance consistency across clinically and
demographically distinct subpopulations, assessed through subgroup AUC
parity and calibration comparisons.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rised_v010.metrics import expected_calibration_error, roc_auc
from rised_v010.results import InclusivityResult


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

    # BCa 95% CI for AUC parity gap; percentile CIs for per-subgroup AUCs
    auc_gap_ci = None
    subgroup_auc_cis: Dict[str, tuple] = {}
    if n_bootstrap > 0 and auc_gap is not None:
        from rised_v010.bootstrap_ci import bca_interval

        rng = np.random.default_rng(random_state)
        n = len(X_arr)
        scores_full = model.predict_proba(X_arr)[:, 1]
        demo_arr = demographic_df.reset_index(drop=True)

        def _gap_on_idx(idx: np.ndarray) -> float:
            scores_b = scores_full[idx]
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
                    boot_aucs.append(roc_auc(y_b[mask_b], scores_b[mask_b]))
            if len(boot_aucs) < 2:
                return float("nan")
            return float(max(boot_aucs) - min(boot_aucs))

        gap_boot = np.empty(n_bootstrap, dtype=float)
        per_label_boot: Dict[str, List[float]] = {label: [] for label in subgroup_aucs}
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            gap_boot[b] = _gap_on_idx(idx)
            scores_b = scores_full[idx]
            y_b = y[idx]
            demo_b = demo_arr.iloc[idx]
            for col in cols:
                for grp_val in demo_b[col].unique():
                    mask_b = (demo_b[col] == grp_val).values
                    if mask_b.sum() < 30:
                        continue
                    n_pos_b = int(y_b[mask_b].sum())
                    n_neg_b = int(mask_b.sum()) - n_pos_b
                    if n_pos_b < 2 or n_neg_b < 2:
                        continue
                    label = f"{col}={grp_val}"
                    if label in per_label_boot:
                        per_label_boot[label].append(
                            roc_auc(y_b[mask_b], scores_b[mask_b])
                        )

        # Jackknife for BCa on the parity gap
        full_idx = np.arange(n)
        gap_jack = np.empty(n, dtype=float)
        for i in range(n):
            gap_jack[i] = _gap_on_idx(np.delete(full_idx, i))

        valid = ~np.isnan(gap_boot)
        if valid.any() and not np.isnan(_gap_on_idx(full_idx)):
            auc_gap_ci = bca_interval(
                auc_gap, gap_boot[valid], gap_jack[~np.isnan(gap_jack)], alpha=0.05
            )
        for label, samples in per_label_boot.items():
            if len(samples) >= 50:
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
