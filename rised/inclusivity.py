"""
Inclusivity dimension: performance consistency across clinically and
demographically distinct subpopulations, assessed through subgroup AUC
parity and calibration comparisons.

Measurement notes
-----------------
1. **Per-partition gaps.** The AUC parity gap is computed *within* each
   demographic column (partition) and the headline statistic is the maximum
   over partitions. Pooling every level of every column into one flat set and
   taking a single ``max - min`` compares levels that do not partition the same
   cohort (e.g. ``race=Other`` against ``age=75+``); that pooled quantity is
   retained only as ``pooled_auc_gap_diagnostic`` and is not the headline.

2. **One exclusion rule everywhere.** Subgroups smaller than
   ``min_subgroup_n`` (default 30) or with degenerate labels are excluded
   identically in the point estimate, in every bootstrap replicate and in every
   jackknife replicate. Applying the rule in only some of those places makes the
   interval target a different parameter from the point estimate; the resulting
   BCa interval need not even contain its own point estimate. Excluded
   subgroups are reported in ``excluded_subgroups`` with the reason.

3. **Clustered resampling.** Pass ``groups`` (e.g. a patient identifier) when
   rows are not independent; resampling then draws whole groups and the
   jackknife deletes whole groups.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from rised.metrics import expected_calibration_error, roc_auc
from rised.results import InclusivityResult

#: Default minimum subgroup size for a subgroup to enter the parity estimand.
DEFAULT_MIN_SUBGROUP_N = 30


def _subgroup_aucs_by_column(
    y: np.ndarray,
    scores: np.ndarray,
    demo: pd.DataFrame,
    cols: List[str],
    min_subgroup_n: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """Subgroup AUCs organised per demographic column, plus exclusion reasons.

    This is the single definition of which subgroups are in the estimand. It is
    called by the point estimate, by every bootstrap replicate and by every
    jackknife replicate, so all three target the same parameter.
    """
    per_column: Dict[str, Dict[str, float]] = {}
    excluded: Dict[str, str] = {}
    for col in cols:
        level_aucs: Dict[str, float] = {}
        for grp_val in demo[col].unique():
            mask = (demo[col] == grp_val).values
            label = f"{col}={grp_val}"
            n_grp = int(mask.sum())
            if n_grp < min_subgroup_n:
                excluded[label] = f"n={n_grp} < min_subgroup_n={min_subgroup_n}"
                continue
            y_g = y[mask]
            n_pos = int(y_g.sum())
            n_neg = n_grp - n_pos
            if n_pos < 2 or n_neg < 2:
                excluded[label] = (
                    f"degenerate labels (n_pos={n_pos}, n_neg={n_neg}); AUC undefined"
                )
                continue
            level_aucs[label] = roc_auc(y_g, scores[mask])
        per_column[col] = level_aucs
    return per_column, excluded


def _gaps_from_per_column(
    per_column: Dict[str, Dict[str, float]]
) -> Tuple[Dict[str, float], Optional[float]]:
    """Within-partition ``max - min`` gaps and the maximum across partitions."""
    gaps: Dict[str, float] = {}
    for col, level_aucs in per_column.items():
        if len(level_aucs) >= 2:
            gaps[col] = float(max(level_aucs.values()) - min(level_aucs.values()))
    max_gap = float(max(gaps.values())) if gaps else None
    return gaps, max_gap


def evaluate_inclusivity(
    model,
    X,
    y_true,
    demographic_df: pd.DataFrame,
    subgroup_columns: Optional[List[str]] = None,
    min_subgroup_n: int = DEFAULT_MIN_SUBGROUP_N,
    n_bootstrap: int = 0,
    random_state: Optional[int] = None,
    groups=None,
) -> InclusivityResult:
    """
    Evaluate the Inclusivity dimension (measurement layer; no thresholds).

    Computes subgroup AUC and ECE for every level of each demographic column
    that meets the inclusion rule, the AUC parity gap *within* each column, and
    the maximum gap across columns.

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
    min_subgroup_n : int
        Minimum subgroup size to enter the estimand. Subgroups below this size
        are excluded from the point estimate, the bootstrap and the jackknife
        alike, and are listed in ``excluded_subgroups``. Set to 0 to include
        every subgroup with estimable AUC.
    n_bootstrap : int
        Bootstrap replicates for the BCa interval on the maximum per-partition
        gap. 0 disables interval estimation.
    random_state : int, optional
        Seed for the bootstrap RNG.
    groups : array-like of shape (n_samples,), optional
        Cluster identifier per row (e.g. patient id when rows are encounters).
        When supplied, the bootstrap resamples whole groups and the jackknife
        deletes whole groups. Default ``None`` is row-level resampling.

    Returns
    -------
    InclusivityResult
    """
    X_arr = np.asarray(X, dtype=float)
    y = np.asarray(y_true)
    scores = model.predict_proba(X_arr)[:, 1]

    cols = (
        subgroup_columns
        if subgroup_columns is not None
        else list(demographic_df.columns)
    )
    demo_arr = demographic_df.reset_index(drop=True)

    per_column, excluded = _subgroup_aucs_by_column(
        y, scores, demo_arr, cols, min_subgroup_n
    )
    per_partition_gaps, max_partition_gap = _gaps_from_per_column(per_column)

    # Flat view over included subgroups only (same set as the per-column view).
    subgroup_aucs: Dict[str, float] = {}
    for level_aucs in per_column.values():
        subgroup_aucs.update(level_aucs)

    subgroup_ece: Dict[str, float] = {}
    for col in cols:
        for grp_val in demo_arr[col].unique():
            label = f"{col}={grp_val}"
            if label not in per_column[col]:
                continue
            mask = (demo_arr[col] == grp_val).values
            subgroup_ece[label] = expected_calibration_error(y[mask], scores[mask])

    # Pooled max-min across *all* columns at once. Retained for continuity with
    # earlier releases and for diagnostics only: it compares levels drawn from
    # different partitions of the cohort and inherits the selection bias of a
    # range statistic over many overlapping groups. Not a headline metric.
    pooled_gap_diagnostic: Optional[float] = None
    if len(subgroup_aucs) >= 2:
        pooled_gap_diagnostic = float(
            max(subgroup_aucs.values()) - min(subgroup_aucs.values())
        )

    # ── BCa interval for the maximum per-partition gap ────────────────────────
    auc_gap_ci = None
    subgroup_auc_cis: Dict[str, tuple] = {}
    resampling_info: Dict[str, Any] = {}
    if n_bootstrap > 0 and max_partition_gap is not None:
        from rised.bootstrap_ci import ResamplingPlan, bca_interval

        rng = np.random.default_rng(random_state)
        n = len(X_arr)
        plan = ResamplingPlan(n, groups)
        resampling_info = plan.describe()

        def _replicate(idx: np.ndarray):
            pc, _ = _subgroup_aucs_by_column(
                y[idx], scores[idx], demo_arr.iloc[idx], cols, min_subgroup_n
            )
            gaps, mx = _gaps_from_per_column(pc)
            return pc, (float(mx) if mx is not None else float("nan"))

        gap_boot = np.empty(n_bootstrap, dtype=float)
        per_label_boot: Dict[str, List[float]] = {
            label: [] for label in subgroup_aucs
        }
        for b in range(n_bootstrap):
            idx = plan.bootstrap_index(rng)
            pc_b, gap_b = _replicate(idx)
            gap_boot[b] = gap_b
            for level_aucs in pc_b.values():
                for label, auc_val in level_aucs.items():
                    if label in per_label_boot:
                        per_label_boot[label].append(auc_val)

        gap_jack = np.array(
            [_replicate(idx)[1] for idx in plan.jackknife_index_iter()],
            dtype=float,
        )

        auc_gap_ci = bca_interval(
            max_partition_gap, gap_boot, gap_jack, alpha=0.05
        )
        for label, samples in per_label_boot.items():
            if len(samples) >= 50:
                subgroup_auc_cis[label] = (
                    float(np.percentile(samples, 2.5)),
                    float(np.percentile(samples, 97.5)),
                )

    return InclusivityResult(
        subgroup_aucs=subgroup_aucs,
        per_partition_aucs={c: dict(v) for c, v in per_column.items()},
        per_partition_auc_gaps=per_partition_gaps,
        auc_parity_gap=max_partition_gap,
        auc_gap_ci=auc_gap_ci,
        pooled_auc_gap_diagnostic=pooled_gap_diagnostic,
        subgroup_calibration=subgroup_ece,
        excluded_subgroups=excluded,
        details={
            "min_subgroup_n": min_subgroup_n,
            "subgroup_auc_cis": subgroup_auc_cis,
            "n_partitions_with_gap": len(per_partition_gaps),
            "resampling": resampling_info,
            "gap_definition": (
                "max over demographic columns of (max - min) subgroup AUC "
                "within that column"
            ),
        },
    )
