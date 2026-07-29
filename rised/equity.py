"""
Equity dimension: alignment between model-predicted need and observed
clinical need across patient groups, extending beyond demographic parity
to need-based fairness.

Key reference: Paulus & Kent (2020), "Predictably unequal", npj Digital Medicine.

Why ``y_true`` is no longer accepted as the need proxy
------------------------------------------------------
With a *binary* need proxy of prevalence ``p`` and untied scores, the Spearman
correlation between scores and proxy satisfies, exactly,

    rho = sqrt(12 p (1 - p)) * (n / sqrt(n^2 - 1)) * (AUC - 0.5)

(verified numerically to 4.4e-16). When the proxy is the outcome the model was
trained and evaluated on, ``AUC`` is the model's own AUROC, so ``rho`` is an
affine reparameterisation of AUROC and carries no information about equity that
the discrimination metric did not already carry. Defaulting to ``y_true``
therefore produced a fairness number that could not fail independently of
accuracy. An independent proxy is now required and an outcome-derived proxy
raises.

The attainable ceiling
----------------------
The same identity bounds ``rho`` at ``AUC = 1``:

    rho_max = sqrt(3 p (1 - p))   (asymptotically; finite-n form below)

so ``rho`` is not comparable across cohorts of different prevalence, and a fixed
target such as 0.70 is **mathematically unreachable** outside
``p`` in [0.2056, 0.7944] at any model quality. At ``p = 0.112`` the ceiling is
0.546. No fixed threshold is applied here; the ceiling and the ratio
``rho / rho_max`` are reported so the number can be read correctly.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from rised.metrics import rank_correlation
from rised.results import EquityResult

#: Prevalence interval in which a Spearman target of 0.70 is attainable at all.
#: Solves sqrt(3 p (1 - p)) >= 0.70, i.e. 3p^2 - 3p + 0.49 <= 0.
CEILING_070_ATTAINABLE_PREVALENCE = (0.2056, 0.7944)


def attainable_rho_ceiling(prevalence: float, n: Optional[int] = None) -> float:
    """Maximum attainable Spearman rho against a binary proxy.

    Parameters
    ----------
    prevalence : float
        Fraction of the cohort positive on the binary need proxy.
    n : int, optional
        Cohort size. When given, applies the finite-sample factor
        ``n / sqrt(n^2 - 1)``; otherwise returns the asymptotic
        ``sqrt(3 p (1 - p))``.
    """
    p = float(prevalence)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"prevalence must be in [0, 1]; got {p!r}.")
    ceiling = float(np.sqrt(3.0 * p * (1.0 - p)))
    if n is not None and n > 1:
        ceiling *= float(n / np.sqrt(n * n - 1.0))
    return ceiling


def _is_outcome_derived(need: np.ndarray, y: np.ndarray) -> Optional[str]:
    """Return a reason string if ``need`` is derived from ``y``, else None."""
    need = np.asarray(need, dtype=float)
    y = np.asarray(y, dtype=float)
    if need.shape != y.shape:
        return None

    finite = np.isfinite(need) & np.isfinite(y)
    if finite.sum() < 3:
        return None
    need_f, y_f = need[finite], y[finite]

    rng_need = need_f.max() - need_f.min()
    rng_y = y_f.max() - y_f.min()
    if rng_need > 0 and rng_y > 0:
        if np.allclose((need_f - need_f.min()) / rng_need,
                       (y_f - y_f.min()) / rng_y, atol=1e-12):
            return "proxy is an affine transform of y_true"

    # Deterministic function of a low-cardinality outcome: the proxy takes a
    # single value within each outcome level, i.e. it is a relabelling of y.
    y_levels = np.unique(y_f)
    if 2 <= len(y_levels) <= 10:
        counts = [int((y_f == lvl).sum()) for lvl in y_levels]
        if min(counts) >= 2:
            if all(
                np.ptp(need_f[y_f == lvl]) <= 1e-12 for lvl in y_levels
            ):
                return "proxy is constant within each y_true level (a relabelling of y_true)"

    if len(np.unique(need_f)) > 1 and len(y_levels) > 1:
        rho = rank_correlation(need_f, y_f)
        if np.isfinite(rho) and abs(rho) >= 1.0 - 1e-9:
            return f"proxy is perfectly rank-correlated with y_true (rho={rho:.6f})"
    return None


def evaluate_equity(
    model,
    X,
    y_true,
    demographic_df: pd.DataFrame,
    need_column: Optional[str] = None,
    subgroup_columns: Optional[List[str]] = None,
    gap_flag_threshold: float = 0.10,
) -> EquityResult:
    """
    Evaluate the Equity dimension (measurement layer; no pass/fail threshold).

    Computes the Spearman correlation between predicted scores and an
    **independent** measure of clinical need, the attainable ceiling for that
    proxy, and the per-group need gap (mean predicted score minus mean need).

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth labels. Used **only** to verify that the supplied proxy is
        not derived from the outcome. Never used as the proxy itself.
    demographic_df : pd.DataFrame
        Demographic/subgroup columns aligned to X.
    need_column : str
        Required. Column in ``demographic_df`` holding an independent clinical
        need measure (e.g. comorbidity count, subsequent hospitalisation,
        acuity score). Normalised to [0, 1] before gaps are computed.
    subgroup_columns : list of str, optional
        Columns to evaluate for group need gaps. Defaults to all columns except
        ``need_column``.
    gap_flag_threshold : float
        Absolute need-gap magnitude above which a subgroup is flagged. This is
        institutional configuration, not a validated cut-point; it only
        populates ``proxy_bias_flags``.

    Raises
    ------
    ValueError
        If ``need_column`` is missing, or if the supplied proxy is derived from
        ``y_true`` (identical, an affine transform, a relabelling, or perfectly
        rank-correlated). Such a proxy makes the correlation an affine
        reparameterisation of AUROC.

    Returns
    -------
    EquityResult
    """
    X_arr = np.asarray(X, dtype=float)
    y = np.asarray(y_true, dtype=float)
    scores = model.predict_proba(X_arr)[:, 1]

    if need_column is None:
        raise ValueError(
            "evaluate_equity requires an explicit need_column naming an "
            "independent measure of clinical need. Using y_true as the proxy "
            "makes the reported correlation an affine reparameterisation of "
            "AUROC (rho = sqrt(12p(1-p)) * (n/sqrt(n^2-1)) * (AUC - 0.5)), so "
            "it cannot fail independently of discrimination. Supply e.g. a "
            "comorbidity count, an acuity score, or subsequent utilisation."
        )
    if need_column not in demographic_df.columns:
        raise ValueError(
            f"need_column={need_column!r} is not a column of demographic_df "
            f"(columns: {list(demographic_df.columns)})."
        )

    need_raw = np.asarray(demographic_df[need_column], dtype=float)
    reason = _is_outcome_derived(need_raw, y)
    if reason is not None:
        raise ValueError(
            f"need_column={need_column!r} is outcome-derived: {reason}. "
            "An outcome-derived proxy reduces the equity statistic to a "
            "monotone function of AUROC and carries no additional information. "
            "Supply a proxy measured independently of the training target."
        )

    need_min, need_max = need_raw.min(), need_raw.max()
    need = (
        (need_raw - need_min) / (need_max - need_min)
        if need_max > need_min
        else np.zeros_like(need_raw)
    )

    need_pred_corr = rank_correlation(scores, need)

    # Attainable ceiling. Defined by the binary-proxy identity; for a proxy with
    # more than two levels the bound does not apply and is reported as None.
    n = len(need_raw)
    proxy_levels = int(len(np.unique(need_raw)))
    if proxy_levels == 2:
        proxy_prevalence = float(np.mean(need_raw == need_raw.max()))
        ceiling = attainable_rho_ceiling(proxy_prevalence, n=n)
        ceiling_note = (
            f"Binary proxy with prevalence p={proxy_prevalence:.4f}; maximum "
            f"attainable Spearman rho is sqrt(3p(1-p)) = {ceiling:.4f}. A fixed "
            "target of 0.70 is unreachable at any model quality outside "
            f"p in [{CEILING_070_ATTAINABLE_PREVALENCE[0]}, "
            f"{CEILING_070_ATTAINABLE_PREVALENCE[1]}]."
        )
    else:
        proxy_prevalence = None
        ceiling = None
        ceiling_note = (
            f"Proxy has {proxy_levels} distinct levels; the binary-proxy "
            "ceiling sqrt(3p(1-p)) does not apply. Ties in the proxy still "
            "bound rho below 1."
        )

    candidate_cols = (
        subgroup_columns if subgroup_columns is not None else list(demographic_df.columns)
    )
    eval_cols = [c for c in candidate_cols if c != need_column]

    group_need_gaps: Dict[str, float] = {}
    proxy_bias_flags: List[str] = []

    for col in eval_cols:
        for grp_val in demographic_df[col].unique():
            mask = (demographic_df[col] == grp_val).values
            label = f"{col}={grp_val}"
            if mask.sum() < 5:
                continue
            gap = float(scores[mask].mean()) - float(need[mask].mean())
            group_need_gaps[label] = gap
            if abs(gap) > gap_flag_threshold:
                proxy_bias_flags.append(label)

    return EquityResult(
        need_prediction_correlation=need_pred_corr,
        attainable_rho_ceiling=ceiling,
        proxy_prevalence=proxy_prevalence,
        group_need_gaps=group_need_gaps,
        proxy_bias_flags=proxy_bias_flags,
        details={
            "need_source": need_column,
            "proxy_levels": proxy_levels,
            "ceiling_note": ceiling_note,
            "correlation_as_fraction_of_ceiling": (
                float(need_pred_corr / ceiling)
                if ceiling not in (None, 0.0) and np.isfinite(need_pred_corr)
                else None
            ),
            "gap_flag_threshold": gap_flag_threshold,
        },
    )
