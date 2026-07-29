"""
RISED: Reliability, Inclusivity, Sensitivity, Equity, and Deployability
Framework for Healthcare AI Decision-Support Evaluation.

The package is organised in two layers.

**Measurement layer** — :func:`evaluate_all` and the individual ``evaluate_*``
functions return metrics with confidence intervals. Fixed, scientific, no
thresholds and no verdicts.

**Policy layer** — :mod:`rised.policy` applies thresholds *supplied by the
caller* and returns advisory verdicts. These are institutional configuration,
not validated science; no RISED cut-point has been calibrated against observed
deployment outcomes, and no verdict constitutes deployment clearance. The
former ``report.all_passed()`` gate has been withdrawn and now raises with a
pointer to :func:`rised.policy.evaluate_policy`.

Reference
---------
Bellibatlu, R. R., Singh, M., Jajoo, Y., Lakhanpal, S., & Israni, A. (2026).
"RISED: A Pre-Deployment Evaluation Framework for High-Stakes AI Decision-Support
Systems, with Application to Healthcare." Expert Systems with Applications
(under review). https://github.com/rohithreddybc/rised-healthcare-eval
"""

__version__ = "0.2.0"
__author__ = "Rohith Reddy Bellibatlu"

from rised.policy import (
    ADVISORY_NOTICE,
    PolicyReport,
    PolicyThresholds,
    Verdict,
    evaluate_policy,
)
from rised.results import (
    DeployabilityResult,
    EquityResult,
    FrameworkReport,
    InclusivityResult,
    ReliabilityResult,
    SensitivityResult,
)

__all__ = [
    "ReliabilityResult",
    "InclusivityResult",
    "SensitivityResult",
    "EquityResult",
    "DeployabilityResult",
    "FrameworkReport",
    "PolicyThresholds",
    "PolicyReport",
    "Verdict",
    "evaluate_policy",
    "ADVISORY_NOTICE",
    "evaluate_all",
]


def evaluate_all(
    model,
    X,
    y_true,
    demographic_df,
    perturbation_specs=None,
    threshold_range=None,
    feature_names=None,
    n_bootstrap: int = 0,
    random_state=None,
    tau_ref: float = 0.5,
    need_column=None,
    groups=None,
    feature_schema=None,
    min_subgroup_n: int = 30,
    **kwargs,
) -> "FrameworkReport":
    """
    Run the RISED measurement layer and return a combined FrameworkReport.

    The returned object contains measurements only. To obtain advisory verdicts,
    pass it to :func:`rised.policy.evaluate_policy` with explicit thresholds.

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
    perturbation_specs : list of dict, optional
        Perturbation specifications for the Reliability dimension. Specs are
        partitioned into semantics-preserving and covariate-shift families;
        only the former enter the Judge Sensitivity Score.
    threshold_range : array-like, optional
        Primary decision-threshold band for Sensitivity. Defaults to the narrow
        band [0.30, 0.70]; the wide band [0.10, 0.90] is always reported too.
    feature_names : list of str, optional
        Feature names, used for schema messages and interpretability reporting.
    n_bootstrap : int
        Bootstrap replicates for confidence intervals. 0 disables.
    random_state : int, optional
        Seed for bootstrap RNGs.
    tau_ref : float
        Reference decision threshold, applied consistently to Reliability flip
        rates and Sensitivity flip rates. Default 0.5 is a convention, not a
        validated operating point; see
        :func:`rised.sensitivity.suggest_tau_ref` to derive a per-cohort
        threshold. A derived threshold is never applied automatically.
    need_column : str, optional
        Column of ``demographic_df`` holding an **independent** clinical-need
        proxy for the Equity dimension. Equity requires one: when omitted the
        dimension is skipped (``report.equity is None``) and the reason is
        recorded in ``report.metadata``. ``y_true`` is never used as the proxy.
    groups : array-like, optional
        Cluster identifier per row (e.g. patient id when rows are encounters).
        When supplied, all bootstrap resampling draws whole groups and all
        jackknives delete whole groups. Default ``None`` is row-level.
    feature_schema : rised.perturbations.FeatureSchema, optional
        Column semantic types. Inferred conservatively when omitted, so binary
        and categorical columns never receive continuous noise.
    min_subgroup_n : int
        Minimum subgroup size entering the Inclusivity parity estimand; applied
        identically in the point estimate, bootstrap and jackknife.

    Returns
    -------
    FrameworkReport
    """
    import numpy as np

    from rised.deployability import evaluate_deployability
    from rised.equity import evaluate_equity
    from rised.inclusivity import evaluate_inclusivity
    from rised.reliability import evaluate_reliability
    from rised.sensitivity import evaluate_sensitivity

    X_arr = np.asarray(X)

    reliability = evaluate_reliability(
        model, X_arr,
        perturbation_specs=perturbation_specs,
        feature_names=feature_names,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        tau_ref=tau_ref,
        schema=feature_schema,
        groups=groups,
    )
    inclusivity = evaluate_inclusivity(
        model, X_arr, y_true, demographic_df,
        min_subgroup_n=min_subgroup_n,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        groups=groups,
    )
    sensitivity = evaluate_sensitivity(
        model, X_arr, y_true,
        threshold_range=threshold_range,
        tau_ref=tau_ref,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
        groups=groups,
    )

    equity = None
    equity_skipped_reason = None
    if need_column is None:
        equity_skipped_reason = (
            "Equity not evaluated: no need_column supplied. RISED no longer "
            "falls back to y_true as the clinical-need proxy, because with a "
            "binary outcome proxy the equity statistic is an affine "
            "reparameterisation of AUROC and carries no independent "
            "information. Supply an independent proxy to evaluate Equity."
        )
    else:
        equity = evaluate_equity(
            model, X_arr, y_true, demographic_df, need_column=need_column,
        )

    deployability = evaluate_deployability(
        model, X_arr,
        feature_names=feature_names,
    )

    metadata = {
        "n_samples": int(X_arr.shape[0]),
        "n_features": int(X_arr.shape[1]),
        "tau_ref": float(tau_ref),
        "clustered_resampling": groups is not None,
        "rised_version": __version__,
    }
    if equity_skipped_reason:
        metadata["equity_skipped_reason"] = equity_skipped_reason

    return FrameworkReport(
        reliability=reliability,
        inclusivity=inclusivity,
        sensitivity=sensitivity,
        equity=equity,
        deployability=deployability,
        metadata=metadata,
    )
