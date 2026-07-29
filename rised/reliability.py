"""
Reliability dimension: output stability under semantically equivalent
but differently encoded inputs.

Formally extends the Judge Sensitivity Score (JudgeSense, 2025) to the
clinical decision-support context.

Two things this module deliberately does *not* do
-------------------------------------------------
1. It does not mix covariate-shift perturbations into the Judge Sensitivity
   Score. Specs that change the patient rather than the encoding (see
   :mod:`rised.perturbations`) are evaluated and reported separately under
   ``covariate_shift_*``; calling that "reliability" would overstate what was
   tested.

2. It does not report perfect stability when nothing was perturbed. With an
   empty perturbation set every statistic is ``None`` and
   ``details["status"]`` is ``"not_evaluated"``. A vacuous 0.0 is
   indistinguishable from a genuinely stable model in every downstream report.

Rank correlation is reported per perturbation together with its **minimum**.
The documented R2 criterion is rho >= 0.95 for *every* perturbation, which the
mean cannot express: a set of five perturbations at rho = 1.00 and one at
rho = 0.75 has a mean of 0.958 and a minimum of 0.75.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from rised.metrics import decision_flip_rate, judge_sensitivity_score, rank_correlation
from rised.perturbations import (
    COVARIATE_SHIFT,
    SEMANTICS_PRESERVING,
    FeatureSchema,
    apply_perturbation,
    perturbation_semantics,
)
from rised.results import ReliabilityResult


def evaluate_reliability(
    model,
    X,
    perturbation_specs: Optional[List[Dict[str, Any]]] = None,
    feature_names: Optional[List[str]] = None,
    n_bootstrap: int = 0,
    random_state: Optional[int] = None,
    tau_ref: float = 0.5,
    schema: Optional[FeatureSchema] = None,
    groups=None,
) -> ReliabilityResult:
    """
    Evaluate the Reliability dimension (measurement layer; no thresholds).

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Original (unperturbed) feature matrix.
    perturbation_specs : list of dict, optional
        Each dict specifies one perturbation: ``{"type": str, ...}``. Specs are
        partitioned into semantics-preserving and covariate-shift families by
        :func:`rised.perturbations.perturbation_semantics`. If None or empty,
        every statistic is ``None`` and nothing is claimed.
    feature_names : list of str, optional
        Feature names corresponding to columns of X. Used for schema inference
        messages and stored in details.
    n_bootstrap : int
        Bootstrap replicates for the JSS confidence interval. 0 disables.
    random_state : int, optional
        Seed for the bootstrap RNG.
    tau_ref : float
        Decision threshold used for flip rates and JSS. Must match the
        threshold used elsewhere in the report; see
        :func:`rised.sensitivity.suggest_tau_ref`.
    schema : FeatureSchema, optional
        Column semantic types. Inferred conservatively from ``X`` when omitted,
        so that binary and categorical columns never receive continuous noise.
    groups : array-like, optional
        Cluster identifier per row for clustered resampling of the JSS
        interval. Default ``None`` is row-level resampling.

    Returns
    -------
    ReliabilityResult
    """
    X_arr = np.asarray(X, dtype=float)
    baseline = model.predict_proba(X_arr)[:, 1]

    if schema is None:
        schema = FeatureSchema.infer(X_arr, names=feature_names)

    base_details: Dict[str, Any] = {
        "feature_names": feature_names,
        "feature_schema": list(schema.types),
        "feature_schema_summary": schema.summary(),
        "reference_threshold": tau_ref,
    }

    if not perturbation_specs:
        return ReliabilityResult(
            judge_sensitivity_score=None,
            perturbation_flip_rate=None,
            rank_correlation_mean=None,
            rank_correlation_min=None,
            details={
                **base_details,
                "status": "not_evaluated",
                "reason": (
                    "No perturbation specs supplied. Reliability is undefined "
                    "when nothing was perturbed; it is not 'perfect'."
                ),
            },
        )

    preserving_flip: Dict[str, float] = {}
    preserving_rho: Dict[str, float] = {}
    shift_flip: Dict[str, float] = {}
    shift_rho: Dict[str, float] = {}
    preserving_scores: List[np.ndarray] = []
    spec_classes: Dict[str, str] = {}

    for spec in perturbation_specs:
        label = spec.get("label", spec["type"])
        semantics = perturbation_semantics(spec)
        spec_classes[label] = semantics
        X_pert = apply_perturbation(X_arr, spec, schema=schema)
        scores = model.predict_proba(X_pert)[:, 1]
        flip = decision_flip_rate(baseline, scores, threshold=tau_ref)
        rho = rank_correlation(baseline, scores)
        if semantics == SEMANTICS_PRESERVING:
            preserving_flip[label] = flip
            preserving_rho[label] = rho
            preserving_scores.append(scores)
        else:
            shift_flip[label] = flip
            shift_rho[label] = rho

    details: Dict[str, Any] = {
        **base_details,
        "status": "evaluated",
        "per_perturbation_flip_rate": preserving_flip,
        "per_perturbation_rank_correlation": preserving_rho,
        "perturbation_classes": spec_classes,
        "n_semantics_preserving": len(preserving_flip),
        "n_covariate_shift": len(shift_flip),
        "covariate_shift_flip_rate": shift_flip,
        "covariate_shift_rank_correlation": shift_rho,
        "covariate_shift_mean_flip_rate": (
            float(np.mean(list(shift_flip.values()))) if shift_flip else None
        ),
        "covariate_shift_min_rank_correlation": (
            float(min(shift_rho.values())) if shift_rho else None
        ),
        "covariate_shift_note": (
            "Covariate-shift perturbations change the patient rather than the "
            "encoding and are excluded from JSS. They are reported here as a "
            "robustness diagnostic, not as reliability."
        ),
    }

    if not preserving_flip:
        return ReliabilityResult(
            judge_sensitivity_score=None,
            perturbation_flip_rate=None,
            rank_correlation_mean=None,
            rank_correlation_min=None,
            details={
                **details,
                "status": "not_evaluated",
                "reason": (
                    "All supplied perturbations are covariate shifts; no "
                    "semantics-preserving perturbation was available for JSS."
                ),
            },
        )

    jss = judge_sensitivity_score(baseline, preserving_scores, threshold=tau_ref)
    mean_flip = float(np.mean(list(preserving_flip.values())))
    mean_rho = float(np.mean(list(preserving_rho.values())))
    min_rho = float(min(preserving_rho.values()))
    worst_label = min(preserving_rho, key=preserving_rho.get)
    details["min_rank_correlation_perturbation"] = worst_label

    # ── BCa 95% CI for JSS ────────────────────────────────────────────────────
    jss_ci = None
    if n_bootstrap > 0:
        from rised.bootstrap_ci import ResamplingPlan, bca_interval

        rng = np.random.default_rng(random_state)
        n = len(X_arr)
        plan = ResamplingPlan(n, groups)
        details["resampling"] = plan.describe()
        baseline_arr = np.asarray(baseline)
        pert_arrs = [np.asarray(s) for s in preserving_scores]

        def _jss_on_idx(idx: np.ndarray) -> float:
            if idx.size == 0:
                return float("nan")
            return judge_sensitivity_score(
                baseline_arr[idx], [s[idx] for s in pert_arrs], threshold=tau_ref
            )

        jss_boot = np.array(
            [_jss_on_idx(plan.bootstrap_index(rng)) for _ in range(n_bootstrap)],
            dtype=float,
        )
        jss_jack = np.array(
            [_jss_on_idx(idx) for idx in plan.jackknife_index_iter()], dtype=float
        )
        jss_ci = bca_interval(jss, jss_boot, jss_jack, alpha=0.05)

    return ReliabilityResult(
        judge_sensitivity_score=jss,
        perturbation_flip_rate=mean_flip,
        rank_correlation_mean=mean_rho,
        rank_correlation_min=min_rho,
        jss_ci=jss_ci,
        details=details,
    )
