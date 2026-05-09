"""
Deployability dimension: operational feasibility for clinical end users,
encompassing inference latency, SHAP explanation faithfulness, and
top-feature stability across the patient cohort.
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional

import numpy as np

from rised.results import DeployabilityResult


def evaluate_deployability(
    model,
    X,
    feature_names: Optional[List[str]] = None,
    n_latency_trials: int = 100,
    n_shap_samples: int = 50,
) -> DeployabilityResult:
    """
    Evaluate the Deployability dimension.

    Measures three operational properties:
    1. Mean inference latency across ``n_latency_trials`` calls to
       ``model.predict_proba(X)``.
    2. SHAP explanation faithfulness: fraction of patients whose locally
       most important feature (by |SHAP|) is among the globally top-3 features.
    3. Top feature stability: fraction of patients for whom the globally
       most important feature appears in their local top-3.

    SHAP is attempted via ``LinearExplainer`` for linear models and
    ``TreeExplainer`` for tree-based models. If neither applies, SHAP
    metrics are set to None and the error is recorded in ``details``.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    feature_names : list of str, optional
        Feature names for reporting top features by name.
    n_latency_trials : int
        Number of repeated predict_proba calls for latency measurement.
    n_shap_samples : int
        Number of samples used for SHAP computation (capped at len(X)).

    Returns
    -------
    DeployabilityResult
    """
    X_arr = np.asarray(X, dtype=float)

    # ── 1. Inference latency ──────────────────────────────────────────────────
    trial_times: List[float] = []
    for _ in range(n_latency_trials):
        t0 = time.perf_counter()
        model.predict_proba(X_arr)
        trial_times.append((time.perf_counter() - t0) * 1000.0)
    mean_latency_ms = float(np.mean(trial_times))
    latency_std_ms = float(np.std(trial_times))
    mean_latency_per_patient_ms = mean_latency_ms / X_arr.shape[0]

    # ── 2. SHAP explanation faithfulness ─────────────────────────────────────
    explanation_faithfulness: Optional[float] = None
    top_feature_stability: Optional[float] = None
    shap_details: dict = {}

    try:
        import shap  # optional heavy dependency

        n_bg = min(n_shap_samples, len(X_arr))
        X_bg = X_arr[:n_bg]

        if hasattr(model, "coef_"):
            # sklearn linear models (LogisticRegression, LinearSVC, etc.)
            explainer = shap.LinearExplainer(model, X_bg)
        elif hasattr(model, "estimators_") or hasattr(model, "tree_"):
            # sklearn ensemble / single-tree models
            explainer = shap.TreeExplainer(model)
        elif "xgb" in type(model).__module__ or "lgbm" in type(model).__module__:
            explainer = shap.TreeExplainer(model)
        else:
            raise ValueError(
                f"No fast SHAP explainer available for {type(model).__name__}. "
                "Add explicit support or use shap.Explainer with a smaller background."
            )

        shap_raw = explainer.shap_values(X_bg)

        # Normalise to shape (n_samples, n_features) for the positive class
        if isinstance(shap_raw, list):
            sv = np.abs(np.asarray(shap_raw[-1], dtype=float))
        else:
            sv = np.abs(np.asarray(shap_raw, dtype=float))
        if sv.ndim == 3:
            sv = sv[:, :, -1]

        # Global importance: mean |SHAP| per feature
        global_importance = sv.mean(axis=0)
        global_top_idx = np.argsort(global_importance)[::-1]
        global_top3: set = set(global_top_idx[:3].tolist())
        global_top1: int = int(global_top_idx[0])

        # Explanation faithfulness: fraction where local top-1 ∈ global top-3
        local_top1 = np.argmax(sv, axis=1)
        explanation_faithfulness = float(
            np.mean([int(t) in global_top3 for t in local_top1])
        )

        # Top feature stability: fraction where global top-1 ∈ local top-3
        stability_hits = []
        for i in range(len(sv)):
            local_top3 = set(np.argsort(sv[i])[::-1][:3].tolist())
            stability_hits.append(global_top1 in local_top3)
        top_feature_stability = float(np.mean(stability_hits))

        if feature_names and len(feature_names) >= len(global_top_idx):
            shap_details["global_top_features"] = [
                feature_names[int(i)] for i in global_top_idx[:5]
            ]
        else:
            shap_details["global_top_feature_indices"] = [
                int(i) for i in global_top_idx[:5]
            ]

    except Exception as exc:
        shap_details["shap_error"] = str(exc)
        warnings.warn(
            f"SHAP explanation evaluation failed: {exc}. "
            "explanation_faithfulness and top_feature_stability will be None.",
            RuntimeWarning,
            stacklevel=2,
        )

    return DeployabilityResult(
        mean_inference_latency_ms=mean_latency_ms,
        mean_latency_per_patient_ms=mean_latency_per_patient_ms,
        explanation_faithfulness=explanation_faithfulness,
        top_feature_stability=top_feature_stability,
        details={
            "n_latency_trials": n_latency_trials,
            "latency_std_ms": latency_std_ms,
            "n_samples": X_arr.shape[0],
            **shap_details,
        },
    )
