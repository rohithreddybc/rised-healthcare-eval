"""
Deployability dimension: operational feasibility for clinical end users.

Timing (formerly "inference latency")
------------------------------------
The original D1 timed ``model.predict_proba(X)`` on the whole cohort and divided
by ``n``. That is **batch throughput**, not per-request latency: it amortises
away model load, per-call overhead and the fixed cost that a single clinical
request actually pays, and it improves as the batch gets larger. The fields are
now named for what they measure — :attr:`batch_scoring_time_ms` and
:attr:`amortised_time_per_row_ms` — and a genuine single-request measurement,
:attr:`single_row_latency_ms`, is taken separately by scoring one row at a time.

Explanation concentration (formerly "explanation faithfulness")
---------------------------------------------------------------
The original D2 derived the global feature ranking and the local rankings from
the *same* 50 rows, so the reference was the column mean of exactly the rows
being scored against it. Two consequences were verified: the statistic is
identically 1.0 whenever ``d <= 3`` (40/40 runs, including with labels
independent of the features), and its permutation null sits above the naive
``3/d`` — 0.394 rather than 0.300 at ``d = 10``.

Two changes follow. The global reference is now computed from a **disjoint**
sample, and the statistic is no longer called faithfulness: it measures how far
local attributions concentrate on the cohort's globally dominant features, which
is agreement between two rankings, not fidelity of an explanation to the model.
When ``d <= top_k`` the statistic is structurally 1.0 and is reported as
``None`` with a reason rather than as a perfect score.
"""

from __future__ import annotations

import time
import warnings
from typing import List, Optional

import numpy as np

from rised.results import DeployabilityResult


def explanation_chance_level(n_features: int, top_k: int = 3) -> float:
    """Chance level for the top-k agreement statistics under a random ranking."""
    if n_features <= 0:
        raise ValueError("n_features must be positive.")
    return float(min(1.0, top_k / n_features))


def evaluate_deployability(
    model,
    X,
    feature_names: Optional[List[str]] = None,
    n_latency_trials: int = 100,
    n_shap_samples: int = 50,
    n_single_row_trials: int = 20,
    top_k: int = 3,
) -> DeployabilityResult:
    """
    Evaluate the Deployability dimension (measurement layer; no thresholds).

    Measures:

    1. ``batch_scoring_time_ms`` — wall-clock time for one
       ``model.predict_proba(X)`` over the whole cohort, averaged over
       ``n_latency_trials`` calls, plus ``amortised_time_per_row_ms``
       (that time divided by ``n``). Neither is a per-request latency.
    2. ``single_row_latency_ms`` — median wall-clock time for
       ``model.predict_proba(X[i:i+1])``, the quantity a single clinical
       request actually pays.
    3. ``local_global_topk_agreement`` — fraction of scored patients whose
       locally most important feature is among the top-k features of a global
       ranking computed on a **disjoint** reference sample.
    4. ``global_top1_in_local_topk`` — fraction of scored patients whose local
       top-k contains the globally most important feature.

    SHAP is attempted via ``LinearExplainer`` for linear models and
    ``TreeExplainer`` otherwise. If SHAP is unavailable, or the cohort is too
    small to split into disjoint reference and scored samples, or
    ``d <= top_k``, the agreement metrics are ``None`` and ``details`` records
    the reason.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    feature_names : list of str, optional
        Feature names for reporting top features by name.
    n_latency_trials : int
        Repeated whole-cohort predict_proba calls for the batch timing.
    n_shap_samples : int
        Target size of each of the two disjoint SHAP samples.
    n_single_row_trials : int
        Number of single-row predict_proba calls for the per-request timing.
        Set to 0 to skip.
    top_k : int
        Size of the top-k sets compared. Default 3.

    Returns
    -------
    DeployabilityResult
    """
    X_arr = np.asarray(X, dtype=float)
    n_rows, n_features = X_arr.shape

    # ── 1. Batch scoring time (NOT per-request latency) ───────────────────────
    trial_times: List[float] = []
    for _ in range(n_latency_trials):
        t0 = time.perf_counter()
        model.predict_proba(X_arr)
        trial_times.append((time.perf_counter() - t0) * 1000.0)
    batch_time_ms = float(np.mean(trial_times))
    batch_time_std_ms = float(np.std(trial_times))
    amortised_ms = batch_time_ms / n_rows if n_rows else float("nan")

    # ── 2. Single-row latency (the per-request quantity) ──────────────────────
    single_row_latency_ms: Optional[float] = None
    if n_single_row_trials > 0 and n_rows > 0:
        single_times: List[float] = []
        for t in range(n_single_row_trials):
            row = X_arr[t % n_rows: t % n_rows + 1]
            t0 = time.perf_counter()
            model.predict_proba(row)
            single_times.append((time.perf_counter() - t0) * 1000.0)
        single_row_latency_ms = float(np.median(single_times))

    # ── 3-4. Explanation concentration, disjoint reference ────────────────────
    local_global_topk_agreement: Optional[float] = None
    global_top1_in_local_topk: Optional[float] = None
    shap_details: dict = {
        "top_k": top_k,
        "explanation_chance_level": explanation_chance_level(n_features, top_k)
        if n_features
        else None,
        "explanation_reference": "disjoint sample",
        "explanation_null_note": (
            "Under the previous design (global reference taken from the same "
            "rows being scored) the permutation null exceeded the analytic k/d "
            "-- 0.394 vs 0.300 at d=10. The disjoint reference removes that "
            "specific circularity; the null under the disjoint design has not "
            "been re-measured here, so compare against k/d with that caveat."
        ),
    }

    n_ref = min(n_shap_samples, n_rows // 2)
    n_score = min(n_shap_samples, n_rows - n_ref)

    if n_features <= top_k:
        shap_details["explanation_metrics_undefined_reason"] = (
            f"d={n_features} <= top_k={top_k}: the global top-{top_k} set is the "
            "entire feature set, so both agreement statistics are identically "
            "1.0 by construction and carry no information. Reported as None "
            "rather than as a perfect score."
        )
    elif n_ref < 2 or n_score < 2:
        shap_details["explanation_metrics_undefined_reason"] = (
            f"cohort of {n_rows} rows cannot be split into two disjoint samples "
            f"of at least 2 rows (reference={n_ref}, scored={n_score})."
        )
    else:
        try:
            import shap  # optional heavy dependency

            X_ref = X_arr[:n_ref]
            X_score = X_arr[n_ref: n_ref + n_score]

            if hasattr(model, "coef_"):
                explainer = shap.LinearExplainer(model, X_ref)
            else:
                # XGBoost >=2.x serialises base_score with brackets (e.g. '[5E-1]')
                # which older SHAP versions cannot parse via TreeExplainer.
                # Workaround: temporarily patch builtins.float to strip brackets.
                import builtins as _builtins
                _orig_float = _builtins.float

                def _safe_float(x):
                    try:
                        return _orig_float(x)
                    except (ValueError, TypeError):
                        return _orig_float(str(x).strip("[]"))

                _builtins.float = _safe_float
                try:
                    explainer = shap.TreeExplainer(model)
                finally:
                    _builtins.float = _orig_float

            def _abs_sv(data: np.ndarray) -> np.ndarray:
                raw = explainer.shap_values(data)
                if isinstance(raw, list):
                    arr = np.abs(np.asarray(raw[-1], dtype=float))
                else:
                    arr = np.abs(np.asarray(raw, dtype=float))
                if arr.ndim == 3:
                    arr = arr[:, :, -1]
                return arr

            sv_ref = _abs_sv(X_ref)
            sv_score = _abs_sv(X_score)

            # Global ranking from the reference sample only.
            global_importance = sv_ref.mean(axis=0)
            global_order = np.argsort(global_importance)[::-1]
            global_topk = set(int(i) for i in global_order[:top_k])
            global_top1 = int(global_order[0])

            local_top1 = np.argmax(sv_score, axis=1)
            local_global_topk_agreement = float(
                np.mean([int(t) in global_topk for t in local_top1])
            )

            hits = []
            for i in range(len(sv_score)):
                local_topk = set(
                    int(j) for j in np.argsort(sv_score[i])[::-1][:top_k]
                )
                hits.append(global_top1 in local_topk)
            global_top1_in_local_topk = float(np.mean(hits))

            shap_details["n_reference_rows"] = int(n_ref)
            shap_details["n_scored_rows"] = int(n_score)
            if feature_names and len(feature_names) >= len(global_order):
                shap_details["global_top_features"] = [
                    feature_names[int(i)] for i in global_order[:5]
                ]
            else:
                shap_details["global_top_feature_indices"] = [
                    int(i) for i in global_order[:5]
                ]

        except Exception as exc:
            shap_details["shap_error"] = str(exc)
            warnings.warn(
                f"SHAP explanation evaluation failed: {exc}. "
                "Explanation agreement metrics will be None.",
                RuntimeWarning,
                stacklevel=2,
            )

    return DeployabilityResult(
        batch_scoring_time_ms=batch_time_ms,
        amortised_time_per_row_ms=amortised_ms,
        single_row_latency_ms=single_row_latency_ms,
        local_global_topk_agreement=local_global_topk_agreement,
        global_top1_in_local_topk=global_top1_in_local_topk,
        details={
            "n_latency_trials": n_latency_trials,
            "batch_scoring_time_std_ms": batch_time_std_ms,
            "n_single_row_trials": n_single_row_trials,
            "n_samples": int(n_rows),
            "n_features": int(n_features),
            "timing_note": (
                "batch_scoring_time_ms is whole-cohort throughput, not "
                "per-request latency; amortised_time_per_row_ms is that time "
                "divided by n and shrinks as the batch grows. Use "
                "single_row_latency_ms for a per-request figure."
            ),
            **shap_details,
        },
    )
