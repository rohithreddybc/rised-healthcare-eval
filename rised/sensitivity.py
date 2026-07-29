"""
Sensitivity dimension: behavioural stability under small perturbations to the
decision threshold, measured through threshold flip rates (TFR) and the
fraction of patients in the borderline decision zone.

What TFR is, and what it is not
-------------------------------
For a reference threshold ``tau_0`` and a candidate threshold ``tau``,

    TFR(tau, tau_0) = |F_n(tau) - F_n(tau_0)|

where ``F_n`` is the empirical CDF of the predicted scores. **TFR never reads
``y_true``.** It is a functional of the score distribution alone, so it is
invariant to any permutation of the labels and it is gameable: a constant
predictor attains TFR = 0 at every threshold while being useless (AUROC 0.5),
and a rank-preserving squeeze of an informative score into a narrow band around
``tau_0`` attains TFR = 1 while retaining full discrimination. TFR must
therefore be read alongside a discrimination metric, never on its own.

Threshold bands
---------------
Two bands are computed. The narrow band
``[0.30, 0.70]`` (:data:`NARROW_THRESHOLD_BAND`) is the **primary** report: it
covers the range of operating points that a deployment might plausibly adopt.
The wide band ``[0.10, 0.90]`` (:data:`WIDE_THRESHOLD_BAND`) is retained as a
secondary, more adversarial sweep. A max TFR quoted over the wide band is not
comparable to one quoted over the narrow band; the band is always recorded in
``details["threshold_band"]``.

Reference threshold
-------------------
``tau_ref`` defaults to 0.5 but is a *choice*, not a property of the model. Use
:func:`suggest_tau_ref` to derive prevalence-matched or Youden-J thresholds for
a cohort. Nothing in this module applies a derived threshold automatically.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from rised.results import SensitivityResult

#: Primary threshold band: plausible deployment operating points.
NARROW_THRESHOLD_BAND = np.linspace(0.30, 0.70, 9)

#: Secondary (wide) threshold band retained from earlier releases.
WIDE_THRESHOLD_BAND = np.linspace(0.10, 0.90, 17)

#: Default reference threshold. A convention, not a validated operating point.
DEFAULT_TAU_REF = 0.5


def prevalence_matched_threshold(scores, prevalence: float) -> float:
    """Threshold whose positive-call rate matches ``prevalence``.

    Returns the ``(1 - prevalence)`` quantile of the score distribution, i.e.
    the threshold at which the model flags the same fraction of the cohort as
    the observed outcome prevalence.
    """
    s = np.asarray(scores, dtype=float)
    if not 0.0 < prevalence < 1.0:
        raise ValueError(f"prevalence must be in (0, 1); got {prevalence!r}.")
    return float(np.quantile(s, 1.0 - prevalence))


def youden_j_threshold(y_true, scores) -> float:
    """Threshold maximising Youden's J = sensitivity + specificity - 1."""
    from sklearn.metrics import roc_curve

    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if len(np.unique(y)) < 2:
        raise ValueError(
            "youden_j_threshold requires both classes to be present in y_true."
        )
    fpr, tpr, thr = roc_curve(y, s)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def suggest_tau_ref(scores, y_true=None) -> Dict[str, Optional[float]]:
    """Derive candidate per-cohort reference thresholds. Applies nothing.

    Returns a dict with the keys ``prevalence_matched``, ``youden_j`` and
    ``default``. The caller must pick one and pass it explicitly as ``tau_ref``;
    RISED never substitutes a derived threshold for the default on its own,
    because the operating point is an institutional decision that changes what
    every downstream flip-rate means.

    Parameters
    ----------
    scores : array-like
        Predicted probabilities for the cohort.
    y_true : array-like, optional
        Labels. Required for the prevalence-matched and Youden-J candidates;
        when omitted those entries are ``None``.
    """
    s = np.asarray(scores, dtype=float)
    out: Dict[str, Optional[float]] = {
        "default": float(DEFAULT_TAU_REF),
        "prevalence_matched": None,
        "youden_j": None,
    }
    if y_true is None:
        return out
    y = np.asarray(y_true)
    prevalence = float(np.mean(y))
    if 0.0 < prevalence < 1.0:
        out["prevalence_matched"] = prevalence_matched_threshold(s, prevalence)
        out["youden_j"] = youden_j_threshold(y, s)
    return out


def _flip_rates(scores: np.ndarray, band, tau_ref: float) -> Dict[float, float]:
    ref_decisions = scores >= tau_ref
    rates: Dict[float, float] = {}
    for tau in np.asarray(band, dtype=float):
        tau_f = float(round(float(tau), 8))
        rates[tau_f] = float(np.mean(ref_decisions != (scores >= tau_f)))
    return rates


def evaluate_sensitivity(
    model,
    X,
    y_true=None,
    threshold_range: Optional[np.ndarray] = None,
    tau_ref: float = DEFAULT_TAU_REF,
    boundary_delta: float = 0.05,
    n_bootstrap: int = 0,
    random_state: Optional[int] = None,
    wide_threshold_range: Optional[np.ndarray] = None,
    groups=None,
) -> SensitivityResult:
    """
    Evaluate the Sensitivity dimension (measurement layer; no thresholds).

    For each threshold in the primary band, computes the fraction of patients
    whose binary decision differs from the decision at ``tau_ref`` (Threshold
    Flip Rate), and the fraction of patients with scores within
    ``boundary_delta`` of ``tau_ref``.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like, optional
        **Not used in any computation.** TFR is a functional of the empirical
        score CDF only (see module docstring). Accepted for API symmetry with
        the other ``evaluate_*`` functions and to allow ``suggest_tau_ref`` to
        be called on the same arguments; passing or omitting it cannot change
        any returned value.
    threshold_range : array-like, optional
        Primary threshold band. Defaults to :data:`NARROW_THRESHOLD_BAND`
        (``[0.30, 0.70]``).
    tau_ref : float
        Reference (operational) threshold. A choice, not a model property;
        see :func:`suggest_tau_ref`. Default 0.5.
    boundary_delta : float
        Half-width of the borderline zone around tau_ref.
    n_bootstrap : int
        Bootstrap replicates for the BCa interval on the primary-band max TFR.
    random_state : int, optional
        Seed for the bootstrap RNG.
    wide_threshold_range : array-like, optional
        Secondary band, reported in ``details``. Defaults to
        :data:`WIDE_THRESHOLD_BAND`. Pass an empty sequence to skip it.
    groups : array-like, optional
        Cluster identifier per row for clustered resampling. Default ``None``
        is row-level resampling.

    Returns
    -------
    SensitivityResult
    """
    X_arr = np.asarray(X, dtype=float)
    scores = model.predict_proba(X_arr)[:, 1]

    primary_band = (
        NARROW_THRESHOLD_BAND if threshold_range is None else np.asarray(threshold_range, dtype=float)
    )
    wide_band = (
        WIDE_THRESHOLD_BAND
        if wide_threshold_range is None
        else np.asarray(wide_threshold_range, dtype=float)
    )

    threshold_flip_rates = _flip_rates(scores, primary_band, tau_ref)
    wide_flip_rates = (
        _flip_rates(scores, wide_band, tau_ref) if len(wide_band) else {}
    )

    mean_flip = (
        float(np.mean(list(threshold_flip_rates.values())))
        if threshold_flip_rates
        else 0.0
    )
    rank_stability_score = 1.0 - mean_flip

    decision_boundary_width = float(
        np.mean(np.abs(scores - tau_ref) <= boundary_delta)
    )

    max_tfr_value = (
        float(max(threshold_flip_rates.values())) if threshold_flip_rates else 0.0
    )
    wide_max_tfr = (
        float(max(wide_flip_rates.values())) if wide_flip_rates else None
    )

    # ── BCa interval for the primary-band max TFR ─────────────────────────────
    max_tfr_ci = None
    resampling_info: Dict[str, object] = {}
    if n_bootstrap > 0 and threshold_flip_rates:
        from rised.bootstrap_ci import ResamplingPlan, bca_interval

        rng = np.random.default_rng(random_state)
        n = len(X_arr)
        plan = ResamplingPlan(n, groups)
        resampling_info = plan.describe()
        thresholds = list(threshold_flip_rates.keys())

        def _max_tfr_on_idx(idx: np.ndarray) -> float:
            sb = scores[idx]
            if sb.size == 0:
                return float("nan")
            ref_b = sb >= tau_ref
            return float(
                max(float(np.mean(ref_b != (sb >= t))) for t in thresholds)
            )

        max_tfr_boot = np.array(
            [_max_tfr_on_idx(plan.bootstrap_index(rng)) for _ in range(n_bootstrap)],
            dtype=float,
        )
        max_tfr_jack = np.array(
            [_max_tfr_on_idx(idx) for idx in plan.jackknife_index_iter()],
            dtype=float,
        )
        max_tfr_ci = bca_interval(
            max_tfr_value, max_tfr_boot, max_tfr_jack, alpha=0.05
        )

    return SensitivityResult(
        threshold_flip_rates=threshold_flip_rates,
        max_threshold_flip_rate=max_tfr_value,
        rank_stability_score=rank_stability_score,
        decision_boundary_width=decision_boundary_width,
        max_tfr_ci=max_tfr_ci,
        wide_band_flip_rates=wide_flip_rates,
        wide_band_max_tfr=wide_max_tfr,
        details={
            "reference_threshold": tau_ref,
            "boundary_delta": boundary_delta,
            "n_thresholds_evaluated": len(threshold_flip_rates),
            "threshold_band": (
                float(min(threshold_flip_rates)),
                float(max(threshold_flip_rates)),
            )
            if threshold_flip_rates
            else None,
            "threshold_band_name": (
                "narrow[0.30,0.70]" if threshold_range is None else "custom"
            ),
            "wide_threshold_band": (
                (float(min(wide_flip_rates)), float(max(wide_flip_rates)))
                if wide_flip_rates
                else None
            ),
            "uses_y_true": False,
            "resampling": resampling_info,
        },
    )
