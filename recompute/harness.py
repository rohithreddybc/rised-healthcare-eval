"""
Shared machinery for recomputing every cohort under both measurement pipelines.

For each cohort we hold the data, the split, the seed and the fitted model
completely fixed, and evaluate the *same* model twice:

  * ``run_old``  -- the vendored 0.1.0 package (``recompute/_vendor/rised_v010``,
    a byte-for-byte copy of ``rised/`` at commit 22b7929, the last commit before
    the F-series corrections, with only its internal import paths renamed).
  * ``run_new``  -- the corrected 0.2.0 package installed at ``rised/``.

Holding everything but the measurement code fixed is the point: any difference
between the two columns is attributable to the measurement correction and to
nothing else. In particular the *same* perturbation spec list is handed to both
versions -- 0.1.0 pools it all into the Judge Sensitivity Score and adds
continuous noise to every column regardless of type, 0.2.0 partitions it into
semantics-preserving and covariate-shift families and respects the column
schema. That divergence is the correction, not a change of input.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Vendored 0.1.0 lives beside us and is imported as `rised_v010`.
sys.path.insert(0, str(HERE / "_vendor"))

SEED = 42

#: Bootstrap replicates for published intervals. 1000 matches the 0.2.0 README.
N_BOOTSTRAP = 1000


# ── Cohort specification ─────────────────────────────────────────────────────
@dataclass
class Cohort:
    """Everything needed to evaluate one cohort under both pipelines."""

    name: str
    label: str
    #: Callable returning a dict with X_train/X_test/y_train/y_test/demo_test/
    #: feature_names and optionally groups_test, need_column, notes.
    loader: Callable[[], Dict[str, Any]]
    #: Perturbation specs, handed unchanged to BOTH versions.
    perturbation_specs: List[Dict[str, Any]] = field(default_factory=list)
    #: Reference decision threshold for the 0.2.0 run. None -> 0.5.
    tau_ref: Optional[float] = None
    #: Use the clustered (group-level) bootstrap in the 0.2.0 run.
    clustered: bool = False
    #: Column of demo_test holding an independent clinical-need proxy.
    need_column: Optional[str] = None
    #: Free-text provenance / caveats carried into the report.
    notes: str = ""
    #: Set when the cohort cannot be evaluated offline.
    offline_blocked_reason: Optional[str] = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def _f(x) -> Optional[float]:
    """Coerce to a JSON-safe float, mapping non-finite to None."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _ci(x) -> Optional[List[Optional[float]]]:
    if x is None:
        return None
    try:
        return [_f(x[0]), _f(x[1])]
    except (TypeError, IndexError, KeyError):
        return None


def subgroup_sizes(demo: pd.DataFrame, cols: Optional[List[str]] = None) -> Dict[str, int]:
    """Row count for every level of every demographic column."""
    cols = cols if cols is not None else list(demo.columns)
    out: Dict[str, int] = {}
    for c in cols:
        vc = demo[c].astype(str).value_counts()
        for lvl, n in vc.items():
            out[f"{c}={lvl}"] = int(n)
    return out


def subgroup_label_counts(
    demo: pd.DataFrame, y, cols: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """Size and positive/negative counts for every subgroup level."""
    y_arr = np.asarray(y).astype(int)
    cols = cols if cols is not None else list(demo.columns)
    out: Dict[str, Dict[str, int]] = {}
    d = demo.reset_index(drop=True)
    for c in cols:
        s = d[c].astype(str)
        for lvl in pd.unique(s):
            mask = (s == lvl).values
            n = int(mask.sum())
            n_pos = int(y_arr[mask].sum())
            out[f"{c}={lvl}"] = {"n": n, "n_pos": n_pos, "n_neg": n - n_pos}
    return out


# ── The two pipelines ────────────────────────────────────────────────────────
def run_old(
    model,
    X_test,
    y_test,
    demo_test: pd.DataFrame,
    perturbation_specs: List[Dict[str, Any]],
    feature_names: Optional[List[str]],
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = SEED,
) -> Dict[str, Any]:
    """Evaluate under the vendored 0.1.0 pipeline (the pre-correction result)."""
    import rised_v010

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        # 0.1.0 warns about the y_true equity fallback on every call; that
        # fallback is exactly the behaviour we are documenting, so record it
        # rather than letting it flood the log.
        warnings.simplefilter("ignore")
        rep = rised_v010.evaluate_all(
            model, X_test, y_test, demo_test,
            perturbation_specs=perturbation_specs,
            feature_names=feature_names,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
    elapsed = time.perf_counter() - t0

    rel, inc, sen, eq, dep = (
        rep.reliability, rep.inclusivity, rep.sensitivity, rep.equity, rep.deployability
    )
    return {
        "version": rised_v010.__version__,
        "runtime_s": elapsed,
        # Reliability: 0.1.0 pooled every perturbation, including the age
        # rescalings later reclassified as covariate shift, and applied
        # continuous noise to binary/categorical columns.
        "jss": _f(rel.judge_sensitivity_score),
        "jss_ci": _ci(rel.jss_ci),
        "rank_correlation_mean": _f(rel.rank_correlation_mean),
        "per_perturbation_flip_rate": {
            k: _f(v) for k, v in rel.details.get("per_perturbation_flip_rate", {}).items()
        },
        "per_perturbation_rank_correlation": {
            k: _f(v)
            for k, v in rel.details.get("per_perturbation_rank_correlation", {}).items()
        },
        # Inclusivity: single pooled max-min across every level of every column,
        # with n<30 only *flagged* in the point estimate but *excluded* in the
        # bootstrap and jackknife.
        "auc_gap_pooled": _f(inc.auc_parity_gap),
        "auc_gap_ci": _ci(inc.auc_gap_ci),
        "subgroup_aucs": {k: _f(v) for k, v in inc.subgroup_aucs.items()},
        "small_group_flags": list(inc.details.get("small_group_flags", [])),
        "max_subgroup_ece": _f(
            max(inc.subgroup_calibration.values()) if inc.subgroup_calibration else None
        ),
        # Sensitivity: single wide band [0.10, 0.90], tau_ref fixed at 0.5.
        # 0.1.0 has no max_threshold_flip_rate field; the statistic is the max
        # over the swept band, which is how its withdrawn passed() computed it.
        "max_tfr": _f(
            max(sen.threshold_flip_rates.values())
            if sen.threshold_flip_rates else None
        ),
        "max_tfr_ci": _ci(sen.max_tfr_ci),
        "threshold_flip_rates": {str(k): _f(v) for k, v in sen.threshold_flip_rates.items()},
        # Equity: silently falls back to y_true as the need proxy.
        "equity_rho": _f(eq.need_prediction_correlation),
        "equity_need_source": eq.details.get("need_source"),
        # Deployability: whole-batch timing misreported as latency.
        "batch_scoring_time_ms": _f(getattr(dep, "mean_inference_latency_ms", None)),
        "amortised_time_per_row_ms": _f(getattr(dep, "mean_latency_per_patient_ms", None)),
        "single_row_latency_ms": None,
        "explanation_agreement": _f(getattr(dep, "explanation_faithfulness", None)),
    }


def run_new(
    model,
    X_test,
    y_test,
    demo_test: pd.DataFrame,
    perturbation_specs: List[Dict[str, Any]],
    feature_names: Optional[List[str]],
    tau_ref: float = 0.5,
    groups=None,
    demo_with_need: Optional[pd.DataFrame] = None,
    need_column: Optional[str] = None,
    n_bootstrap: int = N_BOOTSTRAP,
    random_state: int = SEED,
) -> Dict[str, Any]:
    """Evaluate under the corrected 0.2.0 pipeline.

    ``demo_test`` carries the demographic partitions only. The need proxy is
    handed to Equity separately via ``demo_with_need``/``need_column`` so it
    never becomes an Inclusivity partition -- a proxy such as a comorbidity
    count has dozens of levels and would otherwise manufacture a partition with
    an enormous, purely selective AUC range.
    """
    import rised

    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rep = rised.evaluate_all(
            model, X_test, y_test, demo_test,
            perturbation_specs=perturbation_specs,
            feature_names=feature_names,
            tau_ref=tau_ref,
            groups=groups,
            need_column=None,          # Equity handled explicitly below.
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        warn_msgs = sorted({str(w.message)[:300] for w in caught})
    elapsed = time.perf_counter() - t0

    # Equity, with the proxy attached to a copy of the demographic frame.
    eq_result = None
    eq_error = None
    if need_column is not None and demo_with_need is not None:
        from rised.equity import evaluate_equity

        try:
            eq_result = evaluate_equity(
                model, X_test, y_test, demo_with_need, need_column=need_column,
            )
        except ValueError as exc:
            eq_error = str(exc)

    rel, inc, sen, eq, dep = (
        rep.reliability, rep.inclusivity, rep.sensitivity, rep.equity, rep.deployability
    )
    out: Dict[str, Any] = {
        "version": rised.__version__,
        "runtime_s": elapsed,
        "tau_ref": _f(tau_ref),
        "clustered_resampling": bool(rep.metadata.get("clustered_resampling")),
        "warnings": warn_msgs,
        # Reliability: semantics-preserving perturbations only.
        "jss": _f(rel.judge_sensitivity_score),
        "jss_ci": _ci(rel.jss_ci),
        "rank_correlation_min": _f(rel.rank_correlation_min),
        "rank_correlation_mean": _f(rel.rank_correlation_mean),
        "per_perturbation_flip_rate": {
            k: _f(v) for k, v in rel.details.get("per_perturbation_flip_rate", {}).items()
        },
        "per_perturbation_rank_correlation": {
            k: _f(v)
            for k, v in rel.details.get("per_perturbation_rank_correlation", {}).items()
        },
        "perturbation_classes": dict(rel.details.get("perturbation_classes", {})),
        "n_semantics_preserving": rel.details.get("n_semantics_preserving"),
        "n_covariate_shift": rel.details.get("n_covariate_shift"),
        "covariate_shift_flip_rate": {
            k: _f(v) for k, v in rel.details.get("covariate_shift_flip_rate", {}).items()
        },
        "reliability_status": rel.details.get("status"),
        # Inclusivity: max per-partition gap is the headline; the pooled
        # cross-partition max-min is retained only as a diagnostic.
        "auc_gap_per_partition_max": _f(inc.auc_parity_gap),
        "auc_gap_ci": _ci(inc.auc_gap_ci),
        "auc_gap_pooled_diagnostic": _f(inc.pooled_auc_gap_diagnostic),
        "per_partition_auc_gaps": {k: _f(v) for k, v in inc.per_partition_auc_gaps.items()},
        "worst_partition": inc.worst_partition,
        "subgroup_aucs": {k: _f(v) for k, v in inc.subgroup_aucs.items()},
        "excluded_subgroups": dict(inc.excluded_subgroups),
        "max_subgroup_ece": _f(
            max(inc.subgroup_calibration.values()) if inc.subgroup_calibration else None
        ),
        "resampling": inc.details.get("resampling", {}),
        # Sensitivity: narrow band primary, wide band secondary.
        "max_tfr_narrow": _f(sen.max_threshold_flip_rate),
        "max_tfr_narrow_ci": _ci(sen.max_tfr_ci),
        "max_tfr_wide": _f(sen.wide_band_max_tfr),
        "threshold_band": sen.details.get("threshold_band"),
        "threshold_flip_rates_narrow": {
            str(k): _f(v) for k, v in sen.threshold_flip_rates.items()
        },
        "threshold_flip_rates_wide": {
            str(k): _f(v) for k, v in sen.wide_band_flip_rates.items()
        },
        # Deployability: correctly named timings.
        "batch_scoring_time_ms": _f(dep.batch_scoring_time_ms),
        "amortised_time_per_row_ms": _f(dep.amortised_time_per_row_ms),
        "single_row_latency_ms": _f(dep.single_row_latency_ms),
        "explanation_agreement": _f(dep.local_global_topk_agreement),
        "explanation_chance_level": _f(dep.details.get("explanation_chance_level")),
        "explanation_undefined_reason": dep.details.get(
            "explanation_metrics_undefined_reason"
        ),
    }
    assert eq is None, "evaluate_all was called with need_column=None"
    out["equity_skipped_in_evaluate_all"] = rep.metadata.get("equity_skipped_reason")
    if eq_result is None:
        out["equity_evaluated"] = False
        out["equity_rho"] = None
        out["equity_skipped_reason"] = eq_error or (
            "No independent need proxy supplied for this cohort."
        )
    else:
        out["equity_evaluated"] = True
        out["equity_rho"] = _f(eq_result.need_prediction_correlation)
        out["equity_need_source"] = eq_result.details.get("need_source")
        out["equity_ceiling"] = _f(eq_result.attainable_rho_ceiling)
        out["equity_proxy_prevalence"] = _f(eq_result.proxy_prevalence)
        out["equity_proxy_levels"] = eq_result.details.get("proxy_levels")
        out["equity_fraction_of_ceiling"] = _f(
            eq_result.details.get("correlation_as_fraction_of_ceiling")
        )
    return out


# ── JSON writing ─────────────────────────────────────────────────────────────
class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return v if np.isfinite(v) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, cls=_Enc)
