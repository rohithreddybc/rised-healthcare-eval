"""
Recompute one cohort under both the 0.1.0 and the corrected 0.2.0 pipeline.

    python -m recompute.run_cohort <cohort> [--bootstrap N]

Writes ``recompute/results/<cohort>.json``. Run every cohort with
``python -m recompute.run_all``.

Everything is seeded: data preparation, split, model fit, bootstrap RNG and the
permutation null all use random_state=42.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE / "_vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompute import harness, specs as spec_mod  # noqa: E402
from recompute.cohorts import LOADERS  # noqa: E402
from recompute.null_reference import (  # noqa: E402
    P2_REFERENCE_10x500,
    cohort_null_reference,
)

SEED = 42

COHORT_LABELS = {
    "synthetic": "Synthetic baseline (Synthea-inspired, 10k)",
    "uci_heart": "UCI Heart Disease (Cleveland)",
    "diabetes130": "UCI Diabetes 130-US Hospitals (grouped on patient_nbr)",
    "nhis2024": "NCHS NHIS 2024 (Sample Adult, CHD/MI)",
    "nhis2023": "NCHS NHIS 2023 (Sample Adult, diabetes)",
    "nhanes2123": "NCHS NHANES 2021-2023 (diabetes, with lab HbA1c)",
    "brfss2024": "CDC BRFSS 2024 (CHD/MI)",
    "adult_income": "Cross-domain: UCI Adult Income",
    "acs_income": "Cross-domain: Folktables ACS-Income (CA 2018)",
    "german_credit": "Cross-domain: Statlog German Credit",
}


def _discrimination(model, X, y) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    s = model.predict_proba(np.asarray(X, dtype=float))[:, 1]
    y = np.asarray(y).astype(int)
    return {
        "auroc": float(roc_auc_score(y, s)),
        "brier": float(np.mean((s - y) ** 2)),
        "prevalence": float(y.mean()),
        "scores": s,
    }


def run(cohort: str, n_bootstrap: int = harness.N_BOOTSTRAP) -> Dict[str, Any]:
    t_start = time.perf_counter()
    payload: Dict[str, Any] = {
        "cohort": cohort,
        "label": COHORT_LABELS.get(cohort, cohort),
        "seed": SEED,
        "n_bootstrap": int(n_bootstrap),
    }

    t0 = time.perf_counter()
    bundle = LOADERS[cohort]()
    payload["load_runtime_s"] = time.perf_counter() - t0

    X_te = np.asarray(bundle["X_test"], dtype=float)
    y_te = bundle["y_test"]
    demo_te = bundle["demo_test"]
    demo_need = bundle["demo_test_with_need"]
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    groups_te = bundle.get("groups_test")
    need_col = bundle.get("need_column")
    sub_cols = bundle["subgroup_columns"]

    disc = _discrimination(model, X_te, y_te)
    scores = disc.pop("scores")
    payload["cohort_stats"] = {
        "n_total": bundle["n_total"],
        "n_train": int(np.asarray(bundle["X_train"]).shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_features": int(X_te.shape[1]),
        "feature_names": feature_names,
        "subgroup_columns": sub_cols,
        **disc,
    }
    for k in ("n_patients", "n_test_patients", "group_split_test_row_leakage"):
        if k in bundle:
            payload["cohort_stats"][k] = bundle[k]

    payload["subgroup_counts"] = harness.subgroup_label_counts(
        demo_te, y_te, sub_cols)

    perturbation_specs = spec_mod.specs_for(cohort, bundle)
    payload["perturbation_specs"] = perturbation_specs
    payload["proxy_validity"] = spec_mod.PROXY_VALIDITY.get(cohort)

    # ── 0.1.0 ────────────────────────────────────────────────────────────────
    print(f"[{cohort}] running 0.1.0 pipeline ...", flush=True)
    payload["old"] = harness.run_old(
        model, X_te, y_te, demo_te, perturbation_specs, feature_names,
        n_bootstrap=n_bootstrap, random_state=SEED,
    )

    # ── 0.2.0 ────────────────────────────────────────────────────────────────
    # tau_ref stays at 0.5 for the headline so the before/after difference is
    # attributable to the measurement change alone. The prevalence-matched
    # alternative is reported separately below: at low prevalence tau=0.5 is not
    # an operating point any deployment would choose, and the narrow band around
    # it can be quiet for trivial reasons.
    print(f"[{cohort}] running 0.2.0 pipeline ...", flush=True)
    payload["new"] = harness.run_new(
        model, X_te, y_te, demo_te,
        perturbation_specs, feature_names,
        tau_ref=0.5, groups=groups_te,
        demo_with_need=demo_need, need_column=need_col,
        n_bootstrap=n_bootstrap, random_state=SEED,
    )
    payload["new"]["subgroup_columns_used"] = list(demo_te.columns)

    # ── tau_ref sensitivity (point estimates only, no interval) ──────────────
    from rised.sensitivity import evaluate_sensitivity, suggest_tau_ref

    cand = suggest_tau_ref(scores, y_te)
    tau_alt = cand.get("prevalence_matched")
    payload["tau_ref_candidates"] = {k: harness._f(v) for k, v in cand.items()}
    if tau_alt is not None and np.isfinite(tau_alt):
        sen_alt = evaluate_sensitivity(model, X_te, y_te, tau_ref=float(tau_alt))
        payload["tau_ref_alt"] = {
            "tau_ref": float(tau_alt),
            "basis": "prevalence_matched",
            "max_tfr_narrow": harness._f(sen_alt.max_threshold_flip_rate),
            "max_tfr_wide": harness._f(sen_alt.wide_band_max_tfr),
            "note": (
                "Point estimates only (no bootstrap). Reported because tau=0.5 "
                "is not a plausible operating point at this prevalence."
            ),
        }

    # ── Cohort-specific equality null for the Inclusivity gap ────────────────
    print(f"[{cohort}] simulating the cohort-specific null ...", flush=True)
    t0 = time.perf_counter()
    payload["null_reference"] = cohort_null_reference(
        y_te, scores, demo_te, subgroup_columns=sub_cols,
        min_subgroup_n=30, n_reps=2000, random_state=SEED,
        observed_gap=payload["new"].get("auc_gap_per_partition_max"),
    )
    payload["null_reference"]["runtime_s"] = time.perf_counter() - t0
    # The same null evaluated against the OLD pooled statistic, so the pooled
    # number can be judged on its own (much larger) null.
    payload["null_reference_pooled"] = _pooled_null(
        y_te, scores, demo_te, sub_cols,
        payload["old"].get("auc_gap_pooled"),
    )
    payload["p2_generic_reference"] = P2_REFERENCE_10x500

    # ── Diabetes 130 only: the published leaky row-level split ───────────────
    if cohort == "diabetes130" and "row_split" in bundle:
        print(f"[{cohort}] running 0.1.0 on the original leaky row split ...",
              flush=True)
        rs = bundle["row_split"]
        rdisc = _discrimination(rs["model"], rs["X_test"], rs["y_test"])
        rdisc.pop("scores")
        payload["row_split_reference"] = {
            "purpose": (
                "Traceability for the published 0.1.0 figures, which were "
                "produced on a row-level split that puts the same patient in "
                "train and test."
            ),
            "row_leakage_fraction": rs.get("row_leakage_fraction"),
            "n_test": int(np.asarray(rs["X_test"]).shape[0]),
            **rdisc,
            "old": harness.run_old(
                rs["model"], rs["X_test"], rs["y_test"], rs["demo_test"],
                perturbation_specs, feature_names,
                n_bootstrap=n_bootstrap, random_state=SEED,
            ),
        }

    payload["total_runtime_s"] = time.perf_counter() - t_start
    return payload


def _pooled_null(y, scores, demo, cols, observed_pooled):
    """Equality null for the OLD pooled cross-partition max-min statistic."""
    from scipy.stats import rankdata

    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=float)
    d = demo.reset_index(drop=True)
    codes = {}
    for c in cols:
        _, inv = np.unique(np.asarray(d[c].astype(str)), return_inverse=True)
        codes[c] = inv.astype(np.int32)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(SEED)

    def _fast_auc(yy, ss):
        n1 = int(yy.sum())
        n0 = len(yy) - n1
        if n1 == 0 or n0 == 0:
            return np.nan
        r = rankdata(ss)
        return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))

    vals = np.full(2000, np.nan)
    for r in range(2000):
        aucs = []
        for c, cd in codes.items():
            new = np.empty_like(cd)
            new[pos] = cd[rng.permutation(pos)]
            new[neg] = cd[rng.permutation(neg)]
            for lvl in np.unique(new):
                m = new == lvl
                n = int(m.sum())
                yg = y[m]
                np_ = int(yg.sum())
                # 0.1.0 pooled EVERY level with estimable AUC, including n<30.
                if np_ < 2 or (n - np_) < 2:
                    continue
                a = _fast_auc(yg, s[m])
                if not np.isnan(a):
                    aucs.append(a)
        if len(aucs) >= 2:
            vals[r] = max(aucs) - min(aucs)
    v = vals[~np.isnan(vals)]
    if len(v) == 0:
        return {"null_estimable": False}
    out = {
        "null_design": "same stratified permutation, pooled across all columns, "
                       "no n>=30 exclusion (0.1.0 point-estimate rule)",
        "null_estimable": True,
        "n_valid_reps": int(len(v)),
        "null_mean_gap": float(np.mean(v)),
        "null_p95_gap": float(np.percentile(v, 95)),
    }
    if observed_pooled is not None and np.isfinite(observed_pooled):
        out["observed_gap"] = float(observed_pooled)
        out["p_value_vs_null"] = float(
            (int(np.sum(v >= observed_pooled)) + 1) / (len(v) + 1))
        out["excess_over_null_mean"] = float(observed_pooled - np.mean(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort", choices=sorted(LOADERS))
    ap.add_argument("--bootstrap", type=int, default=harness.N_BOOTSTRAP)
    args = ap.parse_args()

    out_path = harness.RESULTS_DIR / f"{args.cohort}.json"
    try:
        payload = run(args.cohort, n_bootstrap=args.bootstrap)
        payload["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        payload = {
            "cohort": args.cohort,
            "label": COHORT_LABELS.get(args.cohort, args.cohort),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
        harness.write_json(out_path, payload)
        print(f"[{args.cohort}] FAILED: {payload['error']}", file=sys.stderr)
        return 1

    harness.write_json(out_path, payload)
    print(f"[{args.cohort}] wrote {out_path} "
          f"({payload['total_runtime_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
