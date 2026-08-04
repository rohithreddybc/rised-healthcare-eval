"""
Joint-permutation null, with a sweep over the subgroup-inclusion rule.

Two defects in the published null are addressed here.

1. **Independent per-column permutation.** The published null draws a fresh
   permutation for each demographic column, so the per-column gaps it generates
   are independent. In the real cohorts age, sex, race, insurance and income are
   strongly associated. The test statistic is a maximum over columns, and the
   maximum of independent components is stochastically larger than the maximum
   of positively dependent components with the same margins. The published null
   is therefore too wide and its p-values too large -- biased toward the
   manuscript's own negative conclusion. ``scheme="joint"`` permutes row indices
   once per replicate within outcome class and carries every demographic column
   on those permuted rows, which preserves the joint contingency table.

2. **An unexamined ``m_min = 30``.** The null's width is driven by its noisiest
   estimable levels. At NHIS 2024's prevalence of 0.075 a level of n=30 carries
   about 2 events, and an AUROC on 2 events is nearly pure noise. Noise widens a
   max-min range, which again pushes p-values up. ``m_min`` is swept over
   {20, 30, 50, 100} together with the events-based rule
   (>= 10 in each outcome class) that the prognostic-model literature argues
   for.

Every inclusion rule is evaluated on the *same* permutation draws, so the
settings are exactly paired and the sweep costs one pass rather than five.

    python -m recompute.null_joint <cohort> [--reps 10000]

writes ``recompute/results/null_joint/<cohort>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from scipy.stats import skew as _skew

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO), str(HERE / "_vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompute.null_reference import (  # noqa: E402
    INCLUSION_RULES,
    SCHEMES,
    _column_level_stats,
    _rule_admits,
    code_columns,
    draw_permuted_codes,
    mc_pvalue,
    partition_gaps_by_rule,
)

SEED = 42
DEFAULT_REPS = 10_000
OUT_DIR = HERE / "results" / "null_joint"

#: The clinical cohorts. The combined test is taken over the estimable ones.
CLINICAL = ("uci_heart", "diabetes130", "nhis2024", "nhis2023",
            "nhanes2123", "brfss2024")


def _null_summary(v: np.ndarray) -> Dict[str, object]:
    """Location, spread and shape of one null distribution."""
    if len(v) == 0:
        return {"null_estimable": False, "n_valid_reps": 0}
    return {
        "null_estimable": True,
        "n_valid_reps": int(len(v)),
        "null_mean_gap": float(np.mean(v)),
        "null_median_gap": float(np.median(v)),
        "null_sd_gap": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
        "null_skew_gap": float(_skew(v)) if len(v) > 2 else float("nan"),
        "null_p95_gap": float(np.percentile(v, 95)),
        "null_p99_gap": float(np.percentile(v, 99)),
        # Under H0 the statistic is right-skewed, so the mean sits ABOVE the
        # median and a majority of draws fall below the mean. This is the
        # fraction that should, and it is the number the manuscript's
        # "most cohorts fall below their null mean" reading needs.
        "null_frac_below_own_mean": float(np.mean(v < np.mean(v))),
        "null_frac_below_own_median": float(np.mean(v < np.median(v))),
    }


def _level_audit(y, s, codes_by_col) -> Dict[str, Any]:
    """Which observed levels each inclusion rule admits or drops, and why."""
    audit: Dict[str, Any] = {}
    for col, codes in codes_by_col.items():
        rows = []
        for lvl in np.unique(codes):
            mask = codes == lvl
            n = int(mask.sum())
            n_pos = int(np.asarray(y)[mask].sum())
            n_neg = n - n_pos
            rows.append({
                "level_code": int(lvl),
                "n": n,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "auc_estimable": bool(n_pos >= 2 and n_neg >= 2),
                "admitted_by": [
                    k for k, r in INCLUSION_RULES.items()
                    if _rule_admits(r, n, n_pos, n_neg)
                ],
            })
        audit[col] = rows
    return audit


def _levels_used(codes_by_col, y, rule_name: str) -> Dict[str, int]:
    rule = INCLUSION_RULES[rule_name]
    out = {}
    for col, codes in codes_by_col.items():
        k = 0
        for lvl in np.unique(codes):
            mask = codes == lvl
            n = int(mask.sum())
            n_pos = int(np.asarray(y)[mask].sum())
            if _rule_admits(rule, n, n_pos, n - n_pos):
                k += 1
        out[col] = k
    return out


def run_cohort(cohort: str, n_reps: int = DEFAULT_REPS,
               random_state: int = SEED) -> Dict[str, Any]:
    from recompute.cohorts import LOADERS

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    bundle = LOADERS[cohort]()
    load_s = time.perf_counter() - t0

    X_te = np.asarray(bundle["X_test"], dtype=float)
    y = np.asarray(bundle["y_test"]).astype(int)
    demo = bundle["demo_test"]
    cols: List[str] = list(bundle["subgroup_columns"])
    s = bundle["model"].predict_proba(X_te)[:, 1]

    codes_by_col = code_columns(demo, cols)
    rule_names = list(INCLUSION_RULES)

    observed = partition_gaps_by_rule(y, s, codes_by_col, rule_names)

    payload: Dict[str, Any] = {
        "cohort": cohort,
        "is_clinical": cohort in CLINICAL,
        "seed": int(random_state),
        "n_reps": int(n_reps),
        "n_test": int(len(y)),
        "prevalence": float(y.mean()),
        "subgroup_columns": cols,
        "inclusion_rules": {k: dict(v) for k, v in INCLUSION_RULES.items()},
        "observed_gap_by_rule": observed,
        "n_levels_used_by_rule": {
            k: _levels_used(codes_by_col, y, k) for k in rule_names},
        "level_audit": _level_audit(y, s, codes_by_col),
        "load_runtime_s": load_s,
        "results": {},
    }

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)

    for scheme in SCHEMES:
        t0 = time.perf_counter()
        rng = np.random.default_rng(random_state)
        vals = np.full((len(rule_names), n_reps), np.nan, dtype=float)
        for r in range(n_reps):
            permuted = draw_permuted_codes(
                codes_by_col, pos_idx, neg_idx, rng, scheme=scheme)
            g = partition_gaps_by_rule(y, s, permuted, rule_names)
            for i, k in enumerate(rule_names):
                vals[i, r] = g[k]
        runtime = time.perf_counter() - t0

        block: Dict[str, Any] = {"runtime_s": runtime}
        for i, k in enumerate(rule_names):
            v = vals[i][~np.isnan(vals[i])]
            entry: Dict[str, Any] = {
                "rule": k,
                "rule_label": INCLUSION_RULES[k]["label"],
                "permutation_scheme": scheme,
                "n_reps": int(n_reps),
            }
            entry.update(_null_summary(v))
            obs = observed[k]
            if len(v) and np.isfinite(obs):
                entry.update(mc_pvalue(v, float(obs)))
            else:
                entry["observed_gap"] = (
                    float(obs) if np.isfinite(obs) else None)
                entry["p_value_vs_null"] = None
                entry["note"] = (
                    "not estimable: fewer than two admissible subgroups in "
                    "every partition")
            block[k] = entry
        payload["results"][scheme] = block

    payload["total_runtime_s"] = time.perf_counter() - t_start
    return payload


def main() -> int:
    from recompute.cohorts import LOADERS

    ap = argparse.ArgumentParser()
    ap.add_argument("cohort", choices=sorted(LOADERS))
    ap.add_argument("--reps", type=int, default=DEFAULT_REPS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.cohort}.json"
    try:
        payload = run_cohort(args.cohort, n_reps=args.reps,
                             random_state=args.seed)
        payload["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        payload = {
            "cohort": args.cohort,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[{args.cohort}] FAILED: {payload['error']}", file=sys.stderr)
        return 1

    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[{args.cohort}] wrote {out_path} "
          f"({payload['total_runtime_s']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
