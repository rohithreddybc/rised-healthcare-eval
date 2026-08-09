"""
Single entry point for the comparator evaluation.

    python -m recompute.comparators.run --stage all
    python -m recompute.comparators.run --stage cohorts [--jobs 5] [--reps 10000]
    python -m recompute.comparators.run --stage type1   [--jobs 5] [--sims 1000]

``cohorts`` writes ``recompute/results/comparator_comparison.csv``: one row per
(cohort x inclusion rule x method), carrying the conclusion, the statistic, the
p-value where the method defines one, and the runtime.

``type1`` delegates to :mod:`recompute.comparators.type1` and writes
``recompute/results/comparator_type1.csv``.

Every stochastic component is seeded at 42 and every permutation-based method
runs at B = 10,000 on the cohorts, matching the incumbent exactly, so the
comparison is paired down to the individual permutation draw.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from recompute.comparators import ALPHA, N_PERM, SEED
from recompute.comparators.core import (
    COHORT_LABELS,
    COHORT_ORDER,
    REPO,
    RULE_NAMES,
    load_cohort,
)

RESULTS = REPO / "recompute" / "results"
COMPARISON_CSV = RESULTS / "comparator_comparison.csv"
DIAGNOSTICS_DIR = RESULTS / "comparators"

from recompute.null_reference import INCLUSION_RULES  # noqa: E402


def _rows_for_cohort(cohort: str, n_perm: int, seed: int, alpha: float,
                     scheme: str = "joint") -> List[Dict[str, object]]:
    """Run every method on one cohort and return flat rows."""
    from recompute.comparators import diciccio, four_fifths, incumbent, lum, naive

    data = load_cohort(cohort)
    blocks = {
        incumbent.METHOD: incumbent.run_cohort(cohort, alpha=alpha,
                                               scheme=scheme),
        diciccio.METHOD: diciccio.run_cohort(data, n_perm=n_perm, seed=seed,
                                             alpha=alpha, scheme=scheme),
        lum.METHOD: lum.run_cohort(data, alpha=alpha, seed=seed),
        four_fifths.METHOD: four_fifths.run_cohort(data),
        naive.METHOD: naive.run_cohort(data),
    }

    # Lum emits three readings of the same estimator (closed-form z-test,
    # Cochran's Q, parametric-bootstrap CI). All three go in the CSV so the
    # choice of primary is auditable.
    flat = {m: b["results"] for m, b in blocks.items()}
    flat.update(blocks[lum.METHOD].get("variants", {}))

    rows: List[Dict[str, object]] = []
    for method, block_results in flat.items():
        for rule, res in block_results.items():
            rows.append({
                "cohort": cohort,
                "cohort_label": COHORT_LABELS.get(cohort, cohort),
                "is_clinical": data.is_clinical,
                "n_test": data.n_test,
                "prevalence": data.prevalence,
                "n_partitions": len(data.codes_by_col),
                "rule": rule,
                "rule_label": INCLUSION_RULES[rule]["label"],
                "n_perm": n_perm if method in ("permutation_null",
                                               "diciccio2020") else None,
                # Which demographic-column permutation scheme produced this row.
                # Recorded per row so no reader has to infer it from the source;
                # see recompute/scheme_provenance.py. Blank for the closed-form
                # and deterministic methods, which do not permute at all.
                "permutation_scheme": (scheme if method in ("permutation_null",
                                                            "diciccio2020")
                                       else ""),
                "seed": seed,
                "alpha": alpha,
                **res.as_row(),
            })

    # Full per-partition diagnostics, kept out of the CSV so it stays readable.
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    diag = {
        "cohort": cohort,
        "n_test": data.n_test,
        "prevalence": data.prevalence,
        "load_runtime_s": data.load_runtime_s,
        "n_perm": n_perm,
        "seed": seed,
        "diciccio": {
            r: {k: (list(v) if isinstance(v, tuple) else v)
                for k, v in d.items()}
            for r, d in blocks["diciccio2020"].get("diagnostics", {}).items()},
        "lum": blocks["lum2022"].get("diagnostics", {}),
        "four_fifths": blocks["four_fifths"].get("diagnostics", {}),
    }
    (DIAGNOSTICS_DIR / f"{cohort}.json").write_text(
        json.dumps(diag, indent=2, default=_jsonable), encoding="utf-8")
    return rows


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _worker(args):
    cohort, n_perm, seed, alpha, scheme = args
    t0 = time.perf_counter()
    try:
        rows = _rows_for_cohort(cohort, n_perm, seed, alpha, scheme)
        print(f"[ok ] {cohort:<15} {scheme:<11} "
              f"{(time.perf_counter()-t0)/60:6.2f} min", flush=True)
        return rows
    except Exception as exc:                                    # noqa: BLE001
        print(f"[FAIL] {cohort}: {type(exc).__name__}: {exc}", file=sys.stderr,
              flush=True)
        print(traceback.format_exc()[-3000:], file=sys.stderr, flush=True)
        return []


def add_cross_cohort_multiplicity(df: pd.DataFrame, alpha: float
                                  ) -> pd.DataFrame:
    """Holm and Benjamini-Hochberg across cohorts, within each cell.

    Ten cohorts are tested at once and the headline reading is "which cohorts
    were flagged", so the raw per-cohort decision is a decision made ten times.
    ``recompute/results/null_sweep_mmin.csv`` already carried ``p_holm`` and
    ``p_bh`` for the incumbent and neither was surfaced anywhere a reader would
    see. They are now attached to every p-valued row of the main comparator
    table, adjusted within each ``(method, rule, permutation_scheme)`` cell over
    the cohorts that are estimable there -- which is the family the "k of ten
    cohorts are inequitable" sentence is drawn from.

    Holm controls the family-wise error rate under arbitrary dependence; BH
    controls the false discovery rate. Both are reported because they answer
    different questions and, at these p-values, sometimes disagree.
    """
    from recompute.aggregate_null_joint import benjamini_hochberg, holm

    df = df.copy()
    for col in ("p_holm_across_cohorts", "p_bh_across_cohorts"):
        df[col] = np.nan
    keys = ["method", "rule", "permutation_scheme"]
    for _, grp in df.groupby(keys, dropna=False):
        ok = grp["p_value"].notna()
        idx = grp.index[ok]
        if len(idx) == 0:
            continue
        pv = list(df.loc[idx, "p_value"].astype(float))
        df.loc[idx, "p_holm_across_cohorts"] = holm(pv)
        df.loc[idx, "p_bh_across_cohorts"] = benjamini_hochberg(pv)
    df["flag_raw"] = df["conclusion"] == "flag"
    df["flag_holm"] = df["p_holm_across_cohorts"] < alpha
    df["flag_bh"] = df["p_bh_across_cohorts"] < alpha
    # The deterministic rules have no p-value, so multiplicity does not apply;
    # their adjusted decision is their raw decision and is marked as such.
    no_p = df["p_value"].isna()
    for col in ("flag_holm", "flag_bh"):
        df.loc[no_p, col] = df.loc[no_p, "flag_raw"]
    df["multiplicity_applies"] = ~no_p
    return df


def run_cohorts(cohorts: List[str], n_perm: int, seed: int, alpha: float,
                jobs: int, out: Path, schemes: Sequence[str] = ("joint",)
                ) -> pd.DataFrame:
    print(f"Comparator sweep: {len(cohorts)} cohorts x {len(RULE_NAMES)} rules "
          f"x 5 methods x {len(schemes)} scheme(s) {tuple(schemes)}, "
          f"B={n_perm}, seed={seed}, {jobs} worker(s)", flush=True)
    t0 = time.perf_counter()
    payload = [(c, n_perm, seed, alpha, s) for s in schemes for c in cohorts]
    rows: List[Dict[str, object]] = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for r in ex.map(_worker, payload):
                rows.extend(r)
    else:
        for p in payload:
            rows.extend(_worker(p))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_cross_cohort_multiplicity(df, alpha)
        order = {c: i for i, c in enumerate(COHORT_ORDER)}
        rorder = {r: i for i, r in enumerate(RULE_NAMES)}
        df["_c"] = df["cohort"].map(order)
        df["_r"] = df["rule"].map(rorder)
        df = df.sort_values(["permutation_scheme", "_c", "_r", "method"]
                            ).drop(columns=["_c", "_r"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows, "
          f"{(time.perf_counter()-t0)/60:.1f} min)")
    return df


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("all", "cohorts", "type1"),
                    default="all")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--reps", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--perm", type=int, default=999)
    ap.add_argument("--schemes", type=str, default="joint",
                    help="comma-separated permutation schemes to run the "
                         "cohort sweep under: joint, independent, or both")
    args = ap.parse_args(argv)

    rc = 0
    if args.stage in ("all", "cohorts"):
        cohorts = ([c.strip() for c in args.only.split(",") if c.strip()]
                   or list(COHORT_ORDER))
        schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
        df = run_cohorts(cohorts, args.reps, args.seed, args.alpha, args.jobs,
                         COMPARISON_CSV, schemes)
        if df.empty:
            rc = 1
    if args.stage in ("all", "type1"):
        from recompute.comparators import type1

        rc |= type1.main(["--sims", str(args.sims), "--perm", str(args.perm),
                          "--jobs", str(args.jobs), "--seed", str(args.seed),
                          "--alpha", str(args.alpha)])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
