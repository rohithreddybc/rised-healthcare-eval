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
from typing import Dict, List, Optional

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


def _rows_for_cohort(cohort: str, n_perm: int, seed: int, alpha: float
                     ) -> List[Dict[str, object]]:
    """Run every method on one cohort and return flat rows."""
    from recompute.comparators import diciccio, four_fifths, incumbent, lum, naive

    data = load_cohort(cohort)
    blocks = {
        incumbent.METHOD: incumbent.run_cohort(cohort, alpha=alpha),
        diciccio.METHOD: diciccio.run_cohort(data, n_perm=n_perm, seed=seed,
                                             alpha=alpha),
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
    cohort, n_perm, seed, alpha = args
    t0 = time.perf_counter()
    try:
        rows = _rows_for_cohort(cohort, n_perm, seed, alpha)
        print(f"[ok ] {cohort:<15} {(time.perf_counter()-t0)/60:6.2f} min",
              flush=True)
        return rows
    except Exception as exc:                                    # noqa: BLE001
        print(f"[FAIL] {cohort}: {type(exc).__name__}: {exc}", file=sys.stderr,
              flush=True)
        print(traceback.format_exc()[-3000:], file=sys.stderr, flush=True)
        return []


def run_cohorts(cohorts: List[str], n_perm: int, seed: int, alpha: float,
                jobs: int, out: Path) -> pd.DataFrame:
    print(f"Comparator sweep: {len(cohorts)} cohorts x {len(RULE_NAMES)} rules "
          f"x 5 methods, B={n_perm}, seed={seed}, {jobs} worker(s)", flush=True)
    t0 = time.perf_counter()
    payload = [(c, n_perm, seed, alpha) for c in cohorts]
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
        order = {c: i for i, c in enumerate(COHORT_ORDER)}
        rorder = {r: i for i, r in enumerate(RULE_NAMES)}
        df["_c"] = df["cohort"].map(order)
        df["_r"] = df["rule"].map(rorder)
        df = df.sort_values(["_c", "_r", "method"]).drop(columns=["_c", "_r"])
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
    args = ap.parse_args(argv)

    rc = 0
    if args.stage in ("all", "cohorts"):
        cohorts = ([c.strip() for c in args.only.split(",") if c.strip()]
                   or list(COHORT_ORDER))
        df = run_cohorts(cohorts, args.reps, args.seed, args.alpha, args.jobs,
                         COMPARISON_CSV)
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
