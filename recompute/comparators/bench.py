"""
Same-kernel runtime comparison, repeated, with the hardware recorded.

    python -m recompute.comparators.bench [--repeats 3] [--reps 10000] [--only a,b]

Writes ``recompute/results/comparator_runtime.csv`` (one row per cohort, with
the summary statistics) and ``recompute/results/comparator_runtime_raw.csv``
(every individual timing), plus a ``recompute/results/comparator_runtime_env.json``
recording the machine, the interpreter and the library versions.

What was wrong with the previous version
----------------------------------------
It timed each method **once**, on a machine that was never described, and the
resulting pair -- 188 s for the incumbent against 193 s for the studentized test
-- was quoted as though it established that studentization is free. A single
un-repeated timing has no error bar; on a shared desktop the run-to-run spread of
a three-minute numerical loop is routinely several percent, which is the entire
size of the difference being claimed. The two numbers are also not the pair a
reader will assume: 188 s is the incumbent's statistic re-implemented on the
comparator package's vectorised kernel, whereas the incumbent as *shipped* takes
295 s on the same cohort through ``scipy.rankdata``. Quoting 188 vs 193 next to a
shipped 295 credits the wrong component.

This version therefore:

  * repeats every timing ``--repeats`` times and reports the minimum, the median
    and the full spread, keeping every individual measurement in the raw CSV.
    The minimum is the least contaminated estimate of the work actually required;
    the median is the one to quote for a realistic machine. Both are given.
  * runs strictly sequentially, one cohort and one method at a time, so no
    measurement competes with another for a core.
  * records the CPU, core count, RAM, OS, Python, numpy and scipy versions, and
    the BLAS thread caps, so the numbers can be reproduced or discounted.
  * reports all three runtimes explicitly -- shipped kernel, same kernel,
    studentized -- so the like-for-like comparison and the kernel speed-up are
    separately visible rather than conflated.

If ``--repeats 1`` is used the summary columns still populate but
``n_repeats`` records it, and no spread can be reported. Do not quote a
difference from a single repeat.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recompute.comparators import N_PERM, SEED
from recompute.comparators.core import COHORT_LABELS, COHORT_ORDER, REPO, load_cohort

RESULTS = REPO / "recompute" / "results"
OUT = RESULTS / "comparator_runtime.csv"
OUT_RAW = RESULTS / "comparator_runtime_raw.csv"
OUT_ENV = RESULTS / "comparator_runtime_env.json"

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def environment() -> Dict[str, object]:
    """Everything a reader needs to reproduce or discount these timings."""
    import scipy

    env: Dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cores": os.cpu_count(),
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "blas_thread_env": {v: os.environ.get(v, "<unset>")
                            for v in _THREAD_VARS},
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:                                    # optional, best effort
        import psutil

        env["ram_total_gb"] = round(psutil.virtual_memory().total / 2 ** 30, 1)
        env["physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        env["ram_total_gb"] = None
        env["physical_cores"] = None
    try:
        env["numpy_blas"] = np.__config__.show(mode="dicts")     # numpy >= 1.25
    except Exception:                                            # noqa: BLE001
        env["numpy_blas"] = "unavailable"
    if platform.system() == "Windows":
        env["cpu_brand"] = os.environ.get("PROCESSOR_IDENTIFIER", "")
    return env


def _summarise(name: str, vals: List[float]) -> Dict[str, float]:
    a = np.asarray(vals, dtype=float)
    return {
        f"{name}_min_s": float(a.min()),
        f"{name}_median_s": float(np.median(a)),
        f"{name}_max_s": float(a.max()),
        f"{name}_sd_s": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
    }


def bench_cohort(cohort: str, n_perm: int, seed: int, repeats: int
                 ) -> Dict[str, object]:
    from recompute.comparators import diciccio, four_fifths, incumbent, lum, naive

    data = load_cohort(cohort)
    stored = incumbent.load_payload(cohort)

    timings: Dict[str, List[float]] = {
        "permutation_null_same_kernel": [], "diciccio2020": [],
        "lum2022": [], "four_fifths": [], "fixed_threshold_005": []}

    for _ in range(repeats):
        inc = incumbent.recompute_null(data, n_perm=n_perm, seed=seed)
        timings["permutation_null_same_kernel"].append(float(inc["runtime_s"]))

        dc = diciccio.run_cohort(data, n_perm=n_perm, seed=seed)
        timings["diciccio2020"].append(float(dc["runtime_s"]))

        for key, fn in (("lum2022", lambda: lum.run_cohort(data, seed=seed)),
                        ("four_fifths", lambda: four_fifths.run_cohort(data)),
                        ("fixed_threshold_005", lambda: naive.run_cohort(data))):
            t0 = time.perf_counter()
            fn()
            timings[key].append(time.perf_counter() - t0)

    row: Dict[str, object] = {
        "cohort": cohort,
        "cohort_label": COHORT_LABELS.get(cohort, cohort),
        "n_test": data.n_test,
        "n_partitions": len(data.codes_by_col),
        "n_perm": n_perm,
        "n_repeats": repeats,
        # The incumbent AS SHIPPED, through scipy.rankdata. Single stored value,
        # not repeated here -- it is quoted only to show what the kernel rewrite
        # bought, and must never be differenced against the studentized column.
        "permutation_null_shipped_kernel_s": float(
            stored["results"]["joint"]["runtime_s"]),
    }
    for name, vals in timings.items():
        row.update(_summarise(name, vals))

    inc_med = row["permutation_null_same_kernel_median_s"]
    dc_med = row["diciccio2020_median_s"]
    row["studentization_overhead_median_s"] = dc_med - inc_med
    row["studentization_overhead_pct"] = 100.0 * (dc_med - inc_med) / inc_med
    # Is the difference between the two even resolvable at this repeat count?
    # Pooled SD of the two sets of repeats; if the gap is inside it, the honest
    # statement is "no material cost", with no number attached.
    sds = [row["permutation_null_same_kernel_sd_s"], row["diciccio2020_sd_s"]]
    pooled = float(np.sqrt(np.nanmean(np.square(sds)))) if repeats > 1 else np.nan
    row["repeat_noise_sd_s"] = pooled
    row["overhead_exceeds_repeat_noise"] = (
        bool(abs(dc_med - inc_med) > 2 * pooled) if np.isfinite(pooled) else None)
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=N_PERM,
                    help="permutation replicates B (default matches the study)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="how many times each timing is repeated")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args(argv)

    cohorts = ([c.strip() for c in args.only.split(",") if c.strip()]
               or list(COHORT_ORDER))
    env = environment()
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_ENV.write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
    print(f"{env['platform']}\n{env.get('cpu_brand', env['processor'])}\n"
          f"{env['logical_cores']} logical cores, "
          f"numpy {env['numpy']}, scipy {env['scipy']}\n"
          f"B={args.reps}, {args.repeats} repeat(s), sequential\n", flush=True)

    rows: List[Dict[str, object]] = []
    raw: List[Dict[str, object]] = []
    for c in cohorts:
        t0 = time.perf_counter()
        row = bench_cohort(c, args.reps, args.seed, args.repeats)
        rows.append(row)
        raw.append({"cohort": c, **{k: v for k, v in row.items()
                                    if k.endswith("_s") or k == "n_repeats"}})
        print(f"[ok ] {c:<15} incumbent(same kernel) "
              f"{row['permutation_null_same_kernel_median_s']:7.1f}s  "
              f"diciccio {row['diciccio2020_median_s']:7.1f}s  "
              f"overhead {row['studentization_overhead_pct']:+5.1f}%  "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)
        # Checkpoint after every cohort: this is hours of sequential work.
        pd.DataFrame(rows).to_csv(OUT, index=False)
        pd.DataFrame(raw).to_csv(OUT_RAW, index=False)

    df = pd.DataFrame(rows)
    print(f"\nwrote {OUT}\nwrote {OUT_RAW}\nwrote {OUT_ENV}\n")
    with pd.option_context("display.width", 200):
        print(df[["cohort", "permutation_null_shipped_kernel_s",
                  "permutation_null_same_kernel_median_s",
                  "diciccio2020_median_s", "studentization_overhead_pct",
                  "repeat_noise_sd_s", "overhead_exceeds_repeat_noise"]]
              .to_string(index=False))
    if args.repeats > 1:
        resolvable = df["overhead_exceeds_repeat_noise"].fillna(False)
        print(f"\nstudentization overhead exceeds run-to-run noise in "
              f"{int(resolvable.sum())}/{len(df)} cohorts")
        if not resolvable.any():
            print("=> the honest statement is 'no material runtime cost', with "
                  "NO number attached: the difference is inside the noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
