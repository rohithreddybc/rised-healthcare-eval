"""
Same-kernel runtime comparison.

    python -m recompute.comparators.bench [--reps 10000] [--only a,b]

The runtimes stored in ``results/null_joint/`` were produced by the incumbent's
original ``scipy.rankdata`` code path; the comparators run on the vectorised
kernel in :mod:`recompute.comparators.core`. Putting those two numbers in the
same table would credit studentization with a speed-up that belongs to the
kernel. This module times the incumbent's own statistic on the *same* kernel, so
the comparison is like for like, and writes
``recompute/results/comparator_runtime.csv``.
"""

from __future__ import annotations

import argparse
import time
from typing import List, Optional

import pandas as pd

from recompute.comparators import N_PERM, SEED
from recompute.comparators.core import COHORT_LABELS, COHORT_ORDER, REPO, load_cohort

OUT = REPO / "recompute" / "results" / "comparator_runtime.csv"


def main(argv: Optional[List[str]] = None) -> int:
    from recompute.comparators import diciccio, four_fifths, incumbent, lum, naive

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=N_PERM)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args(argv)

    cohorts = ([c.strip() for c in args.only.split(",") if c.strip()]
               or list(COHORT_ORDER))
    rows = []
    for c in cohorts:
        data = load_cohort(c)
        stored = incumbent.load_payload(c)

        inc = incumbent.recompute_null(data, n_perm=args.reps, seed=args.seed)
        dc = diciccio.run_cohort(data, n_perm=args.reps, seed=args.seed)
        t0 = time.perf_counter()
        lum.run_cohort(data, seed=args.seed)
        t_lum = time.perf_counter() - t0
        t0 = time.perf_counter()
        four_fifths.run_cohort(data)
        t_ff = time.perf_counter() - t0
        t0 = time.perf_counter()
        naive.run_cohort(data)
        t_naive = time.perf_counter() - t0

        rows.append({
            "cohort": c,
            "cohort_label": COHORT_LABELS.get(c, c),
            "n_test": data.n_test,
            "n_partitions": len(data.codes_by_col),
            "n_perm": args.reps,
            "permutation_null_original_kernel_s": (
                float(stored["results"]["joint"]["runtime_s"])),
            "permutation_null_same_kernel_s": float(inc["runtime_s"]),
            "diciccio2020_s": float(dc["runtime_s"]),
            "lum2022_s": t_lum,
            "four_fifths_s": t_ff,
            "fixed_threshold_005_s": t_naive,
        })
        print(f"[ok ] {c:<15} incumbent {inc['runtime_s']:7.1f}s  "
              f"diciccio {dc['runtime_s']:7.1f}s  lum {t_lum:6.3f}s",
              flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    print(df[["cohort", "permutation_null_original_kernel_s",
              "permutation_null_same_kernel_s", "diciccio2020_s",
              "lum2022_s"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
