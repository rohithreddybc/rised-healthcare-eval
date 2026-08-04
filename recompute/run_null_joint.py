"""
Run the joint-vs-independent null and the m_min sweep for every cohort.

    python -m recompute.run_null_joint [--jobs N] [--reps 10000] [--only a,b]

One subprocess per cohort, each writing
``recompute/results/null_joint/<cohort>.json``. A failure in one cohort does
not stop the others.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# Longest-first: cost is O(reps * n_test log n_test * n_columns).
ORDER = [
    "diabetes130",
    "brfss2024",
    "adult_income",
    "nhis2023",
    "nhis2024",
    "acs_income",
    "synthetic",
    "nhanes2123",
    "german_credit",
    "uci_heart",
]


def _run_one(cohort: str, reps: int) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "recompute.null_joint", cohort,
         "--reps", str(reps)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    tag = "ok " if proc.returncode == 0 else "FAIL"
    print(f"[{tag}] {cohort:<15} {dt/60:6.2f} min", flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:], file=sys.stderr, flush=True)
    return cohort, proc.returncode, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--reps", type=int, default=10_000)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args()

    cohorts = (
        [c.strip() for c in args.only.split(",") if c.strip()]
        if args.only else list(ORDER)
    )
    print(f"Running {len(cohorts)} cohort(s), {args.jobs} worker(s), "
          f"B={args.reps}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda c: _run_one(c, args.reps), cohorts))
    total = time.perf_counter() - t0

    failed = [c for c, rc, _ in results if rc != 0]
    print(f"\nWall clock: {total/60:.1f} min")
    print(f"Succeeded: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
