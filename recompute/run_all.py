"""
Run every cohort under both pipelines, in parallel subprocesses.

    python -m recompute.run_all [--jobs N] [--only a,b,c] [--bootstrap N]

Each cohort runs in its own process (they are independent and each is
single-threaded in the parts that dominate), and each writes its own
``recompute/results/<cohort>.json``. A failure in one cohort does not stop the
others; the failure is recorded in that cohort's JSON.

Runtime is dominated by the BCa jackknife, which is delete-one-unit and
therefore O(n_test) replicates each costing O(n_test) -- quadratic in the test
split. The cohorts are launched longest-first so the tail does not idle.
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

# Longest-first, by test-split size (the quadratic driver).
ORDER = [
    "diabetes130",    # ~20k test rows, clustered, plus the row-split reference
    "brfss2024",      # ~9k
    "adult_income",   # ~9k
    "nhis2023",       # ~5k
    "acs_income",     # 4k
    "nhis2024",       # ~2k
    "synthetic",      # 2k
    "nhanes2123",     # ~0.8k
    "german_credit",  # 200
    "uci_heart",      # 61
]


def _run_one(cohort: str, bootstrap: int) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "recompute.run_cohort", cohort,
         "--bootstrap", str(bootstrap)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    dt = time.perf_counter() - t0
    tag = "ok " if proc.returncode == 0 else "FAIL"
    print(f"[{tag}] {cohort:<15} {dt/60:6.1f} min", flush=True)
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr, flush=True)
    return cohort, proc.returncode, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--bootstrap", type=int, default=1000)
    args = ap.parse_args()

    cohorts = (
        [c.strip() for c in args.only.split(",") if c.strip()]
        if args.only else list(ORDER)
    )
    print(f"Running {len(cohorts)} cohort(s) with {args.jobs} worker(s), "
          f"B={args.bootstrap}", flush=True)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        results = list(ex.map(lambda c: _run_one(c, args.bootstrap), cohorts))
    total = time.perf_counter() - t0

    failed = [c for c, rc, _ in results if rc != 0]
    print(f"\nWall clock: {total/60:.1f} min")
    print(f"Succeeded: {len(results) - len(failed)}/{len(results)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
