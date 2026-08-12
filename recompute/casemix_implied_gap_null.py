"""A sampling reference for rho-hat and the implied case-mix gap.

Why this module exists
----------------------
``rho_hat`` is a maximum over levels divided by a minimum over levels, and the
implied case-mix gap is a maximum minus a minimum. Both exceed their no-effect
value with probability one, at every sample size, whether or not any case-mix
heterogeneity is present. So "every partition exceeds 1.0" is not evidence of
anything: it is an algebraic certainty. Neither quantity had a sampling
reference on disk, so a reader could not tell how much of a measured value is
case mix and how much is the noise floor of a max-minus-min statistic at that
partition's level sizes.

The null
--------
For one partition, pool the admissible levels' rows and permute the level
assignment, holding every level's size fixed. Under that permutation every level
is an equal-in-distribution draw from one common predictor distribution, so
there is no case-mix heterogeneity at all: one shared location, one shared
spread. Everything the statistic then shows is sampling noise at the observed
level sizes and level count.

Two things to keep in mind when reading the output.

1. The pooled distribution is a mixture of the real levels and is therefore at
   least as dispersed as any one of them. For the implied gap, which depends on
   the absolute spread and not only on its ratio, that makes the null slightly
   wide, so a partition sitting inside its null is a firm negative and a
   partition sitting outside is a conservative positive. ``rho_hat`` is a ratio
   and is unaffected by the common scale.
2. The null is computed per partition, so it carries the partition's own level
   count. That is deliberate: the confound between the number of levels and a
   max-minus-min statistic is exactly what the reference is for.

Which plug-in
-------------
The implied gap is evaluated with :func:`recompute.casemix_theory.auroc_from_risk`,
the distribution-free evaluation of Proposition 1 applied to the level's observed
score distribution. The Gaussian plug-in of
:func:`recompute.casemix_theory.auroc_gaussian_lp` is an adaptive quadrature and
is far too slow for a permutation null; it is also the evaluation that carries a
distributional assumption, which a reference distribution should not.

Usage
-----
    python -m recompute.casemix_implied_gap_null

Writes ``recompute/results/casemix_implied_gap_null.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from recompute.casemix_theory import auroc_from_risk
from recompute.comparators.cohort_casemix import RULES, _levels, linear_predictor
from recompute.comparators.core import CLINICAL, COHORT_LABELS, COHORT_ORDER, REPO
from recompute.null_reference import code_columns
from recompute.refit import build_full_cohort, published_fit

RESULTS = REPO / "recompute" / "results"
OUT_CSV = RESULTS / "casemix_implied_gap_null.csv"

#: Replicates per partition.
B = 2000

#: Pinned so the reference reproduces exactly.
SEED = 20240810


def _stats(scores: np.ndarray, lp: np.ndarray, sizes: Sequence[int]):
    """``(rho_hat, implied_gap)`` for one partition of the rows into levels.

    Rows are taken in order: the first ``sizes[0]`` go to level 0, and so on.
    """
    out_auc: List[float] = []
    out_sd: List[float] = []
    lo = 0
    for n in sizes:
        sl = slice(lo, lo + n)
        lo += n
        out_sd.append(float(np.std(lp[sl], ddof=1)))
        out_auc.append(auroc_from_risk(scores[sl]))
    finite = [a for a in out_auc if np.isfinite(a)]
    gap = (max(finite) - min(finite)) if len(finite) >= 2 else float("nan")
    smin, smax = min(out_sd), max(out_sd)
    rho = (smax / smin) if smin > 0 else float("inf")
    return rho, gap


def null_rows_for_fit(fit, subgroup_columns: Sequence[str], b: int = B,
                      seed: int = SEED) -> List[Dict]:
    """Observed statistics and their permutation null, per (rule, partition)."""
    y = np.asarray(fit.y_test).astype(int)
    s = np.asarray(fit.scores, dtype=float)
    lp = linear_predictor(s)
    codes_by_col = code_columns(fit.demo_test, list(subgroup_columns))
    rng = np.random.default_rng(seed)

    rows: List[Dict] = []
    for rule in RULES:
        for col, codes in codes_by_col.items():
            lv = _levels(y, s, codes, rule)
            if len(lv) < 2:
                continue
            idx = np.concatenate([np.flatnonzero(m) for _, m, *_ in lv])
            sizes = [int(n) for _, _, n, *_ in lv]
            obs_rho, obs_gap = _stats(s[idx], lp[idx], sizes)

            pool_s, pool_lp = s[idx], lp[idx]
            n = pool_s.size
            null_rho = np.empty(b)
            null_gap = np.empty(b)
            for i in range(b):
                p = rng.permutation(n)
                null_rho[i], null_gap[i] = _stats(pool_s[p], pool_lp[p], sizes)

            ok = np.isfinite(null_gap)
            rows.append({
                "cohort": fit.cohort,
                "cohort_label": COHORT_LABELS.get(fit.cohort, fit.cohort),
                "is_clinical": fit.cohort in CLINICAL,
                "rule": rule,
                "partition": col,
                "partition_key": f"{fit.cohort}|{col}",
                "n_levels_admissible": len(lv),
                "min_level_n": min(sizes),
                "max_level_n": max(sizes),
                "n_replicates": int(b),
                "observed_sd_ratio": obs_rho,
                "null_sd_ratio_median": float(np.median(null_rho)),
                "null_sd_ratio_p95": float(np.quantile(null_rho, 0.95)),
                "sd_ratio_null_percentile": float(np.mean(null_rho <= obs_rho)),
                "sd_ratio_exceeds_null_p95": bool(
                    obs_rho > float(np.quantile(null_rho, 0.95))),
                "observed_implied_gap_empirical": obs_gap,
                "null_implied_gap_median": float(np.median(null_gap[ok])),
                "null_implied_gap_p95": float(np.quantile(null_gap[ok], 0.95)),
                "implied_gap_null_percentile": float(
                    np.mean(null_gap[ok] <= obs_gap)),
                "implied_gap_exceeds_null_p95": bool(
                    obs_gap > float(np.quantile(null_gap[ok], 0.95))),
                "implied_gap_minus_null_median": obs_gap - float(
                    np.median(null_gap[ok])),
            })
    return rows


def run(cohorts: Sequence[str], b: int = B, verbose: bool = True) -> pd.DataFrame:
    rows: List[Dict] = []
    for name in cohorts:
        t0 = time.perf_counter()
        fc = build_full_cohort(name)
        f = published_fit(fc)
        rows += null_rows_for_fit(f, fc.subgroup_columns, b=b)
        if verbose:
            print(f"{name:14s} done in {time.perf_counter() - t0:.1f}s",
                  flush=True)
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--b", type=int, default=B)
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args(argv)

    names = ([c.strip() for c in args.only.split(",") if c.strip()]
             or [c for c in COHORT_ORDER if c in CLINICAL])
    t0 = time.perf_counter()
    df = run(names, b=args.b)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(args.out), index=False)
    print(f"wrote {args.out} ({len(df)} rows) in {time.perf_counter() - t0:.1f}s")

    d = df[(df["rule"] == "m30") & df["is_clinical"]]
    print(f"\n=== rule m30, clinical, {len(d)} partitions, B={args.b} ===")
    print(d[["cohort", "partition", "n_levels_admissible", "observed_sd_ratio",
             "null_sd_ratio_median", "null_sd_ratio_p95",
             "observed_implied_gap_empirical", "null_implied_gap_median",
             "null_implied_gap_p95", "implied_gap_exceeds_null_p95"]]
          .to_string(index=False))
    print(f"\nsd_ratio  above own null p95 : "
          f"{int(d['sd_ratio_exceeds_null_p95'].sum())} of {len(d)}")
    print(f"implied   above own null p95 : "
          f"{int(d['implied_gap_exceeds_null_p95'].sum())} of {len(d)}")
    print(f"null sd_ratio median, overall: "
          f"{d['null_sd_ratio_median'].median():.4f} "
          f"({d['null_sd_ratio_median'].min():.4f} to "
          f"{d['null_sd_ratio_median'].max():.4f})")
    print(f"null implied median, overall : "
          f"{d['null_implied_gap_median'].median():.4f} "
          f"({d['null_implied_gap_median'].min():.4f} to "
          f"{d['null_implied_gap_median'].max():.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
