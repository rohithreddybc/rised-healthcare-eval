"""
Type I error of all five procedures under a simulated null.

Why this is the fairest comparison axis
---------------------------------------
On the ten real cohorts every method is being judged by what it concludes, with
no ground truth. Under :mod:`recompute.comparators.simulate` there *is* ground
truth: subgroup membership is independent of the score given the outcome, so
every subgroup shares one true AUROC and every flag is a false positive. A method
that flags 20% of the time at a nominal 5% is not "more sensitive", it is broken;
a method that flags 0% of the time is throwing away power it could have spent.
Neither fact is visible from the cohort table alone, and neither has been checked
for any of these methods in this project, including the incumbent.

The two deterministic rules (four-fifths, fixed 0.05 threshold) have no nominal
level. Their "Type I error" is simply P(flag | null), reported on the same scale;
for those rows the column is a false-positive rate against a rule that never
claimed to control one, and it should be read that way.

Replicate counts
----------------
``--sims`` simulated datasets per (geometry, method); ``--perm`` permutation
replicates inside each permutation-based method. Defaults are 1000 and 999. The
Monte-Carlo standard error of a 0.05 rate at 1000 simulations is 0.0069, which
resolves the difference between 0.05 and, say, 0.08. Both numbers are recorded in
every output row; nothing is reduced silently.
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recompute.comparators import ALPHA, SEED
from recompute.comparators.core import REPO
from recompute.comparators.simulate import GEOMETRIES, GEOMETRY_BY_NAME, Geometry

RESULTS = REPO / "recompute" / "results"
OUT_CSV = RESULTS / "comparator_type1.csv"

DEFAULT_SIMS = 1000
DEFAULT_PERM = 999

#: The inclusion rule the Type I study is run at. m30 is the published default;
#: ev10 is added because the cohort analysis turns on it.
DEFAULT_RULES = ("m30", "ev10")

METHODS = ("permutation_null", "diciccio2020", "lum2022",
           "four_fifths", "fixed_threshold_005")


def _one_sim(geom: Geometry, rep: int, rule: str, n_perm: int,
             seed: int, alpha: float) -> Dict[str, float]:
    """Run every method once on one simulated null dataset."""
    from recompute.comparators import diciccio, four_fifths, incumbent, lum, naive
    from recompute.comparators.core import PermContext
    from recompute.comparators.simulate import make_dataset

    y, s, codes = make_dataset(geom, rep, seed)
    ctx = PermContext(y, s, codes)
    out: Dict[str, float] = {}

    rng = np.random.default_rng([seed, rep, 1])
    t = time.perf_counter()
    p = incumbent.pvalue_only(ctx, rule, n_perm, rng)
    out["permutation_null"] = np.nan if not np.isfinite(p) else float(p < alpha)
    out["permutation_null__p"] = p
    out["permutation_null__t"] = time.perf_counter() - t

    # Same seed, so the studentized test sees exactly the permutation draws the
    # incumbent saw. Any difference in the two rows is the statistic alone.
    rng = np.random.default_rng([seed, rep, 1])
    t = time.perf_counter()
    p = diciccio.pvalue_only(ctx, rule, n_perm, rng)
    out["diciccio2020"] = np.nan if not np.isfinite(p) else float(p < alpha)
    out["diciccio2020__p"] = p
    out["diciccio2020__t"] = time.perf_counter() - t

    t = time.perf_counter()
    d = lum.decide(ctx, rule, alpha=alpha, seed=seed + rep)
    out["lum2022"] = d["flag"]
    out["lum2022__p"] = d["p_value"]
    out["lum2022__q"] = d["flag_q"]
    out["lum2022__boot"] = d["flag_boot"]
    out["lum2022__t"] = time.perf_counter() - t

    t = time.perf_counter()
    out["four_fifths"] = four_fifths.decide(ctx, rule)
    out["four_fifths__t"] = time.perf_counter() - t

    t = time.perf_counter()
    out["fixed_threshold_005"] = naive.decide(ctx, rule)
    out["fixed_threshold_005__t"] = time.perf_counter() - t
    return out


def _run_geometry(args) -> List[Dict[str, object]]:
    geom_name, rule, n_sims, n_perm, seed, alpha = args
    geom = GEOMETRY_BY_NAME[geom_name]
    t0 = time.perf_counter()
    sims = [_one_sim(geom, r, rule, n_perm, seed, alpha) for r in range(n_sims)]
    wall = time.perf_counter() - t0

    rows: List[Dict[str, object]] = []
    base = {
        "geometry": geom.name,
        "description": geom.description,
        "n": geom.n,
        "prevalence": geom.prevalence,
        "n_partitions": len(geom.partitions),
        "max_levels": max(len(p) for p in geom.partitions),
        "min_level_frac": min(min(p) for p in geom.partitions),
        "composite_null": geom.is_composite,
        "true_auc": geom.auc,
        "rule": rule,
        "n_sims": n_sims,
        "n_perm": n_perm,
        "alpha": alpha,
        "geometry_wall_s": wall,
    }
    for m in METHODS:
        vals = np.array([sd.get(m, np.nan) for sd in sims], dtype=float)
        ok = np.isfinite(vals)
        n_ok = int(ok.sum())
        rate = float(vals[ok].mean()) if n_ok else float("nan")
        se = float(np.sqrt(rate * (1 - rate) / n_ok)) if n_ok else float("nan")
        t_key = f"{m}__t"
        rows.append({
            **base,
            "method": m,
            "n_evaluable": n_ok,
            "n_not_evaluable": int(n_sims - n_ok),
            "n_flag": int(np.nansum(vals)),
            "type1_rate": rate,
            "type1_mc_se": se,
            "has_nominal_level": m in ("permutation_null", "diciccio2020",
                                       "lum2022"),
            "mean_runtime_per_dataset_s": float(
                np.mean([sd[t_key] for sd in sims])),
        })
    # The two secondary Lum readings get their own rows: Cochran's Q (our
    # precision-weighted addition) and the parametric-bootstrap CI rule.
    for key, name in (("lum2022__q", "lum2022_cochranQ"),
                      ("lum2022__boot", "lum2022_bootstrapCI")):
        vals = np.array([sd.get(key, np.nan) for sd in sims], dtype=float)
        ok = np.isfinite(vals)
        n_ok = int(ok.sum())
        rate = float(vals[ok].mean()) if n_ok else float("nan")
        rows.append({
            **base,
            "method": name,
            "n_evaluable": n_ok,
            "n_not_evaluable": int(n_sims - n_ok),
            "n_flag": int(np.nansum(vals)),
            "type1_rate": rate,
            "type1_mc_se": (float(np.sqrt(rate * (1 - rate) / n_ok))
                            if n_ok else float("nan")),
            "has_nominal_level": True,
            "mean_runtime_per_dataset_s": float(
                np.mean([sd["lum2022__t"] for sd in sims])),
        })
    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--perm", type=int, default=DEFAULT_PERM)
    ap.add_argument("--rules", type=str, default=",".join(DEFAULT_RULES))
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args(argv)

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    names = ([g.strip() for g in args.only.split(",") if g.strip()]
             or [g.name for g in GEOMETRIES])
    jobs = [(nm, rule, args.sims, args.perm, args.seed, args.alpha)
            for nm in names for rule in rules]

    print(f"Type I study: {len(names)} geometries x {len(rules)} rules, "
          f"{args.sims} sims, B={args.perm}, {args.jobs} worker(s)", flush=True)
    t0 = time.perf_counter()
    rows: List[Dict[str, object]] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for res in ex.map(_run_geometry, jobs):
                rows.extend(res)
                print(f"  done {res[0]['geometry']} / {res[0]['rule']} "
                      f"({res[0]['geometry_wall_s']/60:.1f} min)", flush=True)
    else:
        for j in jobs:
            res = _run_geometry(j)
            rows.extend(res)
            print(f"  done {res[0]['geometry']} / {res[0]['rule']} "
                  f"({res[0]['geometry_wall_s']/60:.1f} min)", flush=True)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows, "
          f"{(time.perf_counter()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
