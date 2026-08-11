"""
The replacement metrics run through the geometries that indict the AUROC gap.

The question
------------
The manuscript establishes that under case mix -- one shared model, identical
coefficients, Bayes-optimal, perfectly calibrated in every subgroup, differing
only in covariate distribution -- every subgroup AUROC-gap procedure flags the
perfectly fair model at high rates. It then recommends **subgroup calibration**
and **subgroup net benefit** in place of the equal-AUROC null.

A referee observed, correctly, that the manuscript recommends metrics whose
operating characteristics it never measured. This module measures them, on the
same geometries, at the same seed, on the same simulated draws.

What is run
-----------
Every geometry in :data:`recompute.comparators.simulate.GEOMETRIES` -- the six
case-mix geometries and the seventeen equal-AUROC (simple and composite) ones --
at 1000 simulated datasets each, B = 999 permutation replicates, seed 42,
inclusion rules ``m30`` and ``ev10``. Identical to the AUROC study on every one
of those knobs, and the datasets are literally the same objects: this module
calls the same :func:`~recompute.comparators.simulate.make_dataset` with the same
``(geometry, replicate, seed)`` and never draws its own cohorts.

What is reported, per (geometry, rule, metric)
----------------------------------------------
``scope="level_k"``  The per-subgroup distribution of the metric on partition
                     ``p0`` across the 1000 replicates, against its exact value
                     where that is known.
``scope="gap"``      The distribution of the max-over-subgroups gap -- the same
                     functional the AUROC study indicts -- with its mean, spread
                     and upper quantiles, and the exact true gap.

and per (geometry, rule, metric, test), the flag rate of three naive procedures:

``fixed_threshold``  Gap >= a conventional cut (:data:`.replacement.FIXED_RULE_CUT`).
                     The direct analogue of the manuscript's ``fixed_threshold_005``.
``wald_maxt``        Largest studentized pairwise contrast, two-sided normal,
                     Bonferroni over pairs. Asymptotically valid under a true
                     null of equal subgroup metric.
``permutation``      Monte-Carlo p-value of the gap under joint label permutation
                     -- the incumbent's own procedure with the AUROC swapped out.

How to read a flag rate
-----------------------
The column ``flag_means`` carries the interpretation row by row and the two
readings must not be averaged together.

* Where the true subgroup gap is **zero** (all simple-null geometries; all
  calibration metrics under case mix) a flag is a Type I error in the ordinary
  sense and the rate should sit at alpha.
* Where the true subgroup gap is **non-zero** (net benefit under case mix) a
  flag is not a Type I error at all -- the null is false -- but it is still a
  **false alarm about fairness**, because the model generating the data is the
  exact conditional probability and is unfair to nobody. That is the same
  reading the AUROC study's case-mix rows carry, and it is the reading that
  decides whether the manuscript's recommendation survives.
* Under the composite null the true calibration and net-benefit gaps are
  genuinely non-zero (a per-level monotone score map preserves AUROC but not
  calibration), so a flag there is a **true** positive and the column says so.

Monte-Carlo error
-----------------
Every rate carries ``flag_mc_se = sqrt(r(1-r)/n)``; every distribution summary
carries ``mc_se_mean``. At 1000 replicates a rate of 0.05 has SE 0.0069 and a
rate of 1.00 has SE 0. Nothing is reduced silently: ``--sims`` and ``--perm``
are recorded in every output row.

Entry point::

    python -m recompute.comparators.replacement_study
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recompute.comparators import ALPHA, SEED
from recompute.comparators.core import REPO
from recompute.comparators.replacement import (
    THRESHOLDS,
    FIXED_RULE_CUT,
    fixed_rule_flag,
    gap_over_partitions,
    level_metrics,
    metric_family,
    metric_names,
    permutation_pvalue,
    true_case_mix_metrics,
    true_gap,
    wald_maxt_pvalue,
)
from recompute.comparators.simulate import (
    GEOMETRIES,
    GEOMETRY_BY_NAME,
    Geometry,
    geometry_seed_word,
    true_subgroup_auc,
)
from recompute.comparators.type1 import HASHSEED, pin_hashseed_for_children

RESULTS = REPO / "recompute" / "results"
OUT_CSV = RESULTS / "replacement_metrics.csv"
CELL_DIR = RESULTS / "replacement_cells"

DEFAULT_SIMS = 1000
DEFAULT_PERM = 999
DEFAULT_RULES = ("m30", "ev10")

#: Metrics carried through the whole study. ``ece_null`` is not a metric anyone
#: would report; it is the finite-sample bias floor of ``ece`` on the same rows,
#: and it is tabulated so that no ECE number in this repository is ever quoted
#: without it.
ALL_METRICS: List[str] = metric_names() + ["ece", "ece_null", "prevalence"]

#: Reported as distributions only, never as something a test is run on.
#: ``ece_null`` is the finite-sample bias floor of ``ece`` and ``prevalence`` is
#: the mechanism behind the net-benefit result; neither is a fairness metric and
#: attaching a flag rate to either would invite it to be read as one.
DIAGNOSTIC_METRICS = {"ece_null", "prevalence"}

#: Which metrics the permutation null can be afforded on; the rest are covered
#: by the studentized Wald test. See :func:`.replacement.permutation_pvalue`.
PERM_METRICS: List[str] = (["mean_cal", "ece"]
                           + [f"nb_{t:.2f}" for t in THRESHOLDS]
                           + [f"snb_{t:.2f}" for t in THRESHOLDS])

#: Metrics with a usable per-level standard error, hence a Wald test.
WALD_METRICS: List[str] = ["citl", "mean_cal", "cal_slope"] + [
    f"{tag}_{t:.2f}" for tag in ("nb", "snb") for t in THRESHOLDS]

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.90, 0.95)


def _threshold_of(metric: str) -> float:
    if "_" in metric and metric[-1].isdigit():
        try:
            return float(metric.rsplit("_", 1)[1])
        except ValueError:
            return float("nan")
    return float("nan")


def _flag_reading(geom: Geometry, metric: str) -> str:
    """What a flag on this (geometry, metric) actually means."""
    if metric in DIAGNOSTIC_METRICS:
        return ("diagnostic quantity, not a fairness metric; reported as a "
                "distribution only and no test is run on it")
    tg = true_gap(geom, metric)
    if tg is None:
        return ("true positive (composite null preserves subgroup AUROC but "
                "NOT subgroup calibration or net benefit; the subgroups really "
                "do differ on this metric)")
    if tg == 0.0:
        if geom.is_case_mix:
            return ("false positive (true subgroup gap exactly 0: the score is "
                    "the exact conditional probability in every subgroup)")
        return "false positive (subgroups fully exchangeable; true gap 0)"
    return (f"false alarm about fairness (true subgroup gap {tg:.4f}, non-zero "
            "by case mix alone; the model is the exact DGP and is unfair to "
            "nobody)")


def _one_sim(geom: Geometry, rep: int, rule: str, n_perm: int, seed: int,
             alpha: float) -> Dict[str, object]:
    """Every replacement metric, gap and naive test on one simulated cohort."""
    from recompute.comparators.simulate import make_dataset

    y, s, codes = make_dataset(geom, rep, seed)
    # Separate stream for the ECE bias reference so that adding it cannot
    # perturb the permutation draws, which must match the AUROC study's.
    rng_ece = np.random.default_rng([seed, rep, 2])
    lv = {c: level_metrics(y, s, cc, rng_ece) for c, cc in codes.items()}

    out: Dict[str, object] = {}
    # Per-level values on partition p0, the column that carries the geometry.
    for k, x in enumerate(lv["p0"]):
        adm = x.admits(rule)
        for m in ALL_METRICS:
            v = (x.ece if m == "ece" else
                 x.ece_null if m == "ece_null" else x.metrics[m].value)
            out[f"lvl{k}__{m}"] = float(v) if adm else float("nan")

    gaps = {m: gap_over_partitions(lv, m, rule) for m in ALL_METRICS}
    for m, g in gaps.items():
        out[f"gap__{m}"] = g
        out[f"fixed__{m}"] = fixed_rule_flag(g, m)
    for m in WALD_METRICS:
        p = wald_maxt_pvalue(lv, m, rule)
        out[f"wald__{m}"] = float("nan") if not np.isfinite(p) else float(p < alpha)

    # Same generator seed the incumbent's permutation loop uses in
    # ``type1._one_sim``, so these p-values ride on the same permutation draws.
    rng = np.random.default_rng([seed, rep, 1])
    pvals = permutation_pvalue(y, s, codes, {m: gaps[m] for m in PERM_METRICS},
                               rule, n_perm, rng)
    for m, p in pvals.items():
        out[f"perm__{m}"] = float("nan") if not np.isfinite(p) else float(p < alpha)
    return out


def cell_path(geom_name: str, rule: str, n_sims: int, n_perm: int) -> Path:
    return CELL_DIR / f"{geom_name}__{rule}__{n_sims}_{n_perm}.json"


def _summarise(vals: np.ndarray) -> Dict[str, float]:
    v = vals[np.isfinite(vals)]
    n = int(v.size)
    if n == 0:
        return {"n_evaluable": 0, "mean": float("nan"), "mc_se_mean": float("nan"),
                "sd": float("nan"), "min": float("nan"), "max": float("nan"),
                **{f"q{int(q*100):02d}": float("nan") for q in _QUANTILES}}
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    return {
        "n_evaluable": n,
        "mean": float(v.mean()),
        "mc_se_mean": float(sd / np.sqrt(n)) if n > 1 else float("nan"),
        "sd": sd,
        "min": float(v.min()),
        "max": float(v.max()),
        **{f"q{int(q*100):02d}": float(np.quantile(v, q)) for q in _QUANTILES},
    }


def _run_cell(args) -> List[Dict[str, object]]:
    geom_name, rule, n_sims, n_perm, seed, alpha, force = args
    path = cell_path(geom_name, rule, n_sims, n_perm)
    if path.exists() and not force:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
            print(f"  [cached] {geom_name} / {rule}", flush=True)
            return rows
        except (ValueError, OSError):
            pass

    geom = GEOMETRY_BY_NAME[geom_name]
    t0 = time.perf_counter()
    sims = [_one_sim(geom, r, rule, n_perm, seed, alpha) for r in range(n_sims)]
    wall = time.perf_counter() - t0

    tsa = true_subgroup_auc(geom)
    tcm = true_case_mix_metrics(geom) if geom.is_case_mix else {}
    base: Dict[str, object] = {
        "geometry": geom.name,
        "description": geom.description,
        "null_family": geom.null_family,
        "n": geom.n,
        "prevalence": geom.prevalence,
        "n_partitions": len(geom.partitions),
        "max_levels": max(len(p) for p in geom.partitions),
        "min_level_frac": min(min(p) for p in geom.partitions),
        "case_mix_null": geom.is_case_mix,
        "composite_null": geom.is_composite,
        "monotone_transform": geom.transform if geom.is_composite else "",
        # Carried through so a reader can put the replacement metric's gap next
        # to the AUROC gap the manuscript indicts, in one row of one table.
        "true_auc_gap": float(tsa.get("max_gap", 0.0)) if geom.is_case_mix else 0.0,
        "rule": rule,
        "n_sims": n_sims,
        "n_perm": n_perm,
        "alpha": alpha,
        "seed": seed,
        "geometry_seed_word": geometry_seed_word(geom.name),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "<unset>"),
        "cell_wall_s": wall,
    }

    rows: List[Dict[str, object]] = []
    n_levels_p0 = len(geom.partitions[0])

    for m in ALL_METRICS:
        fam = metric_family(m)
        mbase = {**base, "metric": m, "metric_family": fam,
                 "threshold": _threshold_of(m),
                 "fixed_rule_cut": FIXED_RULE_CUT.get(fam, float("nan")),
                 "flag_means": _flag_reading(geom, m)}

        # ── per-subgroup rows (partition p0) ─────────────────────────────────
        for k in range(n_levels_p0):
            vals = np.array([sd.get(f"lvl{k}__{m}", np.nan) for sd in sims],
                            dtype=float)
            rows.append({**mbase, "scope": f"level_{k}",
                         "true_value": tcm.get(f"{m}__level_{k}", float("nan")),
                         "test": "", "test_note": "",
                         "n_flag": float("nan"), "flag_rate": float("nan"),
                         "flag_mc_se": float("nan"), **_summarise(vals)})

        # ── the max-over-subgroups gap ───────────────────────────────────────
        gaps = np.array([sd.get(f"gap__{m}", np.nan) for sd in sims], dtype=float)
        tg = true_gap(geom, m)
        gap_row = {**mbase, "scope": "gap",
                   "true_value": float("nan") if tg is None else float(tg),
                   "test": "", "test_note": "",
                   "n_flag": float("nan"), "flag_rate": float("nan"),
                   "flag_mc_se": float("nan"), **_summarise(gaps)}
        rows.append(gap_row)

        # ── the naive tests on that gap ──────────────────────────────────────
        if m in DIAGNOSTIC_METRICS:
            continue
        tests = [("fixed_threshold", f"gap >= {FIXED_RULE_CUT.get(fam, float('nan'))}"
                                     " (deterministic rule, no nominal level)",
                  f"fixed__{m}")]
        if m in WALD_METRICS:
            tests.append(("wald_maxt",
                          "max studentized pairwise contrast, two-sided normal, "
                          "Bonferroni over pairs, < alpha", f"wald__{m}"))
        if m in PERM_METRICS:
            tests.append(("permutation",
                          "max-statistic joint-permutation p-value < alpha "
                          f"(B={n_perm})", f"perm__{m}"))
        for tname, note, key in tests:
            v = np.array([sd.get(key, np.nan) for sd in sims], dtype=float)
            ok = np.isfinite(v)
            n_ok = int(ok.sum())
            rate = float(v[ok].mean()) if n_ok else float("nan")
            rows.append({
                **gap_row, "test": tname, "test_note": note,
                "n_flag": float(np.nansum(v)),
                "flag_rate": rate,
                "flag_mc_se": (float(np.sqrt(rate * (1 - rate) / n_ok))
                               if n_ok else float("nan")),
                "n_not_evaluable": int(n_sims - n_ok),
            })

    CELL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tmp.replace(path)
    print(f"  [done] {geom_name} / {rule}  ({wall/60:.1f} min)", flush=True)
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args(argv)
    pinned = pin_hashseed_for_children()

    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    names = ([g.strip() for g in args.only.split(",") if g.strip()]
             or [g.name for g in GEOMETRIES])
    jobs = [(nm, rule, args.sims, args.perm, args.seed, args.alpha, args.force)
            for nm in names for rule in rules]
    # Longest-first: cost is dominated by n and by the partition count.
    jobs.sort(key=lambda j: -(GEOMETRY_BY_NAME[j[0]].n
                              * len(GEOMETRY_BY_NAME[j[0]].partitions)))

    n_cached = sum(1 for j in jobs
                   if cell_path(j[0], j[1], j[2], j[3]).exists() and not args.force)
    print(f"Replacement-metric study: {len(names)} geometries x {len(rules)} "
          f"rules, {args.sims} sims, B={args.perm}, {args.jobs} worker(s); "
          f"{n_cached}/{len(jobs)} cells already checkpointed", flush=True)
    print(f"  seed={args.seed}  PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}  "
          f"this interpreter pinned: {pinned}", flush=True)

    t0 = time.perf_counter()
    rows: List[Dict[str, object]] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for res in ex.map(_run_cell, jobs):
                rows.extend(res)
    else:
        for j in jobs:
            rows.extend(_run_cell(j))

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows, "
          f"{(time.perf_counter()-t0)/60:.1f} min)")
    return 0


def _relaunch_pinned() -> Optional[int]:
    import subprocess

    if os.environ.get("_RISED_HASHSEED_REEXEC") == "1":
        return None
    env = dict(os.environ, PYTHONHASHSEED=HASHSEED, _RISED_HASHSEED_REEXEC="1")
    print(f"relaunching with PYTHONHASHSEED={HASHSEED}", flush=True)
    return subprocess.run(
        [sys.executable, "-m", "recompute.comparators.replacement_study",
         *sys.argv[1:]], env=env, cwd=str(REPO)).returncode


if __name__ == "__main__":
    _rc = (_relaunch_pinned() if sys.flags.hash_randomization != 0 else None)
    raise SystemExit(main() if _rc is None else _rc)
