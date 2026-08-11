"""
Driver for the round-3 case-mix work.

Produces
--------
``recompute/results/casemix_sweep.csv``
    Flag rate as a function of the per-level linear-predictor SD ratio, for every
    procedure, on a 16-point grid from 1.0 (an exact null) to 3.167 (the round-2
    "strong" geometry), run twice: once with level prevalence free to move with
    the spread as it does throughout round 2, and once with every level's
    prevalence pinned at 0.20 so the curve is a function of spread alone.

``recompute/results/casemix_positive_control.csv``
    Two row types. ``row_type="cell"`` is the flag rate of every procedure on
    each positive-control geometry -- genuine subgroup-specific unfairness, in
    three forms. ``row_type="matched_pair_discrimination"`` is the answer to the
    question the manuscript asserts: for each matched pair (identical true
    per-level AUROC, identical level prevalence, one arm case mix and one arm
    unfairness), the AUC with which each procedure's own statistic separates the
    two arms. 0.5 means the procedure carries no information about which
    mechanism produced the gap.

Everything is checkpointed per ``(geometry, rule)`` cell in
``recompute/results/round3_cells/``, so an interrupted run loses at most one cell
and re-invoking the same command resumes. Replicate counts are the study's:
1,000 simulated datasets and B=999 permutations, never reduced. Monte-Carlo
standard errors are in every row.

Usage
-----
    python -m recompute.comparators.run_round3 --jobs 10
    python -m recompute.comparators.run_round3 --retune     # re-solve the pairs
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from recompute.comparators import ALPHA, SEED
from recompute.comparators.casemix_grid import (
    MATCHED_PAIRS,
    POSITIVE_CONTROL_GEOMETRIES,
    SD_RATIO_GRID,
    SWEEP_GEOMETRIES,
    SWEEP_PREVFIXED_GEOMETRIES,
    solve_matched_pair,
)
from recompute.comparators.core import REPO
from recompute.comparators.round3_sim import (
    DIAGNOSTICS,
    METHODS,
    STATISTIC_SIGN,
    cell_path,
    discrimination_auc,
    run_cell,
)
from recompute.comparators.type1 import HASHSEED, pin_hashseed_for_children

RESULTS = REPO / "recompute" / "results"
SWEEP_CSV = RESULTS / "casemix_sweep.csv"
PC_CSV = RESULTS / "casemix_positive_control.csv"

DEFAULT_SIMS = 1000
DEFAULT_PERM = 999

#: Which of the reported rows are the five procedures under study and which are
#: the two calibration diagnostics added in round 3. The distinction is a column
#: rather than a footnote because the diagnostics are the only things that
#: discriminate, and a reader must not mistake them for a validated recommendation.
IS_PROCEDURE = {m: True for m in METHODS}
IS_PROCEDURE.update({d: False for d in DIAGNOSTICS})


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return "/".join("nan" if x is None or not np.isfinite(x) else f"{x:.6g}"
                        for x in v)
    return str(v)


def _cell_rows(cell: Dict[str, object], sweep_family: str) -> List[Dict]:
    rows = []
    for m, r in cell["rates"].items():
        rows.append({
            "row_type": "cell",
            "family": sweep_family,
            "geometry": cell["geometry"],
            "sd_ratio": cell["sd_ratio"],
            "equalize_prevalence": cell["equalize_prevalence"],
            "lp_dist": cell["lp_dist"],
            "n": cell["n"],
            "prevalence": cell["prevalence"],
            "n_levels": cell["n_levels"],
            "scales": _fmt(cell["scales"]),
            "unfair_w": _fmt(cell["unfair_w"]),
            "miscal_slope": _fmt(cell["miscal_slope"]),
            "miscal_intercept": _fmt(cell["miscal_intercept"]),
            "true_auc_by_level": _fmt(cell["true_auc_by_level"]),
            "oracle_auc_by_level": _fmt(cell["oracle_auc_by_level"]),
            "prevalence_by_level": _fmt(cell["prevalence_by_level"]),
            "true_auc_gap": cell["true_auc_gap"],
            "true_max_excess_auc": cell["true_max_excess_auc"],
            "prevalence_ratio": cell["prevalence_ratio"],
            "rule": cell["rule"],
            "method": m,
            "is_one_of_the_five": IS_PROCEDURE.get(m, False),
            "n_sims": cell["n_sims"],
            "n_perm": cell["n_perm"],
            "alpha": cell["alpha"],
            "seed": cell["seed"],
            "geometry_seed_word": cell["geometry_seed_word"],
            "pythonhashseed": cell["pythonhashseed"],
            "n_evaluable": r["n_evaluable"],
            "n_flag": r["n_flag"],
            "flag_rate": r["flag_rate"],
            "flag_rate_mc_se": r["mc_se"],
            "mean_n_admissible_p0": cell["mean_n_admissible_p0"],
            "frac_reps_with_level_dropped": cell["frac_reps_with_level_dropped"],
            "description": cell["description"],
        })
    return rows


def _discrimination_rows(cells: Dict[str, Dict]) -> List[Dict]:
    """Matched-pair discrimination: can any statistic tell the arms apart?"""
    rows = []
    for tag, par in MATCHED_PAIRS.items():
        a = cells.get(f"pc_casemix_{tag}")
        b = cells.get(f"pc_unfair_{tag}")
        if a is None or b is None:
            continue
        for stat, sign in STATISTIC_SIGN.items():
            va = np.asarray(a["statistics"][stat], dtype=float) * sign
            vb = np.asarray(b["statistics"][stat], dtype=float) * sign
            auc, se = discrimination_auc(va, vb)
            n_a = int(np.isfinite(va).sum())
            n_b = int(np.isfinite(vb).sum())
            rows.append({
                "row_type": "matched_pair_discrimination",
                "family": "positive_control",
                "geometry": f"pc_{tag}",
                "pair_tag": tag,
                "true_auc_gap": par["delta"],
                "casemix_arm": a["geometry"],
                "unfair_arm": b["geometry"],
                "casemix_true_auc_by_level": _fmt(a["true_auc_by_level"]),
                "unfair_true_auc_by_level": _fmt(b["true_auc_by_level"]),
                "casemix_prevalence_by_level": _fmt(a["prevalence_by_level"]),
                "unfair_prevalence_by_level": _fmt(b["prevalence_by_level"]),
                "casemix_true_max_excess_auc": a["true_max_excess_auc"],
                "unfair_true_max_excess_auc": b["true_max_excess_auc"],
                "rule": a["rule"],
                "method": stat,
                "is_one_of_the_five": stat in (
                    "permutation_null__p", "diciccio2020__p", "lum2022__p",
                    "maxmin_gap", "four_fifths_ratio"),
                "statistic_direction": "higher means more evidence of a problem",
                "n_sims": a["n_sims"],
                "n_perm": a["n_perm"],
                "n_evaluable_casemix": n_a,
                "n_evaluable_unfair": n_b,
                "discrimination_auc": auc,
                "discrimination_auc_se": se,
                "discrimination_auc_lo95": auc - 1.96 * se
                if np.isfinite(se) else np.nan,
                "discrimination_auc_hi95": auc + 1.96 * se
                if np.isfinite(se) else np.nan,
                "casemix_flag_rate": a["rates"].get(
                    _flag_key(stat), {}).get("flag_rate", np.nan),
                "unfair_flag_rate": b["rates"].get(
                    _flag_key(stat), {}).get("flag_rate", np.nan),
            })
    return rows


def _flag_key(stat: str) -> str:
    """The flag-rate row corresponding to a per-replicate statistic."""
    return {
        "permutation_null__p": "permutation_null",
        "diciccio2020__p": "diciccio2020",
        "lum2022__p": "lum2022",
        "maxmin_gap": "fixed_threshold_005",
        "four_fifths_ratio": "four_fifths",
        "calibration_cox__p": "calibration_cox",
        "mbc_excess": "mbc_excess",
    }.get(stat, stat)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sims", type=int, default=DEFAULT_SIMS)
    ap.add_argument("--perm", type=int, default=DEFAULT_PERM)
    ap.add_argument("--rule", type=str, default="m30")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated geometry names")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--retune", action="store_true",
                    help="re-solve the matched-pair parameters and print them")
    args = ap.parse_args(argv)

    if args.retune:
        for tag, par in MATCHED_PAIRS.items():
            got = solve_matched_pair(par["delta"])
            print(f'    "{tag}": {{"delta": {par["delta"]}, '
                  f'"casemix_scale0": {got["casemix_scale0"]:.10f}, '
                  f'"unfair_w0": {got["unfair_w0"]:.10f}}},')
            print(f"      # realised gaps: case-mix {got['casemix_gap']:.10f}, "
                  f"unfair {got['unfair_gap']:.10f}", flush=True)
        return 0

    pinned = pin_hashseed_for_children()
    geoms = (SWEEP_GEOMETRIES + SWEEP_PREVFIXED_GEOMETRIES
             + POSITIVE_CONTROL_GEOMETRIES)
    if args.only:
        want = {g.strip() for g in args.only.split(",") if g.strip()}
        geoms = [g for g in geoms if g.name in want]

    jobs = [(g, args.rule, args.sims, args.perm, args.seed, args.alpha,
             args.force) for g in geoms]
    n_cached = sum(1 for j in jobs
                   if cell_path(j[0].name, j[1], j[2], j[3]).exists()
                   and not args.force)
    print(f"round-3 case-mix study: {len(jobs)} cells, rule={args.rule}, "
          f"{args.sims} sims, B={args.perm}, {args.jobs} worker(s); "
          f"{n_cached}/{len(jobs)} already checkpointed", flush=True)
    print(f"  seed={args.seed}  PYTHONHASHSEED={os.environ['PYTHONHASHSEED']}  "
          f"this interpreter pinned: {pinned}", flush=True)

    t0 = time.perf_counter()
    cells: List[Dict] = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for res in ex.map(run_cell, jobs):
                cells.append(res)
    else:
        cells = [run_cell(j) for j in jobs]

    by_name = {c["geometry"]: c for c in cells}
    sweep_names = {g.name for g in SWEEP_GEOMETRIES}
    pf_names = {g.name for g in SWEEP_PREVFIXED_GEOMETRIES}

    sweep_rows: List[Dict] = []
    pc_rows: List[Dict] = []
    for c in cells:
        if c["geometry"] in sweep_names:
            sweep_rows += _cell_rows(c, "sd_ratio_sweep")
        elif c["geometry"] in pf_names:
            sweep_rows += _cell_rows(c, "sd_ratio_sweep_prevalence_fixed")
        else:
            pc_rows += _cell_rows(c, "positive_control")
    pc_rows += _discrimination_rows(by_name)

    RESULTS.mkdir(parents=True, exist_ok=True)
    if sweep_rows:
        df = pd.DataFrame(sweep_rows).sort_values(
            ["family", "sd_ratio", "method"])
        df.to_csv(SWEEP_CSV, index=False)
        print(f"\nwrote {SWEEP_CSV}  ({len(df)} rows)")
    if pc_rows:
        df = pd.DataFrame(pc_rows)
        df.to_csv(PC_CSV, index=False)
        print(f"wrote {PC_CSV}  ({len(df)} rows)")
    print(f"total {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    if sys.flags.hash_randomization != 0 and os.environ.get(
            "_RISED_HASHSEED_REEXEC") != "1":
        import subprocess

        env = dict(os.environ, PYTHONHASHSEED=HASHSEED,
                   _RISED_HASHSEED_REEXEC="1")
        raise SystemExit(subprocess.run(
            [sys.executable, "-m", "recompute.comparators.run_round3",
             *sys.argv[1:]], env=env, cwd=str(REPO)).returncode)
    raise SystemExit(main())
