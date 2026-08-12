"""
Is rho-hat a property of the cohort, or of the model that was fitted to it?

The manuscript's empirical anchor is rho-hat, the ratio of the largest to the
smallest per-level standard deviation of the linear predictor, measured across
21 clinical demographic partitions: median 1.145, min 1.022, max 3.304. Those
numbers come from ``recompute/results/cohort_sd_ratios.csv``, which was computed
from **one** fitted model per cohort -- fixed hyperparameters, seed 42, one
train/test split.

The linear predictor is a property of the fitted model. So rho-hat could move
under a different model class or a different split, and the manuscript's
headline, its age-versus-sex ordering claim, and its induced-false-flag-rate
distribution all rest on it. This module refits every cohort under four model
classes and six seeds (24 refits per cohort, plus the published fit carried
alongside for traceability) and recomputes rho-hat for every partition under
every one.

What is written
---------------
``recompute/results/sd_ratio_robustness.csv``: one row per
(cohort, rule, partition, model_class, seed). ``partition_sd_ratio`` is computed
by the *same* code path as the published table -- ``linear_predictor`` and the
level admissibility predicate are imported from
``recompute.comparators.cohort_casemix``, not reimplemented -- so any difference
between the published number and a refit is the fit and nothing else.

Two mapped quantities travel with each row:

``induced_flag_rate_<method>``
    The case-mix false-alarm rate this rho-hat implies, read off the existing
    ``recompute/results/casemix_sweep.csv`` curve by linear interpolation in
    the SD ratio. This is the quantity the manuscript's "median roughly double
    nominal" claim is about.
``frac_lp_clipped``
    The fraction of test rows whose predicted probability hit the 1e-12 clip in
    ``linear_predictor``. Random forests can return exactly 0 or 1 for a leaf;
    where this is non-zero the level's linear-predictor SD is bounded by the
    clip rather than by the model, and the row is flagged.

Deliberate non-choices
----------------------
No model configuration was tuned toward the published numbers. The four classes
are declared once in ``recompute.refit`` with off-the-shelf settings and applied
unchanged to all ten cohorts. Nothing is dropped for disagreeing.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from recompute.comparators.cohort_casemix import RULES, _levels, linear_predictor
from recompute.comparators.core import (
    COHORT_LABELS,
    COHORT_ORDER,
    CLINICAL,
    REPO,
    auc_delong,
)
from recompute.null_reference import code_columns
from recompute.refit import (
    MODEL_CLASSES,
    PUBLISHED,
    PUBLISHED_CLASS,
    SEEDS,
    build_full_cohort,
    fit_spec,
    iter_specs,
    published_fit,
)

RESULTS = REPO / "recompute" / "results"
OUT_CSV = RESULTS / "sd_ratio_robustness.csv"
SWEEP_CSV = RESULTS / "casemix_sweep.csv"

#: Clip used by :func:`linear_predictor`; a score at the clip has its logit
#: bounded by the clip rather than by the model.
_CLIP = 1e-12

#: Methods whose induced false-flag rate is carried on every row. All are
#: "one of the five" on the sweep except ``calibration_cox``; ``permutation_null``
#: is the incumbent, i.e. the manuscript's own procedure.
SWEEP_METHODS: Tuple[str, ...] = (
    "permutation_null", "diciccio2020", "lum2022", "fixed_threshold_005",
    "four_fifths",
)

#: The sweep family to read the curve from. ``sd_ratio_sweep`` is the main
#: sweep, in which level prevalence moves with spread by construction, and is
#: the one the published claim is stated on. ``sd_ratio_sweep_prevalence_fixed``
#: is carried as a sensitivity.
SWEEP_FAMILY = "sd_ratio_sweep"


# ── the sweep curve ──────────────────────────────────────────────────────────
def load_sweep_curves(path: Path = SWEEP_CSV, family: str = SWEEP_FAMILY
                      ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``method -> (sd_ratio, flag_rate, flag_rate_mc_se)``, ascending in ratio."""
    df = pd.read_csv(path)
    df = df[df["family"] == family]
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for m, g in df.groupby("method"):
        g = g.dropna(subset=["flag_rate"]).sort_values("sd_ratio")
        if g.empty:
            continue
        out[str(m)] = (g["sd_ratio"].to_numpy(dtype=float),
                       g["flag_rate"].to_numpy(dtype=float),
                       g["flag_rate_mc_se"].to_numpy(dtype=float))
    return out


def induced_flag_rate(curve: Tuple[np.ndarray, np.ndarray, np.ndarray],
                      ratio: float) -> Tuple[float, float, bool]:
    """Interpolate a flag rate, and its Monte-Carlo SE, at one SD ratio.

    Returns ``(rate, mc_se, extrapolated)``. The sweep grid stops at 3.167; a
    rho-hat above that is clamped to the endpoint and flagged, because the
    curve carries no information beyond its last node. ``mc_se`` is the linear
    interpolation of the two neighbouring cells' binomial Monte-Carlo SEs --
    the resampling noise the sweep itself has at 1,000 simulations -- and is
    the floor on how finely two induced rates can be told apart.
    """
    x, f, se = curve
    r = float(ratio)
    extrap = bool(r < x[0] or r > x[-1])
    return (float(np.interp(r, x, f)), float(np.interp(r, x, se)), extrap)


# ── rho-hat for one fit ──────────────────────────────────────────────────────
def sd_ratio_rows_for_fit(fit, subgroup_columns: Sequence[str],
                          curves: Dict[str, Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]],
                          ) -> List[Dict]:
    """One row per (rule, partition) for a single fitted specification.

    The linear predictor, the level admissibility predicate and the SD-ratio
    arithmetic are the published module's own, imported rather than restated.
    """
    y = np.asarray(fit.y_test).astype(int)
    s = np.asarray(fit.scores, dtype=float)
    lp = linear_predictor(s)
    frac_clipped = float(np.mean((s <= _CLIP) | (s >= 1.0 - _CLIP)))
    codes_by_col = code_columns(fit.demo_test, list(subgroup_columns))

    rows: List[Dict] = []
    for rule in RULES:
        for col, codes in codes_by_col.items():
            lv = _levels(y, s, codes, rule)
            if len(lv) < 2:
                continue
            sds = {k: float(np.std(lp[m], ddof=1)) for k, m, *_ in lv}
            iqrs = {k: float(np.subtract(*np.percentile(lp[m], [75, 25])))
                    for k, m, *_ in lv}
            lo, hi = min(sds.values()), max(sds.values())
            ratio = hi / lo if lo > 0 else float("inf")
            iqr_lo, iqr_hi = min(iqrs.values()), max(iqrs.values())
            aucs = {k: auc_delong(y[m], s[m])[0] for k, m, *_ in lv}

            row = {
                "cohort": fit.cohort,
                "cohort_label": COHORT_LABELS.get(fit.cohort, fit.cohort),
                "is_clinical": fit.cohort in CLINICAL,
                "rule": rule,
                "partition": col,
                "partition_key": f"{fit.cohort}|{col}",
                "model_class": fit.model_class,
                "is_published_class": (
                    fit.model_class == PUBLISHED_CLASS.get(fit.cohort)),
                "seed": fit.seed if fit.seed is not None else "",
                "spec_id": (PUBLISHED if fit.model_class == PUBLISHED
                            else f"{fit.model_class}|s{fit.seed}"),
                "n_train": fit.n_train,
                "n_test": fit.n_test,
                "train_prevalence": fit.train_prevalence,
                "test_prevalence": fit.test_prevalence,
                "n_levels_admissible": len(lv),
                "partition_sd_ratio": ratio,
                "partition_iqr_ratio": (iqr_hi / iqr_lo if iqr_lo > 0
                                        else float("inf")),
                "partition_min_lp_sd": lo,
                "partition_max_lp_sd": hi,
                "partition_observed_auc_gap": (max(aucs.values())
                                               - min(aucs.values())),
                "partition_level_prevalence_ratio": (
                    max(n_pos / n for _, _, n, n_pos, _ in lv)
                    / min(n_pos / n for _, _, n, n_pos, _ in lv)),
                "overall_auc": auc_delong(y, s)[0],
                "frac_lp_clipped": frac_clipped,
                "fit_runtime_s": fit.fit_runtime_s,
            }
            for m in SWEEP_METHODS:
                if m not in curves:
                    continue
                rate, se, extrap = induced_flag_rate(curves[m], ratio)
                row[f"induced_flag_rate_{m}"] = rate
                row[f"induced_flag_rate_mc_se_{m}"] = se
                row[f"induced_flag_rate_extrapolated_{m}"] = extrap
            rows.append(row)
    return rows


# ── driver ───────────────────────────────────────────────────────────────────
def run(cohorts: Sequence[str] = tuple(COHORT_ORDER),
        seeds: Sequence[int] = SEEDS,
        classes: Sequence[str] = tuple(MODEL_CLASSES),
        verbose: bool = True) -> pd.DataFrame:
    curves = load_sweep_curves()
    specs = iter_specs(seeds, classes)
    rows: List[Dict] = []
    for name in cohorts:
        t0 = time.perf_counter()
        fc = build_full_cohort(name)
        if verbose:
            print(f"{name:14s} n={fc.X.shape[0]:7d} p={fc.X.shape[1]:3d} "
                  f"load={fc.load_runtime_s:6.1f}s "
                  f"partitions={len(fc.subgroup_columns)}", flush=True)
        for mc, seed in specs:
            f = (published_fit(fc) if mc == PUBLISHED
                 else fit_spec(fc, mc, int(seed)))
            n_before = len(rows)
            rows += sd_ratio_rows_for_fit(f, fc.subgroup_columns, curves)
            if verbose:
                tag = mc if seed is None else f"{mc} s{seed}"
                print(f"    {tag:24s} fit={f.fit_runtime_s:6.1f}s "
                      f"rows={len(rows) - n_before:3d} "
                      f"clip={np.mean((f.scores <= _CLIP) | (f.scores >= 1 - _CLIP)):.4f}",
                      flush=True)
        if verbose:
            print(f"  -> {name} done in {time.perf_counter() - t0:.1f}s",
                  flush=True)
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated cohort names")
    ap.add_argument("--classes", type=str, default="",
                    help="comma-separated model classes")
    ap.add_argument("--seeds", type=str, default="",
                    help="comma-separated seeds")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args(argv)

    names = ([c.strip() for c in args.only.split(",") if c.strip()]
             or list(COHORT_ORDER))
    classes = ([c.strip() for c in args.classes.split(",") if c.strip()]
               or list(MODEL_CLASSES))
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             or list(SEEDS))

    t0 = time.perf_counter()
    df = run(names, seeds, classes)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows) in {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
