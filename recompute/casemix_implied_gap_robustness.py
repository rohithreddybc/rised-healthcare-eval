"""The implied case-mix gap under every specification of the refit grid.

Why this module exists
----------------------
``recompute/comparators/sd_ratio_robustness.py`` refits every cohort under four
model classes and six seeds and recomputes ``rho_hat`` under each. That grid
showed the maximum ``rho_hat`` is a property of the fitted model, not of the
cohort: it runs from about 2.0 to about 6.4 across the 24 refits, so the
manuscript stopped quoting the published 3.304 as a point estimate.

The implied case-mix gap of ``recompute/casemix_implied_gap.py`` is computed
from the *same* fitted linear predictor, so it inherits the same exposure. It
was still being reported from the published fit alone. This module propagates
the refit grid through to it: the same closed form, evaluated on every
specification's own per-level score distribution.

Two evaluations of the same identity
------------------------------------
Proposition 1 is distribution free. For a well-specified model in level ``g``,

    AUROC_g = 1/2 + Delta_g / (4 pi_g (1 - pi_g)),

with ``pi_g = E[S]`` and ``Delta_g = E|S - S'|`` the Gini mean difference of the
level's risk distribution. Nothing about the shape of that distribution enters.
``recompute/results/casemix_theory.json`` verifies the identity to about 3e-15
across eight shapes including Laplace, lognormal, bimodal, uniform, t3 and
chi-square.

What *is* an assumption is the plug-in used to evaluate it. Two are computed
here for every level of every specification:

``gaussian``
    :func:`recompute.casemix_theory.auroc_gaussian_lp`, which recovers ``pi``
    and ``Delta`` from the level's linear-predictor mean and standard deviation
    alone, assuming that linear predictor is Gaussian. This is the published
    manuscript's evaluation. Two parameters are not a sufficient summary: the
    ``sd_counterexample`` block of ``casemix_theory.json`` gives AUROC 0.658 for
    a Gaussian linear predictor against 0.646 for a two-atom distribution and
    0.595 for spike-and-slab, all at the same SD. An error of that size per
    level is not negligible against implied gaps whose median is near 0.02.

``empirical``
    :func:`recompute.casemix_theory.auroc_from_risk`, which computes ``pi`` and
    ``Delta`` directly from the level's observed fitted scores. No shape is
    assumed, so this evaluation drops the only distributional assumption the
    quantity had. It still assumes well-specification, which is what makes the
    implied gap a model-based reference value rather than a decomposition.

Both are the same closed form. The difference between the two columns is
exactly the cost of the Gaussian plug-in.

What is written
---------------
``recompute/results/casemix_implied_gap_robustness.csv``
    One row per (cohort, rule, partition, spec_id). Carries the per-level
    geometry, both implied gaps, and the observed gap on that specification.

``recompute/results/casemix_implied_gap_robustness_summary.csv``
    One row per (rule, partition): the published-fit value, and the median, min
    and max of the implied gap across the 24 refits, for both evaluations.

Usage
-----
    python -m recompute.casemix_implied_gap_robustness
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from recompute.casemix_theory import auroc_from_risk, auroc_gaussian_lp
from recompute.comparators.cohort_casemix import RULES, _levels, linear_predictor
from recompute.comparators.core import (
    CLINICAL,
    COHORT_LABELS,
    COHORT_ORDER,
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
OUT_CSV = RESULTS / "casemix_implied_gap_robustness.csv"
SUMMARY_CSV = RESULTS / "casemix_implied_gap_robustness_summary.csv"

#: The reference admissibility rule used throughout the manuscript.
REFERENCE_RULE = "m30"

#: The 0.05 AUROC convention the manuscript counts partitions against.
CONVENTION = 0.05


def _max_min(values: Sequence[float]) -> float:
    """Max-minus-min over the finite entries, NaN if fewer than two remain."""
    finite = [float(v) for v in values if np.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    return float(max(finite) - min(finite))


def implied_gap_rows_for_fit(fit, subgroup_columns: Sequence[str]) -> List[Dict]:
    """One row per (rule, partition) for a single fitted specification.

    The linear predictor and the level admissibility predicate are the published
    module's own, imported rather than restated, so any difference between the
    published row and a refit row is the fit and nothing else. Both AUROC
    evaluations come from ``recompute.casemix_theory``.
    """
    y = np.asarray(fit.y_test).astype(int)
    s = np.asarray(fit.scores, dtype=float)
    lp = linear_predictor(s)
    codes_by_col = code_columns(fit.demo_test, list(subgroup_columns))

    rows: List[Dict] = []
    for rule in RULES:
        for col, codes in codes_by_col.items():
            lv = _levels(y, s, codes, rule)
            if len(lv) < 2:
                continue
            means = [float(np.mean(lp[m])) for _, m, *_ in lv]
            sds = [float(np.std(lp[m], ddof=1)) for _, m, *_ in lv]
            gaussian = [auroc_gaussian_lp(mu, sd) for mu, sd in zip(means, sds)]
            empirical = [auroc_from_risk(s[m]) for _, m, *_ in lv]
            aucs = [auc_delong(y[m], s[m])[0] for _, m, *_ in lv]
            lo, hi = min(sds), max(sds)

            rows.append({
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
                "n_test": fit.n_test,
                "n_levels_admissible": len(lv),
                "partition_sd_ratio": (hi / lo if lo > 0 else float("inf")),
                "min_level_lp_sd": lo,
                "max_level_lp_sd": hi,
                "min_level_lp_mean": min(means),
                "max_level_lp_mean": max(means),
                "level_lp_mean_spread": max(means) - min(means),
                "implied_auroc_min_gaussian": min(gaussian),
                "implied_auroc_max_gaussian": max(gaussian),
                "implied_casemix_gap_gaussian": _max_min(gaussian),
                "implied_auroc_min_empirical": min(empirical),
                "implied_auroc_max_empirical": max(empirical),
                "implied_casemix_gap_empirical": _max_min(empirical),
                "observed_auc_gap": max(aucs) - min(aucs),
            })
    return rows


def run(cohorts: Sequence[str] = tuple(COHORT_ORDER),
        seeds: Sequence[int] = SEEDS,
        classes: Sequence[str] = tuple(MODEL_CLASSES),
        verbose: bool = True) -> pd.DataFrame:
    specs = iter_specs(seeds, classes)
    rows: List[Dict] = []
    for name in cohorts:
        t0 = time.perf_counter()
        fc = build_full_cohort(name)
        if verbose:
            print(f"{name:14s} n={fc.X.shape[0]:7d} p={fc.X.shape[1]:3d} "
                  f"partitions={len(fc.subgroup_columns)}", flush=True)
        for mc, seed in specs:
            f = (published_fit(fc) if mc == PUBLISHED
                 else fit_spec(fc, mc, int(seed)))
            n_before = len(rows)
            rows += implied_gap_rows_for_fit(f, fc.subgroup_columns)
            if verbose:
                tag = mc if seed is None else f"{mc} s{seed}"
                print(f"    {tag:24s} rows={len(rows) - n_before:3d}", flush=True)
        if verbose:
            print(f"  -> {name} done in {time.perf_counter() - t0:.1f}s",
                  flush=True)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, rule: str = REFERENCE_RULE,
              clinical_only: bool = True) -> pd.DataFrame:
    """Published value, plus median/min/max across the refits, per partition."""
    d = df[df["rule"] == rule]
    if clinical_only:
        d = d[d["is_clinical"]]
    refits = d[d["spec_id"] != PUBLISHED]
    pub = d[d["spec_id"] == PUBLISHED].set_index("partition_key")

    out: List[Dict] = []
    for key, g in refits.groupby("partition_key"):
        rec: Dict = {
            "partition_key": key,
            "cohort_label": g["cohort_label"].iloc[0],
            "partition": g["partition"].iloc[0],
            "n_specs_measured": int(len(g)),
        }
        for tag in ("gaussian", "empirical"):
            col = f"implied_casemix_gap_{tag}"
            v = g[col].dropna()
            rec[f"published_{tag}"] = (float(pub.loc[key, col])
                                       if key in pub.index else float("nan"))
            rec[f"median_{tag}"] = float(v.median())
            rec[f"min_{tag}"] = float(v.min())
            rec[f"max_{tag}"] = float(v.max())
        rec["published_sd_ratio"] = (float(pub.loc[key, "partition_sd_ratio"])
                                     if key in pub.index else float("nan"))
        rec["min_sd_ratio"] = float(g["partition_sd_ratio"].min())
        rec["max_sd_ratio"] = float(g["partition_sd_ratio"].max())
        out.append(rec)
    res = pd.DataFrame(out).sort_values("published_gaussian")
    return res.reset_index(drop=True)


def headline_stability(df: pd.DataFrame, rule: str = REFERENCE_RULE,
                       tag: str = "gaussian") -> pd.DataFrame:
    """Per-specification median, maximum and 0.05 count over the partitions.

    One row per specification. ``n_partitions`` is carried because a partition
    can lose admissibility under a refit, and a count of partitions reaching the
    convention is not comparable across specifications without it.
    """
    col = f"implied_casemix_gap_{tag}"
    d = df[(df["rule"] == rule) & df["is_clinical"]]
    out: List[Dict] = []
    for spec, g in d.groupby("spec_id"):
        v = g[col].dropna()
        out.append({
            "spec_id": spec,
            "n_partitions": int(len(v)),
            "median": float(v.median()),
            "max": float(v.max()),
            "min": float(v.min()),
            f"n_at_{CONVENTION}": int((v >= CONVENTION).sum()),
        })
    res = pd.DataFrame(out)
    res["_pub"] = res["spec_id"] != PUBLISHED
    return res.sort_values(["_pub", "spec_id"]).drop(columns="_pub").reset_index(
        drop=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", type=str, default="",
                    help="comma-separated cohort names")
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args(argv)

    names = ([c.strip() for c in args.only.split(",") if c.strip()]
             or [c for c in COHORT_ORDER if c in CLINICAL])

    t0 = time.perf_counter()
    df = run(names)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    df.to_csv(out, index=False)
    summarise(df).to_csv(SUMMARY_CSV, index=False)
    print(f"wrote {out} ({len(df)} rows) in {time.perf_counter() - t0:.1f}s")
    print(f"wrote {SUMMARY_CSV}")

    for tag in ("gaussian", "empirical"):
        print(f"\n=== {tag} plug-in, rule {REFERENCE_RULE}, clinical ===")
        print(headline_stability(df, tag=tag).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
