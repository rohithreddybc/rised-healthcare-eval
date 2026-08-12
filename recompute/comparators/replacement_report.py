"""
Render ``replacement_metrics.csv`` into the tables of
``docs/replacement_metrics_evaluation.md``.

Every number in the report is produced here from the CSV. Nothing is typed by
hand, so the report cannot drift from the run that backs it: re-running the study
and re-running this module is the whole update path.

    python -m recompute.comparators.replacement_report > docs/replacement_metrics_evaluation.tables.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from recompute.comparators.core import REPO
from recompute.comparators.replacement import THRESHOLDS
from recompute.comparators.simulate import GEOMETRY_BY_NAME, true_subgroup_auc

CSV = REPO / "recompute" / "results" / "replacement_metrics.csv"

CASE_MIX_ORDER = ["casemix_location_3", "casemix_mild_3", "casemix_moderate_3",
                  "casemix_moderate_3_n10000", "casemix_moderate_3part",
                  "casemix_strong_4"]

METRIC_LABEL = {
    "citl": "Calibration-in-the-large",
    "mean_cal": "Mean calibration (O-E)",
    "cal_slope": "Calibration slope",
    "ece": "ECE (10 bins)",
    "ece_null": "  ECE bias floor",
    "prevalence": "Subgroup prevalence",
}
for _t in THRESHOLDS:
    METRIC_LABEL[f"nb_{_t:.2f}"] = f"Net benefit, t={_t:.2f}"
    METRIC_LABEL[f"snb_{_t:.2f}"] = f"Standardised NB, t={_t:.2f}"

CAL_METRICS = ["citl", "mean_cal", "cal_slope", "ece"]
NB_METRICS = [f"snb_{t:.2f}" for t in THRESHOLDS]
NB_RAW = [f"nb_{t:.2f}" for t in THRESHOLDS]


def _fmt(x: float, d: int = 3) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def _rate(row) -> str:
    """A flag rate with its Monte-Carlo standard error."""
    if row is None or not np.isfinite(row["flag_rate"]):
        return "--"
    return f"{row['flag_rate']:.3f} ({row['flag_mc_se']:.3f})"


def _pick(df: pd.DataFrame, **kw) -> Optional[pd.Series]:
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= (df[k] == v)
    sub = df[m]
    return None if sub.empty else sub.iloc[0]


def table_true_gaps(df: pd.DataFrame) -> str:
    """The exact population truth. No simulation, no Monte-Carlo error."""
    out = ["| Geometry | n | true AUROC gap | true prevalence gap | "
           + " | ".join(f"true sNB gap t={t:.2f}" for t in THRESHOLDS)
           + " | true calibration gap |",
           "|---|---|---|---|" + "---|" * (len(THRESHOLDS) + 1)]
    for g in CASE_MIX_ORDER:
        geom = GEOMETRY_BY_NAME[g]
        sub = df[(df.geometry == g) & (df.scope == "gap")]
        auc_gap = true_subgroup_auc(geom)["max_gap"]
        prev = _pick(sub, metric="prevalence")
        cells = []
        for t in THRESHOLDS:
            r = _pick(sub, metric=f"snb_{t:.2f}")
            cells.append(_fmt(r["true_value"]) if r is not None else "--")
        # Measured prevalence gap stands in for the true one where the CSV
        # carries it; both are exact for these geometries.
        prev_true = (prev["true_value"] if prev is not None
                     and np.isfinite(prev["true_value"]) else np.nan)
        out.append(f"| `{g}` | {geom.n} | {_fmt(auc_gap)} | {_fmt(prev_true)} | "
                   + " | ".join(cells) + " | 0.000 (exact) |")
    return "\n".join(out)


def table_geometry(df: pd.DataFrame, geom_name: str, rule: str,
                   metrics: List[str]) -> str:
    sub = df[(df.geometry == geom_name) & (df.rule == rule)]
    out = ["| Metric | true gap | mean gap (MC SE) | SD | q95 | "
           "fixed cut | Wald max-T | permutation |",
           "|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        gap = _pick(sub, metric=m, scope="gap", test=np.nan)
        if gap is None:
            g2 = sub[(sub.metric == m) & (sub.scope == "gap")]
            if g2.empty:
                continue
            gap = g2.iloc[0]
        tv = gap["true_value"]
        tests = {t: _pick(sub, metric=m, scope="gap", test=t)
                 for t in ("fixed_threshold", "wald_maxt", "permutation")}
        out.append(
            f"| {METRIC_LABEL.get(m, m)} | "
            f"{'n/a' if not np.isfinite(tv) else _fmt(tv)} | "
            f"{_fmt(gap['mean'])} ({_fmt(gap['mc_se_mean'], 4)}) | "
            f"{_fmt(gap['sd'])} | {_fmt(gap['q95'])} | "
            f"{_rate(tests['fixed_threshold'])} | {_rate(tests['wald_maxt'])} | "
            f"{_rate(tests['permutation'])} |")
    return "\n".join(out)


def table_levels(df: pd.DataFrame, geom_name: str, rule: str,
                 metrics: List[str]) -> str:
    """Per-subgroup values: what each level actually shows, against its truth."""
    sub = df[(df.geometry == geom_name) & (df.rule == rule)]
    levels = sorted({s for s in sub.scope.unique() if str(s).startswith("level_")},
                    key=lambda z: int(z.split("_")[1]))
    out = ["| Metric | " + " | ".join(f"level {i.split('_')[1]} (true)"
                                      for i in levels) + " |",
           "|---|" + "---|" * len(levels)]
    for m in metrics:
        cells = []
        for lv in levels:
            r = _pick(sub, metric=m, scope=lv)
            if r is None:
                cells.append("--")
                continue
            tv = r["true_value"]
            cells.append(f"{_fmt(r['mean'])} ({_fmt(tv) if np.isfinite(tv) else 'n/a'})")
        out.append(f"| {METRIC_LABEL.get(m, m)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_flagrate_matrix(df: pd.DataFrame, geometries: List[str], rule: str,
                          metrics: List[str], test: str) -> str:
    """One test's flag rate, every geometry x metric, in a single grid."""
    out = ["| Geometry | " + " | ".join(METRIC_LABEL.get(m, m) for m in metrics)
           + " |", "|---|" + "---|" * len(metrics)]
    for g in geometries:
        sub = df[(df.geometry == g) & (df.rule == rule)]
        cells = [_rate(_pick(sub, metric=m, scope="gap", test=test))
                 for m in metrics]
        out.append(f"| `{g}` | " + " | ".join(cells) + " |")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(CSV))
    ap.add_argument("--rule", default="m30")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    df["test"] = df["test"].fillna("")
    rule = args.rule
    simple = sorted(df[df.null_family == "simple"].geometry.unique())
    composite = sorted(df[df.null_family == "composite"].geometry.unique())
    w = sys.stdout.write

    w(f"<!-- generated by recompute.comparators.replacement_report "
      f"(rule={rule}) -->\n\n")
    w("### T1. Exact population truth under case mix (quadrature; no MC error)\n\n")
    w(table_true_gaps(df) + "\n\n")

    for g in CASE_MIX_ORDER:
        w(f"### T2.{CASE_MIX_ORDER.index(g)+1} `{g}` (rule {rule})\n\n")
        w(table_geometry(df, g, rule,
                         CAL_METRICS + ["ece_null"] + NB_RAW + NB_METRICS
                         + ["prevalence"]) + "\n\n")

    w(f"### T3. Per-subgroup values, `casemix_moderate_3` (rule {rule}); "
      "mean over 1000 sims, exact truth in brackets\n\n")
    w(table_levels(df, "casemix_moderate_3", rule,
                   CAL_METRICS + ["ece_null"] + NB_METRICS + ["prevalence"])
      + "\n\n")
    w(f"### T3b. Per-subgroup values, `casemix_location_3` (rule {rule})\n\n")
    w(table_levels(df, "casemix_location_3", rule,
                   CAL_METRICS + NB_METRICS + ["prevalence"]) + "\n\n")

    for test, label in (("wald_maxt", "studentized max-T Wald"),
                        ("permutation", "joint-label permutation"),
                        ("fixed_threshold", "fixed conventional cut")):
        w(f"### T4.{['wald_maxt','permutation','fixed_threshold'].index(test)+1} "
          f"Flag rate of the {label} test, simple (exchangeable) null -- "
          f"this is Type I error in the ordinary sense (rule {rule})\n\n")
        w(table_flagrate_matrix(df, simple, rule,
                                CAL_METRICS + NB_METRICS, test) + "\n\n")

    w(f"### T5. Flag rate under case mix, all three tests (rule {rule})\n\n")
    for test in ("wald_maxt", "permutation", "fixed_threshold"):
        w(f"**{test}**\n\n")
        w(table_flagrate_matrix(df, CASE_MIX_ORDER, rule,
                                CAL_METRICS + NB_METRICS, test) + "\n\n")

    w(f"### T6. Composite null -- subgroups genuinely differ on these metrics, "
      f"so a flag is a TRUE positive (rule {rule})\n\n")
    for test in ("wald_maxt", "permutation"):
        w(f"**{test}**\n\n")
        w(table_flagrate_matrix(df, composite, rule,
                                CAL_METRICS + NB_METRICS, test) + "\n\n")

    w(f"### T7. Rule sensitivity: case-mix flag rates under `ev10` instead of "
      f"`m30`\n\n")
    for test in ("wald_maxt", "permutation"):
        w(f"**{test}**\n\n")
        w(table_flagrate_matrix(df, CASE_MIX_ORDER, "ev10",
                                CAL_METRICS + NB_METRICS, test) + "\n\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
