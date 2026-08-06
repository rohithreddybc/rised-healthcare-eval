"""
Render ``COMPARATOR_EVALUATION.md`` from the two result CSVs.

    python -m recompute.comparators.report

Reads ``recompute/results/comparator_comparison.csv`` and
``recompute/results/comparator_type1.csv`` and writes the tables of the report to
``COMPARATOR_EVALUATION.tables.md`` at the repo root. The narrative sections of
the report are written by hand around these tables; keeping the tables generated
means no number in the document is transcribed by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from recompute.comparators.core import COHORT_ORDER, RULE_NAMES, REPO

RESULTS = REPO / "recompute" / "results"
OUT = REPO / "COMPARATOR_EVALUATION.tables.md"

METHOD_ORDER = [
    "permutation_null", "diciccio2020", "lum2022", "lum2022_cochranQ",
    "lum2022_bootstrapCI", "four_fifths", "fixed_threshold_005",
]
METHOD_LABEL = {
    "permutation_null": "Permutation null (incumbent)",
    "diciccio2020": "DiCiccio 2020 (studentized max-T)",
    "lum2022": "Lum 2022 (V_dc, closed-form z)",
    "lum2022_cochranQ": "Lum 2022 inputs, Cochran Q (our addition)",
    "lum2022_bootstrapCI": "Lum 2022 (V_dc, bootstrap CI)",
    "four_fifths": "Four-fifths (0.80 ratio)",
    "fixed_threshold_005": "Fixed threshold (Delta >= 0.05)",
}
MARK = {"flag": "**F**", "no_flag": ".", "not_evaluable": "n/e"}


def _fmt_p(p, floor: bool = False) -> str:
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "--"
    if floor:
        return "<= 1.0e-04"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def verdict_grid(df: pd.DataFrame, rule: str) -> str:
    d = df[df["rule"] == rule]
    piv = d.pivot_table(index="cohort", columns="method", values="conclusion",
                        aggfunc="first")
    rows = []
    for c in COHORT_ORDER:
        if c not in piv.index:
            continue
        lab = d[d["cohort"] == c]["cohort_label"].iloc[0]
        clin = "clin" if bool(d[d["cohort"] == c]["is_clinical"].iloc[0]) else "cross"
        row = {"cohort": lab, "type": clin}
        for m in METHOD_ORDER:
            row[m] = MARK.get(piv.loc[c].get(m), "?")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.columns = ["cohort", "type"] + [METHOD_LABEL[m] for m in METHOD_ORDER]
    return _md(out)


def detail_table(df: pd.DataFrame, rule: str) -> str:
    d = df[(df["rule"] == rule)]
    rows = []
    for c in COHORT_ORDER:
        sub = d[d["cohort"] == c]
        if sub.empty:
            continue
        for m in METHOD_ORDER:
            r = sub[sub["method"] == m]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({
                "cohort": r["cohort_label"],
                "method": METHOD_LABEL[m],
                "conclusion": r["conclusion"],
                "statistic": ("--" if pd.isna(r["statistic"])
                              else f"{r['statistic']:.4g}"),
                "stat name": r["statistic_name"] or "--",
                "p": _fmt_p(r["p_value"], bool(r.get("p_is_floor", False))),
                "runtime s": f"{r['runtime_s']:.2f}",
            })
    return _md(pd.DataFrame(rows))


def rule_sensitivity(df: pd.DataFrame) -> str:
    rows = []
    for m in METHOD_ORDER:
        piv = df[df["method"] == m].pivot_table(
            index="cohort", columns="rule", values="conclusion", aggfunc="first")
        piv = piv.reindex(columns=RULE_NAMES)
        n_flip, who = 0, []
        for c, r in piv.iterrows():
            seen = set(r[r.isin(["flag", "no_flag"])])
            if len(seen) > 1:
                n_flip += 1
                who.append(c)
        counts = {ru: int((df[(df["method"] == m) & (df["rule"] == ru)]
                           ["conclusion"] == "flag").sum())
                  for ru in RULE_NAMES}
        rows.append({
            "method": METHOD_LABEL[m],
            **{f"flags @{ru}": counts[ru] for ru in RULE_NAMES},
            "cohorts whose verdict flips": f"{n_flip}/10",
            "which": ", ".join(who) if who else "--",
        })
    return _md(pd.DataFrame(rows))


def runtime_table(df: pd.DataFrame) -> str:
    rows = []
    for m in METHOD_ORDER:
        d = df[(df["method"] == m) & (df["rule"] == "m30")]
        rows.append({
            "method": METHOD_LABEL[m],
            "min s": f"{d['runtime_s'].min():.3f}",
            "median s": f"{d['runtime_s'].median():.3f}",
            "max s": f"{d['runtime_s'].max():.3f}",
            "total over 10 cohorts s": f"{d['runtime_s'].sum():.1f}",
        })
    return _md(pd.DataFrame(rows))


def type1_table(t1: pd.DataFrame, rule: str) -> str:
    d = t1[t1["rule"] == rule]
    geoms = list(dict.fromkeys(d["geometry"]))
    rows = []
    for g in geoms:
        sub = d[d["geometry"] == g]
        row = {"geometry": g,
               "composite": "yes" if bool(sub["composite_null"].iloc[0]) else "no"}
        for m in METHOD_ORDER:
            r = sub[sub["method"] == m]
            if r.empty:
                row[m] = "--"
                continue
            r = r.iloc[0]
            row[m] = (f"{r['type1_rate']:.3f}"
                      + (f" ({r['type1_mc_se']:.3f})"
                         if np.isfinite(r["type1_mc_se"]) else ""))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.columns = ["geometry", "composite"] + [METHOD_LABEL[m]
                                               for m in METHOD_ORDER]
    return _md(out)


def type1_geometry_key(t1: pd.DataFrame) -> str:
    d = t1.drop_duplicates("geometry")
    rows = [{
        "geometry": r["geometry"], "n": int(r["n"]),
        "prevalence": f"{r['prevalence']:.3f}",
        "partitions": int(r["n_partitions"]),
        "max levels": int(r["max_levels"]),
        "smallest level": f"{r['min_level_frac']:.2f}",
        "composite null": "yes" if bool(r["composite_null"]) else "no",
        "description": r["description"],
    } for _, r in d.iterrows()]
    return _md(pd.DataFrame(rows))


def main() -> int:
    df = pd.read_csv(RESULTS / "comparator_comparison.csv")
    t1p = RESULTS / "comparator_type1.csv"
    t1 = pd.read_csv(t1p) if t1p.exists() else None

    parts: List[str] = ["<!-- generated by `python -m recompute.comparators.report`"
                        " -- do not edit by hand -->\n"]
    parts.append("## T1. Verdict grid, published rule m30\n")
    parts.append(verdict_grid(df, "m30") + "\n")
    parts.append("## T2. Verdict grid, events rule ev10\n")
    parts.append(verdict_grid(df, "ev10") + "\n")
    parts.append("## T3. Inclusion-rule sensitivity\n")
    parts.append(rule_sensitivity(df) + "\n")
    parts.append("## T4. Runtime at m30\n")
    parts.append(runtime_table(df) + "\n")
    parts.append("## T5. Full detail, m30\n")
    parts.append(detail_table(df, "m30") + "\n")
    parts.append("## T6. Full detail, ev10\n")
    parts.append(detail_table(df, "ev10") + "\n")
    if t1 is not None:
        parts.append("## T7. Simulated-null geometries\n")
        parts.append(type1_geometry_key(t1) + "\n")
        for rule in sorted(set(t1["rule"])):
            parts.append(f"## T8. Type I error at nominal 0.05, rule {rule}\n")
            parts.append(type1_table(t1, rule) + "\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
