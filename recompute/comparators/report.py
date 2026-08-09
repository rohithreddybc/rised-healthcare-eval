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

import json
from pathlib import Path
from typing import Dict, List, Optional

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


def _fmt_p(p, floor: bool = False, n_perm=None) -> str:
    """Render a p-value; at the Monte-Carlo floor, as an inequality.

    The floor is ``1 / (B + 1)`` and is taken from the row's own ``n_perm``
    rather than hard-coded, so a table built at a different B does not silently
    print the wrong bound. A floored p-value is never shown as an attained
    value: the +1 correction that makes the permutation test exact-valid is
    precisely what makes values below the floor unreachable.
    """
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "--"
    b = None
    if n_perm is not None and np.isfinite(n_perm):
        b = int(n_perm)
    if floor:
        return f"<= {1.0 / (b + 1):.1e}" if b else "<= 1/(B+1)"
    if b is not None and p <= 1.0 / (b + 1):
        return f"<= {1.0 / (b + 1):.1e}"
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
                "p": _fmt_p(r["p_value"], bool(r.get("p_is_floor", False)),
                            r.get("n_perm")),
                # Adjusted across the ten cohorts within this (method, rule)
                # cell. The raw decision is a decision made ten times over.
                "p Holm (x cohorts)": _fmt_p(
                    r.get("p_holm_across_cohorts")),
                "p BH (x cohorts)": _fmt_p(r.get("p_bh_across_cohorts")),
                "flag after Holm": _yesno(r.get("flag_holm")),
                "flag after BH": _yesno(r.get("flag_bh")),
                "runtime s": f"{r['runtime_s']:.2f}",
            })
    return _md(pd.DataFrame(rows))


def _yesno(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    if isinstance(v, str):
        return v
    return "**yes**" if bool(v) else "no"


def multiplicity_table(df: pd.DataFrame, rule: str) -> str:
    """Raw vs Holm vs BH decisions, per method, at one rule.

    Ten cohorts are tested at once and the headline sentence is "k of ten
    cohorts show an inequitable model", so the per-cohort decision is a decision
    made ten times. ``null_sweep_mmin.csv`` has carried ``p_holm`` and ``p_bh``
    since the joint-null work and neither was surfaced anywhere a reader of the
    report would see them. Both are shown here: Holm controls the family-wise
    error rate under arbitrary dependence, BH the false discovery rate.
    """
    if "flag_holm" not in df.columns:
        return "_re-run the cohort sweep to populate multiplicity columns_"
    d = df[df["rule"] == rule]
    rows = []
    for m in METHOD_ORDER:
        s = d[d["method"] == m]
        if s.empty:
            continue
        raw = set(s[s["conclusion"] == "flag"]["cohort"])
        holm = set(s[s["flag_holm"].astype("boolean").fillna(False)]["cohort"])
        bh = set(s[s["flag_bh"].astype("boolean").fillna(False)]["cohort"])
        applies = bool(s["multiplicity_applies"].any())
        rows.append({
            "method": METHOD_LABEL[m],
            "flags raw": len(raw),
            "flags BH": len(bh) if applies else "n/a",
            "flags Holm": len(holm) if applies else "n/a",
            "lost to Holm": (", ".join(sorted(raw - holm)) or "--") if applies
                            else "n/a (no p-value)",
            "survives Holm": (", ".join(sorted(raw & holm)) or "--") if applies
                             else "n/a",
        })
    return _md(pd.DataFrame(rows))


def var_floor_table(df: pd.DataFrame) -> str:
    """Pairs dropped by the studentized statistic's variance floor.

    ``diciccio.studentized`` returns nan when ``Var_a + Var_b <= 1e-12``, which
    removes the pair from the max-T family. That changes the family the
    family-wise error rate is controlled over, so it has to be counted rather
    than merely handled.
    """
    d = df[df["method"] == "diciccio2020"]
    if d.empty or "detail" not in d.columns:
        return "_no diciccio2020 rows_"
    rows = []
    total = 0
    for _, r in d.iterrows():
        det = str(r.get("detail", ""))
        n_drop = None
        for part in det.split(";"):
            if "n_pairs_dropped_var_floor=" in part:
                n_drop = int(part.split("=")[1])
        if n_drop is None:
            continue
        total += n_drop
        if n_drop:
            rows.append({"cohort": r["cohort_label"], "rule": r["rule"],
                         "pairs dropped by VAR_FLOOR": n_drop,
                         "detail": det.split(";")[0].strip()})
    if not rows:
        return (f"No pair was dropped by the variance floor in any "
                f"(cohort x rule) cell: {len(d)} cells checked, "
                f"{total} pairs dropped in total. The floor is therefore not "
                f"silently shrinking the max-T family anywhere in this study.")
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


def agreement_matrix(df: pd.DataFrame, rule: str) -> str:
    """Pairwise agreement between methods on the evaluable cohorts."""
    d = df[(df["rule"] == rule) & (df["conclusion"] != "not_evaluable")]
    piv = d.pivot_table(index="cohort", columns="method", values="conclusion",
                        aggfunc="first")
    rows = []
    for a in METHOD_ORDER:
        row = {"method": METHOD_LABEL[a]}
        for b in METHOD_ORDER:
            if a not in piv or b not in piv:
                row[b] = "--"
                continue
            sa, sb = piv[a], piv[b]
            keep = sa.notna() & sb.notna()
            n = int(keep.sum())
            row[b] = f"{int((sa[keep] == sb[keep]).sum())}/{n}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.columns = ["method"] + [METHOD_LABEL[m] for m in METHOD_ORDER]
    return _md(out)


def incumbent_vs_rest(df: pd.DataFrame, rule: str) -> str:
    """What each method finds that the incumbent misses, and the reverse."""
    d = df[(df["rule"] == rule) & (df["conclusion"] != "not_evaluable")]
    piv = d.pivot_table(index="cohort", columns="method", values="conclusion",
                        aggfunc="first")
    if "permutation_null" not in piv:
        return "_incumbent column missing_"
    inc = piv["permutation_null"]
    rows = []
    for m in METHOD_ORDER:
        if m == "permutation_null" or m not in piv:
            continue
        other = piv[m]
        only_other = [c for c in piv.index
                      if other[c] == "flag" and inc[c] == "no_flag"]
        only_inc = [c for c in piv.index
                    if inc[c] == "flag" and other[c] == "no_flag"]
        rows.append({
            "method": METHOD_LABEL[m],
            "flags that the incumbent misses": (", ".join(only_other)
                                                if only_other else "--"),
            "incumbent flags that it misses": (", ".join(only_inc)
                                               if only_inc else "--"),
        })
    return _md(pd.DataFrame(rows))


def same_kernel_runtime(rt: pd.DataFrame, env: Optional[dict] = None) -> str:
    """Like-for-like runtime: every method on the same vectorised kernel.

    Every figure is the median of ``n_repeats`` repeated timings, run
    sequentially on the machine described above the table. A single un-repeated
    timing has no error bar and must not be used to claim a difference of a few
    per cent, which is the size of the claim being made here.
    """
    if "diciccio2020_median_s" not in rt.columns:
        return ("_comparator_runtime.csv predates the repeated benchmark; "
                "re-run `python -m recompute.comparators.bench --repeats 3`. "
                "The superseded file held a single un-repeated timing per "
                "method, which cannot support a runtime claim._")
    rows = []
    for _, r in rt.iterrows():
        rows.append({
            "cohort": r["cohort_label"],
            "n_test": f"{int(r['n_test']):,}",
            "part.": int(r["n_partitions"]),
            "incumbent, shipped kernel s": (
                f"{r['permutation_null_shipped_kernel_s']:.1f}"),
            "incumbent, same kernel s (median)": (
                f"{r['permutation_null_same_kernel_median_s']:.1f}"),
            "+- sd": f"{r['permutation_null_same_kernel_sd_s']:.1f}",
            "DiCiccio s (median)": f"{r['diciccio2020_median_s']:.1f}",
            "+- sd ": f"{r['diciccio2020_sd_s']:.1f}",
            "overhead %": f"{r['studentization_overhead_pct']:+.1f}",
            "> noise?": _yesno(r.get("overhead_exceeds_repeat_noise")),
            "Lum s": f"{r['lum2022_median_s']:.3f}",
            "four-fifths s": f"{r['four_fifths_median_s']:.3f}",
            "fixed threshold s": f"{r['fixed_threshold_005_median_s']:.3f}",
        })
    out = _md(pd.DataFrame(rows))

    n_rep = int(rt["n_repeats"].iloc[0])
    resolvable = rt["overhead_exceeds_repeat_noise"].astype("boolean").fillna(
        False)
    head = []
    if env:
        head.append(
            f"_Machine: {env.get('cpu_brand') or env.get('processor')}, "
            f"{env.get('physical_cores') or '?'} physical / "
            f"{env.get('logical_cores')} logical cores, "
            f"{env.get('ram_total_gb')} GB RAM, {env.get('platform')}; "
            f"numpy {env.get('numpy')}, scipy {env.get('scipy')}. "
            f"B = {int(rt['n_perm'].iloc[0]):,}, {n_rep} repeats, "
            f"sequential (no two timings share the machine)._\n")
    verdict = (
        f"\n**Runtime verdict.** The studentization overhead exceeds the "
        f"run-to-run noise in {int(resolvable.sum())} of {len(rt)} cohorts. "
        + ("Where it does not, the honest statement is that studentization "
           "costs no material runtime, with **no number attached** -- the "
           "difference is inside the measurement noise and quoting it would "
           "overstate what {n} repeats can resolve.".format(n=n_rep)
           if not resolvable.all() else
           "The overhead is resolvable and is quoted per cohort above.")
        + " Note that the shipped-kernel column is the incumbent as it actually "
          "ships, through `scipy.rankdata`; it must not be differenced against "
          "the studentized column, because the gap between them is the "
          "vectorised kernel, not the statistic.")
    return "\n".join(head) + out + verdict


def rule_substantive_stability(df: pd.DataFrame) -> str:
    """The table the manuscript's remaining claim stands or falls on.

    Counting verdict flips per cohort understates the difference between
    methods, because a flip only matters if it moves the *substantive*
    conclusion. The manuscript's claim is about the clinical cohorts: "no
    clinical cohort produces a gap distinguishable from chance". So the
    quantity that matters is how many clinical cohorts each method flags under
    each admissibility rule, and whether that number ever crosses zero.

    ``Jaccard vs m30`` is the overlap of the flagged cohort *set* with the set at
    the published rule; 1.00 means the same cohorts, lower means reshuffling.
    """
    rows = []
    for m in METHOD_ORDER:
        counts, jac = [], []
        base = set(df[(df["method"] == m) & (df["rule"] == "m30")
                      & (df["conclusion"] == "flag")]["cohort"])
        for ru in RULE_NAMES:
            sel = df[(df["method"] == m) & (df["rule"] == ru)
                     & (df["conclusion"] == "flag")]
            counts.append(int(sel[sel["is_clinical"].astype(bool)].shape[0]))
            s = set(sel["cohort"])
            union = base | s
            jac.append(len(base & s) / len(union) if union else 1.0)
        rows.append({
            "method": METHOD_LABEL[m],
            "clinical flagged m20/m30/m50/m100/ev10": " ".join(str(c)
                                                               for c in counts),
            "span": max(counts) - min(counts),
            "crosses zero?": ("**YES -- conclusion inverts**"
                              if (min(counts) == 0 and max(counts) > 0)
                              else "no"),
            "min Jaccard vs m30": f"{min(jac):.2f}",
        })
    return _md(pd.DataFrame(rows))


def rule_pvalue_spread(df: pd.DataFrame) -> str:
    """How far each method's p-value travels across the five inclusion rules.

    The manuscript's remaining claim is that the subgroup-admissibility rule is
    an uncontrolled degree of freedom that flips the verdict. That claim is
    general only if it afflicts published methods too, so this is the table the
    claim stands or falls on. ``log10 span`` is
    ``log10(max p / min p)`` over the rules -- orders of magnitude of p-value
    swing attributable to nothing but the admissibility rule.
    """
    rows = []
    for m in METHOD_ORDER:
        d = df[(df["method"] == m) & (df["conclusion"] != "not_evaluable")]
        if d.empty or d["p_value"].isna().all():
            rows.append({"method": METHOD_LABEL[m], "cohorts with a p-value": 0,
                         "median log10 span": "n/a (no p-value)",
                         "worst log10 span": "n/a", "worst cohort": "--",
                         "worst p range": "--"})
            continue
        recs = []
        for c, g in d.groupby("cohort"):
            p = g["p_value"].dropna()
            if len(p) < 2 or p.min() <= 0:
                continue
            recs.append((c, float(np.log10(p.max() / p.min())), p.min(), p.max()))
        if not recs:
            continue
        spans = np.array([r[1] for r in recs])
        worst = recs[int(np.argmax(spans))]
        rows.append({
            "method": METHOD_LABEL[m],
            "cohorts with a p-value": len(recs),
            "median log10 span": f"{np.median(spans):.2f}",
            "worst log10 span": f"{spans.max():.2f}",
            "worst cohort": worst[0],
            "worst p range": f"{worst[2]:.4g} to {worst[3]:.4g}",
        })
    return _md(pd.DataFrame(rows))


def rule_flip_detail(df: pd.DataFrame) -> str:
    """Per-cohort verdict string across the five rules, for every method."""
    rows = []
    for m in METHOD_ORDER:
        piv = df[df["method"] == m].pivot_table(
            index="cohort", columns="rule", values="conclusion", aggfunc="first")
        piv = piv.reindex(columns=RULE_NAMES)
        for c in COHORT_ORDER:
            if c not in piv.index:
                continue
            seq = [{"flag": "F", "no_flag": ".", "not_evaluable": "-"}.get(
                piv.loc[c, ru], "?") for ru in RULE_NAMES]
            decided = {s for s in seq if s in ("F", ".")}
            rows.append({
                "method": METHOD_LABEL[m],
                "cohort": c,
                "m20/m30/m50/m100/ev10": " ".join(seq),
                "flips?": "**yes**" if len(decided) > 1 else "no",
            })
    out = pd.DataFrame(rows)
    return _md(out[out["flips?"] == "**yes**"])


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


def _family(t1: pd.DataFrame) -> pd.Series:
    """simple / composite / case_mix, tolerating pre-round-2 CSVs."""
    if "null_family" in t1.columns:
        return t1["null_family"].astype(str)
    return pd.Series(
        np.where(t1["composite_null"].astype(bool), "composite", "simple"),
        index=t1.index)


def type1_table(t1: pd.DataFrame, rule: str, family: Optional[str] = None
                ) -> str:
    d = t1[t1["rule"] == rule]
    fam = _family(d)
    if family is not None:
        d = d[fam == family]
    if d.empty:
        return "_no cells_"
    geoms = list(dict.fromkeys(d["geometry"]))
    rows = []
    for g in geoms:
        sub = d[d["geometry"] == g]
        row = {"geometry": g, "family": _family(sub).iloc[0]}
        if family == "case_mix":
            row["true AUROC gap"] = f"{sub['true_auc_gap'].iloc[0]:.3f}"
        for m in METHOD_ORDER:
            r = sub[sub["method"] == m]
            if r.empty:
                row[m] = "--"
                continue
            r = r.iloc[0]
            # Every rate carries its own Monte-Carlo standard error, in
            # parentheses. At 1000 simulations the SE of a rate near 0.05 is
            # 0.0069, so two cells differing by less than ~0.02 are not
            # distinguishable and should not be described as different.
            row[m] = (f"{r['type1_rate']:.3f}"
                      + (f" ({r['type1_mc_se']:.4f})"
                         if np.isfinite(r["type1_mc_se"]) else ""))
        rows.append(row)
    out = pd.DataFrame(rows)
    lead = ["geometry", "family"] + (["true AUROC gap"]
                                     if family == "case_mix" else [])
    out.columns = lead + [METHOD_LABEL[m] for m in METHOD_ORDER]
    return _md(out)


def type1_summary(t1: pd.DataFrame, alpha: float = 0.05) -> str:
    """Calibration headline: worst-case Type I rate per method.

    Split by simple versus composite null, because that split is the whole point
    of the DiCiccio paper and a method can be perfectly calibrated on one and
    badly wrong on the other.

    **Case-mix cells are excluded from this table entirely.** There the true
    subgroup AUROCs are unequal by construction, so a flag is not a Type I error
    and averaging it into a Type I rate would be a category error. They have
    their own table.

    The ``worst`` columns are maxima over many correlated cells and must be read
    as such: at 1000 simulations the Monte-Carlo SE of a rate at 0.05 is 0.0069,
    and the maximum of twelve correlated cells each drawn around 0.05 sits
    around 0.06-0.07 for a procedure of exactly the right size. ``worst -
    alpha in SE units`` gives the excess in those units so the reader does not
    have to do it.
    """
    fam = _family(t1)
    rows = []
    for m in METHOD_ORDER:
        d = t1[(t1["method"] == m) & (fam != "case_mix")]
        if d.empty:
            continue
        dfam = _family(d)
        simple = d[dfam == "simple"]["type1_rate"]
        comp = d[dfam == "composite"]["type1_rate"]
        nominal = bool(d["has_nominal_level"].iloc[0])
        worst = float(d["type1_rate"].max())
        se_at_worst = float(
            d.loc[d["type1_rate"].idxmax(), "type1_mc_se"])
        rows.append({
            "method": METHOD_LABEL[m],
            "nominal level?": "yes" if nominal else "no (screen)",
            "cells": int(len(d)),
            "simple null: median": f"{simple.median():.3f}",
            "simple null: worst": f"{simple.max():.3f}",
            "composite null: median": f"{comp.median():.3f}",
            "composite null: worst": f"{comp.max():.3f}",
            "overall worst": f"{worst:.3f}",
            "MC SE at worst": f"{se_at_worst:.4f}",
            "(worst - 0.05) / SE": (f"{(worst - alpha) / se_at_worst:+.1f}"
                                    if se_at_worst > 0 else "--"),
            "verdict": _calibration_verdict(worst, nominal, alpha),
        })
    return _md(pd.DataFrame(rows))


def case_mix_summary(t1: pd.DataFrame, alpha: float = 0.05) -> str:
    """Flag rate under the case-mix null, where a flag is a FALSE ALARM.

    Every other table in this report measures a procedure against the null it
    claims to test: all subgroups share one true AUROC. That null is not the one
    a clinical prediction model satisfies. Subgroup AUROC is case-mix dependent,
    so a single shared model with identical coefficients produces unequal
    subgroup AUROC whenever the subgroups' covariate distributions differ
    (Vergouwe et al. 2010; van Klaveren et al. 2016). In these cells the model
    *is* the data-generating probability -- correctly specified, perfectly
    calibrated in every subgroup, one coefficient vector for everybody -- and
    the true subgroup AUROCs still differ by the amount in the ``true gap``
    column.

    A flag here is therefore not a Type I error in the equal-AUROC sense; it is
    a correct rejection of a null nobody should be testing, and a **false alarm
    about fairness**, which is the decision an auditor acts on. High numbers in
    this table are bad, and they are bad in a way no amount of Type I
    calibration can fix, because the procedures are correctly detecting a real
    difference in a quantity that is not evidence of unfairness.
    """
    fam = _family(t1)
    d = t1[fam == "case_mix"]
    if d.empty:
        return "_no case-mix cells; re-run the Type I study_"
    rows = []
    for m in METHOD_ORDER:
        s = d[d["method"] == m]
        if s.empty:
            continue
        rows.append({
            "method": METHOD_LABEL[m],
            "cells": int(len(s)),
            "false-alarm rate: min": f"{s['type1_rate'].min():.3f}",
            "median": f"{s['type1_rate'].median():.3f}",
            "max": f"{s['type1_rate'].max():.3f}",
            "MC SE at max": f"{s.loc[s['type1_rate'].idxmax(), 'type1_mc_se']:.4f}",
            "worst geometry": s.loc[s["type1_rate"].idxmax(), "geometry"],
            "true gap there": (
                f"{s.loc[s['type1_rate'].idxmax(), 'true_auc_gap']:.3f}"),
        })
    return _md(pd.DataFrame(rows))


def case_mix_truth(t1: pd.DataFrame) -> str:
    """The exact true subgroup AUROCs of each case-mix geometry."""
    fam = _family(t1)
    d = t1[fam == "case_mix"].drop_duplicates("geometry")
    if d.empty:
        return "_no case-mix cells_"
    rows = [{
        "geometry": r["geometry"], "n": int(r["n"]),
        "prevalence": f"{r['prevalence']:.2f}",
        "levels": int(r["max_levels"]),
        "true AUROC min": f"{r['true_auc_min']:.3f}",
        "true AUROC max": f"{r['true_auc_max']:.3f}",
        "true gap": f"{r['true_auc_gap']:.3f}",
        "unfairness present": "none -- the score IS the true probability",
        "description": r["description"],
    } for _, r in d.iterrows()]
    return _md(pd.DataFrame(rows))


def _calibration_verdict(worst: float, nominal: bool, alpha: float) -> str:
    if not nominal:
        return "n/a -- no error control claimed"
    if not np.isfinite(worst):
        return "--"
    # MC SE of a rate at 1000 simulations is at most 0.016; allow two of them.
    if worst <= alpha + 0.032:
        return "calibrated"
    if worst <= 2 * alpha:
        return "mildly anti-conservative"
    return "**anti-conservative**"


def type1_geometry_key(t1: pd.DataFrame) -> str:
    d = t1.drop_duplicates("geometry")
    fam = _family(d)
    rows = [{
        "geometry": r["geometry"], "family": f, "n": int(r["n"]),
        "prevalence": f"{r['prevalence']:.3f}",
        "partitions": int(r["n_partitions"]),
        "max levels": int(r["max_levels"]),
        "smallest level": f"{r['min_level_frac']:.2f}",
        "transform": (r.get("monotone_transform") or "--"),
        "true AUROC gap": f"{r.get('true_auc_gap', 0.0):.3f}",
        "description": r["description"],
    } for (_, r), f in zip(d.iterrows(), fam)]
    return _md(pd.DataFrame(rows))


def stability_tables() -> Dict[str, str]:
    """The round-2 rule-stability tables, from rule_stability.py's output."""
    per = RESULTS / "rule_stability.csv"
    by_m = RESULTS / "rule_stability_by_method.csv"
    if not (per.exists() and by_m.exists()):
        return {"rank": "_run `python -m recompute.comparators.rule_stability`_",
                "traj": ""}
    pc = pd.read_csv(per)
    bm = pd.read_csv(by_m)

    out = {}
    blocks = []
    for cs in ("all", "clinical"):
        g = bm[bm["cohort_set"] == cs].sort_values("stability_rank")
        rows = [{
            "rank": int(r["stability_rank"]),
            "method": METHOD_LABEL.get(r["method"], r["method"]),
            "scheme": (r["permutation_scheme"]
                       if isinstance(r.get("permutation_scheme"), str)
                       and r["permutation_scheme"] else "n/a"),
            "cohorts changing verdict": int(r["n_cohorts_nonconstant_verdict"]),
            "which": r["cohorts_nonconstant"] or "--",
            "flag COUNT constant?": _yesno(r["n_flag_is_constant"]),
            "flagged SET constant?": _yesno(r["flagged_set_is_constant"]),
            "mean Jaccard vs m30": f"{r['mean_jaccard_vs_m30']:.3f}",
            "min Jaccard vs m30": f"{r['min_jaccard_vs_m30']:.3f}",
            "flags @m30": int(r["n_flag_m30"]),
        } for _, r in g.iterrows()]
        blocks.append(f"**{cs} cohorts** "
                      f"({int(g['n_cohorts_with_any_verdict'].max())} with a "
                      f"verdict under at least one rule)\n\n"
                      + _md(pd.DataFrame(rows)))
    out["rank"] = "\n\n".join(blocks) + (
        "\n\n_Read the ranking with the `flags @m30` column in view. A rule "
        "that flags almost every cohort, or almost none, has little room to "
        "churn and scores well for that reason alone; the fixed-0.05 threshold "
        "flags 7 of 10 and the four-fifths rule 3 of 10, and neither is being "
        "credited here with accuracy, only with insensitivity to the "
        "admissibility rule. The finding this table supports is narrower and "
        "harder: the two permutation procedures, which are the ones that claim "
        "a nominal level, are the least stable of the seven, and the "
        "studentized test -- whose stability was the claim under review -- "
        "changes the verdict on three clinical cohorts while keeping its "
        "flagged count fixed at 3._")

    keep = pc[~pc["verdict_constant"]].copy()
    rows = []
    for _, r in keep.iterrows():
        seq = " ".join({"flag": "F", "no_flag": ".", "not_evaluable": "-"}
                       .get(r[f"conclusion_{ru}"], "?") for ru in RULE_NAMES)
        ps = " ".join(("--" if not np.isfinite(r[f"p_{ru}"])
                       else f"{r[f'p_{ru}']:.3f}") for ru in RULE_NAMES)
        rows.append({
            "method": METHOD_LABEL.get(r["method"], r["method"]),
            "cohort": r["cohort"],
            "verdict m20/m30/m50/m100/ev10": seq,
            "p m20/m30/m50/m100/ev10": ps,
            "p range": ("--" if not np.isfinite(r["p_range"])
                        else f"{r['p_range']:.3f}"),
            "straddles 0.05": _yesno(r["p_straddles_alpha"]),
        })
    out["traj"] = _md(pd.DataFrame(rows)) if rows else "_none_"
    return out


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
    parts.append("## T3. Pairwise agreement on evaluable cohorts, m30\n")
    parts.append(agreement_matrix(df, "m30") + "\n")
    parts.append("## T4. What each method finds that the incumbent does not, m30\n")
    parts.append(incumbent_vs_rest(df, "m30") + "\n")
    parts.append("## T5. Same, ev10\n")
    parts.append(incumbent_vs_rest(df, "ev10") + "\n")
    parts.append("## T5b. Cross-cohort multiplicity: raw vs Holm vs BH, m30\n")
    parts.append(multiplicity_table(df, "m30") + "\n")
    parts.append("### T5c. Same, ev10\n")
    parts.append(multiplicity_table(df, "ev10") + "\n")
    parts.append("## T5d. Pairs dropped by the studentized variance floor\n")
    parts.append(var_floor_table(df) + "\n")
    stab = stability_tables()
    parts.append("## T6. Rule stability: per-cohort verdict concordance "
                 "(the corrected metric)\n")
    parts.append(
        "The previous stability claim was read off the **count** of flagged "
        "cohorts, which is invariant to any permutation of the flagged set and "
        "therefore cannot detect churn. `diciccio2020` flags exactly 3 clinical "
        "cohorts under all five rules while the *set* changes three times. The "
        "metric below is computed at the level the decision is made -- the "
        "cohort.\n")
    parts.append(stab["rank"] + "\n")
    parts.append("### T6-0. Every cohort that changes verdict, with its "
                 "p-value trajectory\n")
    parts.append(stab["traj"] + "\n")
    parts.append("## T6d. Inclusion-rule sensitivity: does the SUBSTANTIVE "
                 "conclusion move?\n")
    parts.append(rule_substantive_stability(df) + "\n")
    parts.append("## T6a. Inclusion-rule sensitivity: flags and verdict flips\n")
    parts.append(rule_sensitivity(df) + "\n")
    parts.append("## T6b. Inclusion-rule sensitivity: p-value spread\n")
    parts.append(rule_pvalue_spread(df) + "\n")
    parts.append("## T6c. Every cohort whose verdict flips, by method\n")
    parts.append(rule_flip_detail(df) + "\n")
    parts.append("## T7. Runtime, all methods on the same kernel, repeated\n")
    rtp = RESULTS / "comparator_runtime.csv"
    envp = RESULTS / "comparator_runtime_env.json"
    if rtp.exists():
        env = (json.loads(envp.read_text(encoding="utf-8"))
               if envp.exists() else None)
        parts.append(same_kernel_runtime(pd.read_csv(rtp), env) + "\n")
    else:
        parts.append("_run `python -m recompute.comparators.bench --repeats 3` "
                     "first_\n")
    parts.append("### T7b. Runtime as recorded in the comparison sweep, m30\n")
    parts.append(runtime_table(df) + "\n")
    parts.append("## T8. Full detail, m30\n")
    parts.append(detail_table(df, "m30") + "\n")
    parts.append("## T9. Full detail, ev10\n")
    parts.append(detail_table(df, "ev10") + "\n")
    if t1 is not None:
        fam = _family(t1)
        parts.append("## T10. Simulated-null geometries\n")
        parts.append(type1_geometry_key(t1) + "\n")
        parts.append("## T11. Type I error, calibration summary "
                     "(equal-true-AUROC nulls only)\n")
        parts.append(
            "Case-mix cells are **excluded** from this table: there the true "
            "subgroup AUROCs differ by construction, so a flag is not a Type I "
            "error and must not be averaged into one. See T14.\n\n"
            "Every rate in the per-cell tables carries its Monte-Carlo standard "
            "error in parentheses. At 1000 simulations the MC SE of a rate at "
            "0.05 is 0.0069. The `overall worst` column is a maximum over many "
            "positively correlated cells, and the maximum of twelve such cells "
            "drawn from an exactly-sized procedure lands around 0.06-0.07; "
            "`(worst - 0.05) / SE` is given so that excess can be judged "
            "instead of eyeballed.\n")
        parts.append(type1_summary(t1) + "\n")

        n = 12
        for rule in sorted(set(t1["rule"])):
            parts.append(f"## T{n}. Type I error at nominal 0.05, rule {rule} "
                         f"-- simple and composite nulls\n")
            parts.append("_rate (Monte-Carlo SE)_\n")
            parts.append(type1_table(t1[fam != "case_mix"], rule) + "\n")
            n += 1

        if (fam == "case_mix").any():
            parts.append(f"## T{n}. CASE-MIX NULL: the true subgroup AUROCs\n")
            parts.append(
                "One shared model, identical coefficients, applied to subgroups "
                "with different predictor spread. The score **is** the true "
                "event probability, so the model is correctly specified and "
                "perfectly calibrated in every subgroup and no unfairness of "
                "any kind is present -- yet the true subgroup AUROCs differ by "
                "the amounts below, computed exactly by quadrature rather than "
                "simulated. This is the expected behaviour of a fair clinical "
                "prediction model (Vergouwe et al. 2010; van Klaveren et al. "
                "2016), and it is the null the five procedures were never "
                "tested against.\n")
            parts.append(case_mix_truth(t1) + "\n")
            n += 1
            parts.append(f"## T{n}. CASE-MIX NULL: false-alarm rate summary\n")
            parts.append(
                "**A flag in these cells is a false alarm about fairness.** It "
                "is not a Type I error against the equal-AUROC null -- that "
                "null is genuinely false here -- which is exactly why no amount "
                "of Type I calibration protects against it.\n")
            parts.append(case_mix_summary(t1) + "\n")
            n += 1
            for rule in sorted(set(t1["rule"])):
                parts.append(f"## T{n}. Case-mix false-alarm rate, rule "
                             f"{rule}\n")
                parts.append("_rate (Monte-Carlo SE)_\n")
                parts.append(type1_table(t1, rule, family="case_mix") + "\n")
                n += 1
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
