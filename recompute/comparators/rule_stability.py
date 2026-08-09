"""
Per-cohort verdict concordance across the subgroup-inclusion rules.

    python -m recompute.comparators.rule_stability

Writes ``recompute/results/rule_stability.csv`` (one row per method x cohort,
with the full verdict and p-value trajectory across the five rules) and
``recompute/results/rule_stability_by_method.csv`` (the per-method summary,
including the flagged sets and their Jaccard overlap with the m30 baseline).

Why the previous stability claim was wrong
------------------------------------------
The claim was that "the studentized test returns the same verdict under every
admissibility rule". It was read off the **count** of flagged cohorts, which for
``diciccio2020`` on the six clinical cohorts is 3 under every rule. But the count
is not the verdict. The flagged *set* is:

    m20 / m30 / m50   {brfss2024, diabetes130, nhis2024}
    m100              {diabetes130, nhis2023, nhis2024}
    ev10              {brfss2024, diabetes130, nhis2023}

Three different sets with the same cardinality. Three of the six clinical
cohorts -- brfss2024, nhis2023, nhis2024 -- receive a different verdict depending
on which inclusion rule the analyst picked, and only diabetes130 is flagged under
all five. A constant count concealed complete churn in *which* hospital would be
told its model is inequitable, which is the only thing a reader of the audit
cares about.

A count is invariant to any permutation of the flagged set, so it cannot detect
this and should never have been used as a stability measure. The metrics below
are computed at the level the decision is actually made: the cohort.

  ``n_cohorts_nonconstant``   how many cohorts change verdict across the rules.
                              This is the headline. Zero means genuinely stable.
  ``jaccard_vs_m30``          |A n B| / |A u B| between each rule's flagged set
                              and the published m30 baseline's. 1.0 means the
                              same cohorts, not merely the same number of them.
  p-value trajectory          ``p_m20 ... p_ev10`` per cohort, so a verdict that
                              flips because a p-value sits at 0.049 vs 0.051 is
                              distinguishable from one that moves across the
                              whole unit interval.

``not_evaluable`` is handled two ways and both are reported, because they answer
different questions. ``verdict_constant`` ignores non-evaluable rules and asks
whether the *decisions actually issued* agree; ``verdict_constant_strict`` treats
non-evaluable as its own state, so a cohort that is evaluable under some rules
and not others counts as unstable. UCI Heart is non-evaluable everywhere for the
permutation methods and is constant under both readings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from recompute.comparators.core import REPO

RESULTS = REPO / "recompute" / "results"
IN_CSV = RESULTS / "comparator_comparison.csv"
OUT_CSV = RESULTS / "rule_stability.csv"
OUT_METHOD_CSV = RESULTS / "rule_stability_by_method.csv"

#: Rule order used everywhere, with the published default in the middle.
RULES: Sequence[str] = ("m20", "m30", "m50", "m100", "ev10")
BASELINE = "m30"


def jaccard(a: Set[str], b: Set[str]) -> float:
    """|A n B| / |A u B|, with the empty-empty case defined as 1.0.

    Two rules that both flag nothing agree perfectly about every cohort, so 1.0
    is the right value; ``jaccard_vacuous`` records that it was reached that way
    and carries no information about which cohorts were selected.
    """
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── per method x cohort ──────────────────────────────────────────────────────
def cohort_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (method, scheme, cohort), g in df.groupby(
            ["method", "permutation_scheme", "cohort"], sort=False,
            dropna=False):
        g = g.set_index("rule")
        meta = g.iloc[0]
        verdicts = {r: (g.loc[r, "conclusion"] if r in g.index else None)
                    for r in RULES}
        pvals = {r: (float(g.loc[r, "p_value"])
                     if r in g.index and pd.notna(g.loc[r, "p_value"])
                     else np.nan) for r in RULES}
        stats = {r: (float(g.loc[r, "statistic"])
                     if r in g.index and pd.notna(g.loc[r, "statistic"])
                     else np.nan) for r in RULES}

        issued = [v for v in verdicts.values() if v in ("flag", "no_flag")]
        strict = [v for v in verdicts.values() if v is not None]
        base = verdicts.get(BASELINE)
        changed = sum(1 for r in RULES
                      if verdicts[r] in ("flag", "no_flag")
                      and base in ("flag", "no_flag")
                      and verdicts[r] != base)
        finite_p = [p for p in pvals.values() if np.isfinite(p)]

        rows.append({
            "method": method,
            "permutation_scheme": scheme,
            "cohort": cohort,
            "cohort_label": meta["cohort_label"],
            "is_clinical": bool(meta["is_clinical"]),
            "n_test": int(meta["n_test"]),
            **{f"conclusion_{r}": verdicts[r] for r in RULES},
            **{f"p_{r}": pvals[r] for r in RULES},
            **{f"stat_{r}": stats[r] for r in RULES},
            "n_rules_evaluable": len(issued),
            "n_distinct_verdicts": len(set(issued)),
            "verdict_constant": len(set(issued)) <= 1,
            "verdict_constant_strict": len(set(strict)) <= 1,
            "n_rules_differing_from_m30": changed,
            "differs_from_m30": changed > 0,
            "rules_flagged": ";".join(r for r in RULES
                                      if verdicts[r] == "flag"),
            "p_min": min(finite_p) if finite_p else np.nan,
            "p_max": max(finite_p) if finite_p else np.nan,
            "p_range": (max(finite_p) - min(finite_p)) if finite_p else np.nan,
            # A verdict flip driven by a p-value that never leaves 0.03-0.07 is
            # a different phenomenon from one driven by a p-value that sweeps
            # the unit interval. Both are instability; only the second is also
            # a substantive disagreement about the evidence.
            "p_straddles_alpha": (bool(finite_p)
                                  and min(finite_p) < 0.05 <= max(finite_p)),
        })
    return pd.DataFrame(rows)


# ── per method ───────────────────────────────────────────────────────────────
def method_frame(df: pd.DataFrame, per_cohort: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for cohort_set, mask in (("all", pd.Series(True, index=df.index)),
                             ("clinical", df["is_clinical"].astype(bool))):
        sub = df[mask]
        pc = per_cohort[per_cohort["is_clinical"]] if cohort_set == "clinical" \
            else per_cohort
        for (method, scheme), g in sub.groupby(
                ["method", "permutation_scheme"], sort=False, dropna=False):
            flagged = {r: set(g[(g["rule"] == r)
                               & (g["conclusion"] == "flag")]["cohort"])
                       for r in RULES}
            gpc = pc[(pc["method"] == method)
                     & (pc["permutation_scheme"] == scheme)]
            base = flagged[BASELINE]
            jac = {r: jaccard(flagged[r], base) for r in RULES}
            vac = all(not flagged[r] for r in RULES)
            n_nonconst = int((~gpc["verdict_constant"]).sum())
            rows.append({
                "method": method,
                "permutation_scheme": scheme,
                "cohort_set": cohort_set,
                "n_cohorts": int(gpc.shape[0]),
                "n_cohorts_with_any_verdict": int(
                    (gpc["n_rules_evaluable"] > 0).sum()),
                # THE headline stability number.
                "n_cohorts_nonconstant_verdict": n_nonconst,
                "n_cohorts_nonconstant_strict": int(
                    (~gpc["verdict_constant_strict"]).sum()),
                "frac_cohorts_nonconstant": (
                    n_nonconst / max(int((gpc["n_rules_evaluable"] > 0).sum()), 1)),
                "cohorts_nonconstant": ";".join(
                    sorted(gpc[~gpc["verdict_constant"]]["cohort"])),
                # The number that made the old claim look true.
                **{f"n_flag_{r}": len(flagged[r]) for r in RULES},
                "n_flag_is_constant": len({len(flagged[r]) for r in RULES}) == 1,
                **{f"flagged_{r}": ";".join(sorted(flagged[r])) for r in RULES},
                "flagged_set_is_constant": len(
                    {frozenset(flagged[r]) for r in RULES}) == 1,
                **{f"jaccard_{r}_vs_m30": jac[r] for r in RULES},
                "mean_jaccard_vs_m30": float(np.mean(
                    [jac[r] for r in RULES if r != BASELINE])),
                "min_jaccard_vs_m30": float(np.min(
                    [jac[r] for r in RULES if r != BASELINE])),
                "jaccard_vacuous": vac,
                "cohorts_flagged_under_all_rules": ";".join(
                    sorted(set.intersection(*flagged.values()))
                    if all(flagged.values()) else []),
                "cohorts_flagged_under_some_rule_only": ";".join(sorted(
                    set.union(*flagged.values())
                    - (set.intersection(*flagged.values())
                       if all(flagged.values()) else set()))),
            })
    out = pd.DataFrame(rows)
    # Honest ranking: fewest cohorts changing verdict first, ties broken by how
    # much of the flagged set survives a change of rule. A method that flags
    # nothing anywhere is trivially stable, so `jaccard_vacuous` is carried
    # alongside the rank rather than silently rewarding it.
    # One ranking per cohort set, over every (method, scheme) row -- the ranked
    # unit is a procedure as actually run, so if both permutation schemes are
    # present they compete in the same ranking rather than in separate ones.
    for _, g in out.groupby("cohort_set", dropna=False):
        order = g.sort_values(
            ["n_cohorts_nonconstant_verdict", "mean_jaccard_vs_m30"],
            ascending=[True, False]).index
        out.loc[order, "stability_rank"] = np.arange(1, len(order) + 1)
    out["stability_rank"] = out["stability_rank"].astype(int)
    return out.sort_values(["cohort_set", "permutation_scheme",
                            "stability_rank"])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", type=str, default=str(IN_CSV))
    ap.add_argument("--out", type=str, default=str(OUT_CSV))
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"{src} missing; run "
                         "`python -m recompute.comparators.run --stage cohorts`")
    df = pd.read_csv(src)
    df = df[df["rule"].isin(RULES)]
    if "permutation_scheme" not in df.columns:
        # Pre-round-2 tables carry no scheme column; every permutation-based row
        # in them was produced under the joint scheme (verified in
        # recompute/scheme_provenance.py).
        df["permutation_scheme"] = np.where(
            df["method"].isin(["permutation_null", "diciccio2020"]), "joint", "")
    df["permutation_scheme"] = df["permutation_scheme"].fillna("")

    per_cohort = cohort_frame(df)
    per_method = method_frame(df, per_cohort)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    per_cohort.to_csv(out, index=False)
    per_method.to_csv(OUT_METHOD_CSV, index=False)
    print(f"wrote {out}  ({len(per_cohort)} rows)")
    print(f"wrote {OUT_METHOD_CSV}  ({len(per_method)} rows)\n")

    for cs in ("all", "clinical"):
        g = per_method[per_method["cohort_set"] == cs]
        print(f"-- stability ranking, {cs} cohorts "
              f"({int(g['n_cohorts_with_any_verdict'].max())} evaluable) "
              "-------------")
        show = g[["stability_rank", "method", "permutation_scheme",
                  "n_cohorts_nonconstant_verdict",
                  "cohorts_nonconstant", "n_flag_is_constant",
                  "flagged_set_is_constant", "mean_jaccard_vs_m30",
                  "min_jaccard_vs_m30", "jaccard_vacuous"]]
        with pd.option_context("display.width", 220,
                               "display.max_colwidth", 46):
            print(show.to_string(index=False))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
