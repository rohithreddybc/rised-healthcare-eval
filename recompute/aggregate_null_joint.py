"""
Aggregate the joint-vs-independent null and the m_min sweep.

    python -m recompute.aggregate_null_joint

Reads ``recompute/results/null_joint/*.json`` and writes

  * ``recompute/results/null_comparison_joint.csv``  -- headline rule (m30),
    independent vs joint side by side, with MC SEs and multiplicity control
  * ``recompute/results/null_sweep_mmin.csv``        -- every scheme x rule
  * ``recompute/results/null_joint_combined.csv``    -- Stouffer / Fisher
  * ``docs/permutation_null_specification.md``       -- the write-up

Multiplicity is controlled across the estimable cohorts within each
(scheme, rule) cell -- nine tests at the published m30 rule.

What the ``scheme`` column means, and what it does not
------------------------------------------------------
``scheme`` appears in ``null_sweep_mmin.csv``, ``null_joint_combined.csv`` and
``null_joint_sign_tests.csv``, and in all three it means exactly one thing: **how
the demographic columns were permuted when the per-cohort p-values were
computed.**

``independent``
    a fresh within-outcome-class permutation for each demographic column
    separately, which destroys the association between age, sex, race,
    insurance and income.
``joint``
    one within-outcome-class permutation of the row indices per replicate,
    carried across every demographic column, which preserves the joint
    contingency table exactly.

It is **not** a statement about the combination step. In
``null_joint_combined.csv`` both rows -- ``scheme=independent`` and
``scheme=joint`` -- combine their k cohort p-values with the *same* estimator:
:func:`stouffer` computes ``z = sum(z_i) / sqrt(k)`` with equal weights, which is
Stouffer's method **under the assumption that the k p-values are independent**,
and :func:`fisher` likewise assumes independence. Neither row uses a
dependence-corrected combination. That independence assumption concerns the ten
cohorts being separate datasets -- which they are, so the assumption is
reasonable -- and is entirely unrelated to the permutation scheme. The two rows
differ only in how the inputs to the combination were generated, never in how
they were combined.

``recompute/scheme_provenance.py`` writes the same statement to
``recompute/results/scheme_provenance.csv`` for every published artefact.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, norm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
IN_DIR = HERE / "results" / "null_joint"
RESULTS = HERE / "results"

SCHEMES = ("independent", "joint")
RULES = ("m20", "m30", "m50", "m100", "ev10")
HEADLINE_RULE = "m30"

ORDER = [
    "synthetic", "uci_heart", "diabetes130", "nhis2024", "nhis2023",
    "nhanes2123", "brfss2024", "adult_income", "acs_income", "german_credit",
]


# ── multiplicity ─────────────────────────────────────────────────────────────
def holm(p: List[float]) -> List[float]:
    """Holm-Bonferroni step-down adjusted p-values (FWER)."""
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (k - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return list(adj)


def benjamini_hochberg(p: List[float]) -> List[float]:
    """Benjamini-Hochberg step-up adjusted p-values (FDR)."""
    k = len(p)
    order = np.argsort(p)
    adj = np.empty(k, dtype=float)
    running = 1.0
    for rank in range(k - 1, -1, -1):
        idx = order[rank]
        val = k / (rank + 1) * p[idx]
        running = min(running, val)
        adj[idx] = min(running, 1.0)
    return list(adj)


def stouffer(p: List[float]) -> Dict[str, float]:
    """Stouffer's z for one-sided p-values, equal weights.

    ``z = sum(z_i) / sqrt(k)`` is the variance of the sum under **independence**
    of the k p-values. This is used identically for both permutation schemes;
    the ``scheme`` column of the output says how the inputs were generated and
    says nothing about this step. ``combination_assumes`` is emitted alongside so
    the assumption travels with the number.
    """
    z = np.array([norm.isf(v) for v in p], dtype=float)
    zc = float(np.sum(z) / math.sqrt(len(z)))
    return {"stouffer_z": zc, "stouffer_p": float(norm.sf(zc)), "k": len(p),
            "combination_assumes": "independent p-values across cohorts"}


def fisher(p: List[float]) -> Dict[str, float]:
    """Fisher's combined probability test; also assumes independent p-values."""
    stat = float(-2.0 * np.sum(np.log(np.asarray(p, dtype=float))))
    return {"fisher_chi2": stat, "fisher_df": 2 * len(p),
            "fisher_p": float(chi2.sf(stat, 2 * len(p))), "k": len(p)}


# ── loading ──────────────────────────────────────────────────────────────────
def load() -> Dict[str, Dict[str, Any]]:
    out = {}
    for f in sorted(IN_DIR.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("status") == "ok":
            out[d["cohort"]] = d
    return out


def _cell(d: Dict[str, Any], scheme: str, rule: str) -> Dict[str, Any]:
    return d["results"][scheme][rule]


def _estimable(e: Dict[str, Any]) -> bool:
    return bool(e.get("null_estimable")) and e.get("p_value_vs_null") is not None


# ── long-format sweep table ──────────────────────────────────────────────────
def sweep_frame(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name in ORDER:
        d = data.get(name)
        if d is None:
            continue
        for scheme in SCHEMES:
            for rule in RULES:
                e = _cell(d, scheme, rule)
                lv = d["n_levels_used_by_rule"][rule]
                rows.append({
                    "cohort": name,
                    "is_clinical": d["is_clinical"],
                    "scheme": scheme,
                    "rule": rule,
                    "rule_label": e["rule_label"],
                    "n_test": d["n_test"],
                    "prevalence": d["prevalence"],
                    "n_levels_admitted": int(sum(lv.values())),
                    "n_partitions_usable": int(sum(1 for v in lv.values()
                                                   if v >= 2)),
                    "observed_gap": e.get("observed_gap"),
                    "null_mean": e.get("null_mean_gap"),
                    "null_median": e.get("null_median_gap"),
                    "null_sd": e.get("null_sd_gap"),
                    "null_skew": e.get("null_skew_gap"),
                    "null_p95_mde": e.get("null_p95_gap"),
                    "null_frac_below_own_mean": e.get(
                        "null_frac_below_own_mean"),
                    "p_value": e.get("p_value_vs_null"),
                    "p_mc_se": e.get("p_value_mc_se"),
                    "p_report": e.get("p_report"),
                    "p_is_floor": e.get("p_is_floor"),
                    "n_reps": e.get("n_reps"),
                    "exceeds_null_p95": e.get("exceeds_null_p95"),
                    "exceeds_null_median": e.get("exceeds_null_median"),
                })
    df = pd.DataFrame(rows)

    # Multiplicity within each (scheme, rule) cell, over estimable cohorts.
    df["p_holm"] = np.nan
    df["p_bh"] = np.nan
    for (scheme, rule), grp in df.groupby(["scheme", "rule"]):
        ok = grp["p_value"].notna()
        idx = grp.index[ok]
        if len(idx) == 0:
            continue
        pv = list(df.loc[idx, "p_value"].astype(float))
        df.loc[idx, "p_holm"] = holm(pv)
        df.loc[idx, "p_bh"] = benjamini_hochberg(pv)
    df["sig_raw"] = df["p_value"] < 0.05
    df["sig_holm"] = df["p_holm"] < 0.05
    df["sig_bh"] = df["p_bh"] < 0.05
    return df


# ── headline side-by-side table ──────────────────────────────────────────────
def comparison_frame(df: pd.DataFrame) -> pd.DataFrame:
    ind = df[(df["scheme"] == "independent") & (df["rule"] == HEADLINE_RULE)]
    joi = df[(df["scheme"] == "joint") & (df["rule"] == HEADLINE_RULE)]
    ind = ind.set_index("cohort")
    joi = joi.set_index("cohort")
    keep = [c for c in ORDER if c in ind.index]
    out = pd.DataFrame({
        "cohort": keep,
        "is_clinical": ind.loc[keep, "is_clinical"].values,
        "n_test": ind.loc[keep, "n_test"].values,
        "prevalence": ind.loc[keep, "prevalence"].values,
        "n_reps": ind.loc[keep, "n_reps"].values,
        "observed_gap": ind.loc[keep, "observed_gap"].values,
        # old = published independent-per-column null
        "old_null_mean": ind.loc[keep, "null_mean"].values,
        "old_null_median": ind.loc[keep, "null_median"].values,
        "old_null_skew": ind.loc[keep, "null_skew"].values,
        "old_null_p95_mde": ind.loc[keep, "null_p95_mde"].values,
        "old_p_value": ind.loc[keep, "p_value"].values,
        "old_p_mc_se": ind.loc[keep, "p_mc_se"].values,
        "old_p_report": ind.loc[keep, "p_report"].values,
        "old_p_holm": ind.loc[keep, "p_holm"].values,
        "old_p_bh": ind.loc[keep, "p_bh"].values,
        # new = joint row permutation
        "new_null_mean": joi.loc[keep, "null_mean"].values,
        "new_null_median": joi.loc[keep, "null_median"].values,
        "new_null_skew": joi.loc[keep, "null_skew"].values,
        "new_null_p95_mde": joi.loc[keep, "null_p95_mde"].values,
        "new_p_value": joi.loc[keep, "p_value"].values,
        "new_p_mc_se": joi.loc[keep, "p_mc_se"].values,
        "new_p_report": joi.loc[keep, "p_report"].values,
        "new_p_holm": joi.loc[keep, "p_holm"].values,
        "new_p_bh": joi.loc[keep, "p_bh"].values,
    })
    out["delta_p"] = out["new_p_value"] - out["old_p_value"]
    out["null_p95_shrinkage"] = (
        out["old_null_p95_mde"] - out["new_null_p95_mde"])
    out["sig_old_raw"] = out["old_p_value"] < 0.05
    out["sig_new_raw"] = out["new_p_value"] < 0.05
    out["sig_new_holm"] = out["new_p_holm"] < 0.05
    out["sig_new_bh"] = out["new_p_bh"] < 0.05
    return out


# ── combined clinical test ───────────────────────────────────────────────────
def combined_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in SCHEMES:
        for rule in RULES:
            g = df[(df["scheme"] == scheme) & (df["rule"] == rule)
                   & df["is_clinical"] & df["p_value"].notna()]
            if len(g) == 0:
                continue
            pv = list(g["p_value"].astype(float))
            r = {"scheme": scheme,
                 # Spelled out in the row itself so a reader of the CSV alone
                 # cannot mistake `scheme` for an assumption about the
                 # combination step. See the module docstring.
                 "scheme_means": ("demographic-column permutation scheme used "
                                  "to compute the per-cohort p-values"),
                 "rule": rule,
                 "cohorts": ";".join(g["cohort"]), "k": len(pv)}
            r.update(stouffer(pv))
            r.update(fisher(pv))
            r["any_p_at_floor"] = bool(g["p_is_floor"].any())
            rows.append(r)
    return pd.DataFrame(rows)


# ── skewness / sign diagnostics ──────────────────────────────────────────────
def sign_frame(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scheme in SCHEMES:
        for rule in RULES:
            g = df[(df["scheme"] == scheme) & (df["rule"] == rule)
                   & df["p_value"].notna()]
            if len(g) == 0:
                continue
            k = len(g)
            n_above_med = int(g["exceeds_null_median"].sum())
            n_above_mean = int((g["observed_gap"] > g["null_mean"]).sum())
            bt = binomtest(n_above_med, k, 0.5, alternative="two-sided")
            # Against the null MEAN the reference is NOT 0.5: because the
            # statistic is right-skewed, P(draw > own mean) < 0.5. Average
            # that per-cohort probability and test the observed count against
            # it. One-sided: the manuscript's reading is that too FEW cohorts
            # sit above, so the alternative of interest is "too many".
            p_above_mean_h0 = float(
                1.0 - g["null_frac_below_own_mean"].mean())
            bt_mean = binomtest(n_above_mean, k, p_above_mean_h0,
                                alternative="greater")
            rows.append({
                "scheme": scheme, "rule": rule, "k_cohorts": k,
                "n_obs_above_null_median": n_above_med,
                "n_obs_above_null_mean": n_above_mean,
                "expected_above_median_under_H0": 0.5 * k,
                # Under H0 the statistic is right-skewed, so a MAJORITY of
                # draws sit below their own mean. That is the number the
                # "cohorts fall below their null mean" reading must be
                # compared against -- not 50%.
                "mean_expected_frac_below_null_mean_under_H0": float(
                    g["null_frac_below_own_mean"].mean()),
                "expected_below_null_mean_under_H0": float(
                    g["null_frac_below_own_mean"].sum()),
                "n_obs_below_null_mean": int(k - n_above_mean),
                "binom_p_above_median_vs_half": float(bt.pvalue),
                "p_above_null_mean_under_H0": p_above_mean_h0,
                "expected_above_null_mean_under_H0": p_above_mean_h0 * k,
                "binom_p_above_mean_vs_H0": float(bt_mean.pvalue),
                "median_null_skew": float(g["null_skew"].median()),
            })
    return pd.DataFrame(rows)


def main() -> int:
    data = load()
    if not data:
        raise SystemExit(f"no inputs in {IN_DIR}")
    RESULTS.mkdir(parents=True, exist_ok=True)

    sweep = sweep_frame(data)
    comp = comparison_frame(sweep)
    comb = combined_frame(sweep)
    sign = sign_frame(sweep)

    comp.to_csv(RESULTS / "null_comparison_joint.csv", index=False)
    sweep.to_csv(RESULTS / "null_sweep_mmin.csv", index=False)
    comb.to_csv(RESULTS / "null_joint_combined.csv", index=False)
    sign.to_csv(RESULTS / "null_joint_sign_tests.csv", index=False)
    print("wrote null_comparison_joint.csv, null_sweep_mmin.csv, "
          "null_joint_combined.csv, null_joint_sign_tests.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
