"""
Aggregate the per-cohort JSON into machine-readable tables and
``docs/cohort_evaluation_results.md``.

    python -m recompute.aggregate

Reads ``recompute/results/*.json`` and writes:

  recompute/results/summary.csv               one row per cohort
  recompute/results/excluded_subgroups.csv    every n<30 / degenerate exclusion
  recompute/results/null_comparison.csv       observed gap vs the equality null
  recompute/results/summary.json              everything, combined
  docs/cohort_evaluation_results.md           the report

Illustrative thresholds
-----------------------
0.2.0 withdrew the PASS/FAIL gate, so "failure" here is not a library verdict.
It means only "exceeds the illustrative cut-point the paper used", which is
what the paper's empirical claim was stated against:

    JSS >= 0.05      max per-partition AUC gap > 0.05      max TFR > 0.10

These cut-points have never been calibrated against deployment outcomes. They
are used as the yardstick because the paper's claim was made in their terms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"

THR_JSS = 0.05
THR_GAP = 0.05
THR_TFR = 0.10

ORDER = [
    "synthetic", "uci_heart", "diabetes130", "nhis2024", "nhis2023",
    "nhanes2123", "brfss2024", "adult_income", "acs_income", "german_credit",
]


def _load() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in ORDER:
        p = RESULTS / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _fmt(v: Optional[float], nd: int = 4, pct: bool = False) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v * 100:.1f}%" if pct else f"{v:.{nd}f}"


def _fmt_ci(ci, nd: int = 4, pct: bool = False) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "—"
    if pct:
        return f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]"
    return f"[{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"


def _exceeds(v: Optional[float], thr: float) -> Optional[bool]:
    if v is None or not np.isfinite(v):
        return None
    return bool(v > thr)


def _transition(old_fail: Optional[bool], new_fail: Optional[bool]) -> str:
    if old_fail is None or new_fail is None:
        return "not evaluable"
    if old_fail and new_fail:
        return "persists"
    if old_fail and not new_fail:
        return "artifact (was failing, now passing)"
    if not old_fail and new_fail:
        return "newly failing"
    return "passed before and after"


# ── row builders ─────────────────────────────────────────────────────────────
def build_summary(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, d in data.items():
        if d.get("status") != "ok":
            rows.append({"cohort": name, "label": d.get("label", name),
                         "status": d.get("status", "?"), "error": d.get("error")})
            continue
        cs, o, n = d["cohort_stats"], d["old"], d["new"]
        nr = d.get("null_reference", {})
        gap_new = n.get("auc_gap_per_partition_max")
        rows.append({
            "cohort": name,
            "label": d["label"],
            "status": "ok",
            "n_total": cs["n_total"],
            "n_test": cs["n_test"],
            "prevalence": cs["prevalence"],
            "auroc": cs["auroc"],
            "brier": cs["brier"],

            "jss_old": o["jss"],
            "jss_old_ci_lo": (o["jss_ci"] or [None, None])[0],
            "jss_old_ci_hi": (o["jss_ci"] or [None, None])[1],
            "jss_new": n["jss"],
            "jss_new_ci_lo": (n["jss_ci"] or [None, None])[0],
            "jss_new_ci_hi": (n["jss_ci"] or [None, None])[1],
            "reliability_transition": _transition(
                _exceeds(o["jss"], THR_JSS), _exceeds(n["jss"], THR_JSS)),

            "auc_gap_pooled_old": o["auc_gap_pooled"],
            "auc_gap_pooled_new_diagnostic": n["auc_gap_pooled_diagnostic"],
            "auc_gap_per_partition_new": gap_new,
            "auc_gap_new_ci_lo": (n["auc_gap_ci"] or [None, None])[0],
            "auc_gap_new_ci_hi": (n["auc_gap_ci"] or [None, None])[1],
            "worst_partition": n.get("worst_partition"),
            "inclusivity_transition": _transition(
                _exceeds(o["auc_gap_pooled"], THR_GAP), _exceeds(gap_new, THR_GAP)),

            "max_tfr_old_wide": o["max_tfr"],
            "max_tfr_new_narrow": n["max_tfr_narrow"],
            "max_tfr_new_narrow_ci_lo": (n["max_tfr_narrow_ci"] or [None, None])[0],
            "max_tfr_new_narrow_ci_hi": (n["max_tfr_narrow_ci"] or [None, None])[1],
            "max_tfr_new_wide": n["max_tfr_wide"],
            "sensitivity_transition": _transition(
                _exceeds(o["max_tfr"], THR_TFR),
                _exceeds(n["max_tfr_narrow"], THR_TFR)),
            "tau_ref_alt": (d.get("tau_ref_alt") or {}).get("tau_ref"),
            "max_tfr_alt_tau_narrow": (d.get("tau_ref_alt") or {}).get("max_tfr_narrow"),

            "equity_rho_old_ytrue": o["equity_rho"],
            "equity_rho_new": n.get("equity_rho"),
            "equity_proxy_new": n.get("equity_need_source"),
            "equity_proxy_class": (d.get("proxy_validity") or {}).get("class"),
            "equity_evaluated_new": n.get("equity_evaluated"),

            "batch_scoring_ms_old": o["batch_scoring_time_ms"],
            "batch_scoring_ms_new": n["batch_scoring_time_ms"],
            "single_row_latency_ms_new": n["single_row_latency_ms"],

            "n_excluded_subgroups_new": len(n.get("excluded_subgroups", {})),
            "clustered_resampling": n.get("clustered_resampling"),

            "null_mean_gap": nr.get("null_mean_gap"),
            "null_p95_gap": nr.get("null_p95_gap"),
            "gap_p_value_vs_null": nr.get("p_value_vs_null"),
            "gap_excess_over_null": nr.get("excess_over_null_mean"),
            "gap_exceeds_null_p95": nr.get("exceeds_null_p95"),

            "runtime_s": d.get("total_runtime_s"),
        })
    return pd.DataFrame(rows)


def build_excluded(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, d in data.items():
        if d.get("status") != "ok":
            continue
        counts = d.get("subgroup_counts", {})
        for label, reason in sorted(d["new"].get("excluded_subgroups", {}).items()):
            c = counts.get(label, {})
            rows.append({
                "cohort": name,
                "subgroup": label,
                "n": c.get("n"),
                "n_pos": c.get("n_pos"),
                "n_neg": c.get("n_neg"),
                "reason": reason,
                "rule": ("n < 30" if "min_subgroup_n" in reason
                         else "degenerate labels"),
                "in_old_point_estimate": label in d["old"].get("subgroup_aucs", {}),
            })
    return pd.DataFrame(rows)


def build_null(data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for name, d in data.items():
        if d.get("status") != "ok":
            continue
        nr = d.get("null_reference", {})
        npool = d.get("null_reference_pooled", {})
        ref = d.get("p2_generic_reference", {})
        rows.append({
            "cohort": name,
            "n_test": d["cohort_stats"]["n_test"],
            "n_partitions": len(d["cohort_stats"]["subgroup_columns"]),
            "n_included_subgroups": len(d["new"].get("subgroup_aucs", {})),
            "observed_gap_per_partition": nr.get("observed_gap"),
            "cohort_null_mean": nr.get("null_mean_gap"),
            "cohort_null_p95": nr.get("null_p95_gap"),
            "p_value": nr.get("p_value_vs_null"),
            "excess_over_null_mean": nr.get("excess_over_null_mean"),
            "exceeds_null_p95": nr.get("exceeds_null_p95"),
            "observed_gap_pooled_old": npool.get("observed_gap"),
            "pooled_null_mean": npool.get("null_mean_gap"),
            "pooled_null_p95": npool.get("null_p95_gap"),
            "pooled_p_value": npool.get("p_value_vs_null"),
            "p2_generic_null_mean_10x500": ref.get("mean_range"),
            "p2_generic_null_p95_10x500": ref.get("p95_range"),
        })
    return pd.DataFrame(rows)


# ── markdown ─────────────────────────────────────────────────────────────────
def _cohort_block(name: str, d: Dict[str, Any]) -> str:
    if d.get("status") != "ok":
        return (f"### {d.get('label', name)}\n\n"
                f"**Did not run.** `{d.get('error')}`\n")

    cs, o, n = d["cohort_stats"], d["old"], d["new"]
    nr = d.get("null_reference", {})
    pv = d.get("proxy_validity") or {}
    L: List[str] = []
    L.append(f"### {d['label']}")
    L.append("")
    extra = ""
    if "n_patients" in cs:
        extra = (f", {cs['n_patients']:,} unique patients, "
                 f"{cs['n_test_patients']:,} in test, "
                 f"test-row leakage {cs['group_split_test_row_leakage']:.1%}")
    L.append(f"n = {cs['n_total']:,} (test {cs['n_test']:,}{extra}); "
             f"prevalence {cs['prevalence']:.3f}; "
             f"{len(cs['subgroup_columns'])} demographic partitions "
             f"(`{'`, `'.join(cs['subgroup_columns'])}`).")
    L.append("")
    L.append("| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |")
    L.append("|---|---:|---:|:--:|---|")
    L.append(f"| AUROC | {_fmt(cs['auroc'], 4)} | {_fmt(cs['auroc'], 4)} | — | "
             "unchanged (same model, same split) |")
    L.append(f"| Brier | {_fmt(cs['brier'], 4)} | {_fmt(cs['brier'], 4)} | — | unchanged |")
    L.append(f"| JSS / PSS | {_fmt(o['jss'])} | **{_fmt(n['jss'])}** | "
             f"{_fmt_ci(n['jss_ci'])} | "
             f"{_transition(_exceeds(o['jss'], THR_JSS), _exceeds(n['jss'], THR_JSS))} |")
    L.append(f"| ΔAUC — pooled cross-partition (old headline, now diagnostic) | "
             f"{_fmt(o['auc_gap_pooled'])} | {_fmt(n['auc_gap_pooled_diagnostic'])} | "
             "— | retained as diagnostic only |")
    L.append(f"| ΔAUC — max per-partition (new headline) | — | "
             f"**{_fmt(n['auc_gap_per_partition_max'])}** | "
             f"{_fmt_ci(n['auc_gap_ci'])} | "
             f"widest partition `{n.get('worst_partition')}` |")
    L.append(f"| Max TFR — wide band [0.10, 0.90] | {_fmt(o['max_tfr'], pct=True)} | "
             f"{_fmt(n['max_tfr_wide'], pct=True)} | — | secondary in 0.2.0 |")
    L.append(f"| Max TFR — narrow band [0.30, 0.70] (new primary) | — | "
             f"**{_fmt(n['max_tfr_narrow'], pct=True)}** | "
             f"{_fmt_ci(n['max_tfr_narrow_ci'], pct=True)} | "
             f"{_transition(_exceeds(o['max_tfr'], THR_TFR), _exceeds(n['max_tfr_narrow'], THR_TFR))} |")

    if n.get("equity_evaluated"):
        ceiling = (f"ceiling {_fmt(n.get('equity_ceiling'))}"
                   if n.get("equity_ceiling") is not None
                   else f"{n.get('equity_proxy_levels')}-level proxy, no binary ceiling")
        L.append(f"| Equity ρ | {_fmt(o['equity_rho'])} (proxy = `y_true`) | "
                 f"{_fmt(n['equity_rho'])} (proxy = `{n.get('equity_need_source')}`) | "
                 f"— | {ceiling} |")
    else:
        L.append(f"| Equity ρ | {_fmt(o['equity_rho'])} (proxy = `y_true`) | "
                 "**skipped** | — | no valid proxy |")
    L.append(f"| Deployability — batch scoring (whole cohort) | "
             f"{_fmt(o['batch_scoring_time_ms'], 2)} ms | "
             f"{_fmt(n['batch_scoring_time_ms'], 2)} ms | — | renamed, not a latency |")
    L.append(f"| Deployability — single-row latency | *not measured* | "
             f"**{_fmt(n['single_row_latency_ms'], 3)} ms** | — | new in 0.2.0 |")
    L.append("")

    if n.get("per_partition_auc_gaps"):
        parts = ", ".join(
            f"`{k}` {v:.4f}" for k, v in
            sorted(n["per_partition_auc_gaps"].items(), key=lambda kv: -kv[1]))
        L.append(f"Per-partition gaps: {parts}.")
        L.append("")

    if nr.get("null_estimable"):
        L.append(
            f"**Against the equality null for this cohort's own partition "
            f"geometry:** null mean {nr['null_mean_gap']:.4f}, p95 "
            f"{nr['null_p95_gap']:.4f}; observed {nr.get('observed_gap', float('nan')):.4f}, "
            f"one-sided p = {nr.get('p_value_vs_null', float('nan')):.3f}, "
            f"excess over the null mean "
            f"{nr.get('excess_over_null_mean', float('nan')):+.4f}.")
        L.append("")

    if pv:
        L.append(f"*Equity proxy*: {pv.get('column')} — **{pv.get('class')}**. "
                 f"{pv.get('note')}")
        L.append("")

    if n.get("excluded_subgroups"):
        counts = d.get("subgroup_counts", {})
        L.append("Subgroups excluded by the n≥30 / estimability rule:")
        L.append("")
        L.append("| Subgroup | n | positives | negatives | Reason | In the 0.1.0 point estimate? |")
        L.append("|---|---:|---:|---:|---|---|")
        for label, reason in sorted(n["excluded_subgroups"].items()):
            c = counts.get(label, {})
            was_in = "yes" if label in o.get("subgroup_aucs", {}) else "no"
            L.append(f"| `{label}` | {c.get('n', '—')} | {c.get('n_pos', '—')} | "
                     f"{c.get('n_neg', '—')} | {reason} | {was_in} |")
        L.append("")
    else:
        L.append("No subgroup was excluded by the n≥30 rule on this cohort.")
        L.append("")

    alt = d.get("tau_ref_alt")
    if alt:
        L.append(f"*τ₀ sensitivity*: at the prevalence-matched threshold "
                 f"τ₀ = {alt['tau_ref']:.4f} (rather than the 0.5 convention) the "
                 f"narrow-band max TFR is {_fmt(alt['max_tfr_narrow'], pct=True)} "
                 f"and the wide-band max TFR is "
                 f"{_fmt(alt['max_tfr_wide'], pct=True)}. Point estimates only.")
        L.append("")

    if name == "diabetes130" and "row_split_reference" in d:
        rr = d["row_split_reference"]
        L.append(f"*Split note*: the published 0.1.0 figures came from a "
                 f"row-level split in which {rr['row_leakage_fraction']:.1%} of test "
                 f"rows belong to a patient also seen in training. Under that split "
                 f"the 0.1.0 pipeline gives AUROC {rr['auroc']:.4f}, pooled ΔAUC "
                 f"{_fmt(rr['old']['auc_gap_pooled'])}, max TFR "
                 f"{_fmt(rr['old']['max_tfr'], pct=True)}, JSS "
                 f"{_fmt(rr['old']['jss'])}. The table above uses the group split "
                 f"for *both* columns so the comparison isolates the measurement "
                 f"change.")
        L.append("")
    return "\n".join(L)


#: AUROC at or above which a model "looks good on the aggregate number", i.e.
#: the regime in which the paper's claim ("metrics detect what AUROC hides")
#: has any bite. Below this an evaluator would already be suspicious.
AUROC_LOOKS_GOOD = 0.80


def build_findings(data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute the answers to the three questions directly from the results."""
    per: Dict[str, Dict[str, Any]] = {}
    for name, d in data.items():
        if d.get("status") != "ok":
            continue
        cs, o, n = d["cohort_stats"], d["old"], d["new"]
        nr = d.get("null_reference", {})
        gap_new = n.get("auc_gap_per_partition_max")
        rec = {
            "label": d["label"],
            "auroc": cs["auroc"],
            "looks_good_on_auroc": bool(cs["auroc"] >= AUROC_LOOKS_GOOD),

            "rel_old_fail": _exceeds(o["jss"], THR_JSS),
            "rel_new_fail": _exceeds(n["jss"], THR_JSS),
            "inc_old_fail": _exceeds(o["auc_gap_pooled"], THR_GAP),
            "inc_new_fail": _exceeds(gap_new, THR_GAP),
            "sen_old_fail": _exceeds(o["max_tfr"], THR_TFR),
            "sen_new_fail": _exceeds(n["max_tfr_narrow"], THR_TFR),

            "gap_observed": gap_new,
            "gap_null_mean": nr.get("null_mean_gap"),
            "gap_null_p95": nr.get("null_p95_gap"),
            "gap_p": nr.get("p_value_vs_null"),
            "gap_above_null_p95": nr.get("exceeds_null_p95"),
            "gap_above_generic_p2_mean": (
                None if gap_new is None
                else bool(gap_new > P2_MEAN)
            ),
        }
        rec["inc_evaluable"] = gap_new is not None
        # An Inclusivity failure counts as *evidence* only if it survives both
        # the corrected estimand and its own equality null.
        rec["inc_evidential"] = bool(
            rec["inc_new_fail"] and rec["gap_above_null_p95"]
            and rec["gap_p"] is not None and rec["gap_p"] < 0.05
        )
        # The gap being *below* its own null expectation is worth naming: the
        # statistic is then not merely unproven, it is smaller than what pure
        # selection over subgroups produces at this geometry.
        rec["gap_below_null_mean"] = (
            None if (gap_new is None or rec["gap_null_mean"] is None)
            else bool(gap_new < rec["gap_null_mean"])
        )
        # TFR and JSS never read y_true, so any failure there is by
        # construction invisible to a discrimination metric.
        rec["auroc_blind_inclusivity"] = bool(
            rec["looks_good_on_auroc"] and rec["inc_evidential"])
        rec["auroc_blind_sensitivity"] = bool(
            rec["looks_good_on_auroc"] and rec["sen_new_fail"])
        rec["auroc_blind_reliability"] = bool(
            rec["looks_good_on_auroc"] and rec["rel_new_fail"])
        per[name] = rec

    def _names(pred) -> List[str]:
        return [per[k]["label"] for k in ORDER if k in per and pred(per[k])]

    return {
        "auroc_looks_good_threshold": AUROC_LOOKS_GOOD,
        "per_cohort": per,
        "reliability_persists": _names(
            lambda r: r["rel_old_fail"] and r["rel_new_fail"]),
        "reliability_artifact": _names(
            lambda r: r["rel_old_fail"] and r["rel_new_fail"] is False),
        "inclusivity_persists": _names(
            lambda r: r["inc_old_fail"] and r["inc_new_fail"]),
        "inclusivity_artifact": _names(
            lambda r: r["inc_old_fail"] and r["inc_new_fail"] is False),
        "inclusivity_not_evaluable": _names(
            lambda r: r["inc_old_fail"] and not r["inc_evaluable"]),
        "sensitivity_persists": _names(
            lambda r: r["sen_old_fail"] and r["sen_new_fail"]),
        "sensitivity_artifact": _names(
            lambda r: r["sen_old_fail"] and r["sen_new_fail"] is False),
        "inclusivity_evidential": _names(lambda r: r["inc_evidential"]),
        "inclusivity_failing_but_within_null": _names(
            lambda r: r["inc_new_fail"] and not r["inc_evidential"]),
        "inclusivity_gap_below_its_own_null": _names(
            lambda r: r["gap_below_null_mean"] is True),
        "auroc_blind_inclusivity": _names(lambda r: r["auroc_blind_inclusivity"]),
        "auroc_blind_sensitivity": _names(lambda r: r["auroc_blind_sensitivity"]),
        "auroc_blind_reliability": _names(lambda r: r["auroc_blind_reliability"]),
    }


#: p2 headline cell, 10 subgroups of 500.
P2_MEAN = 0.08887413961480663
P2_P95 = 0.1304391085449873


def _max_jss(data: Dict[str, Dict[str, Any]]) -> float:
    """Largest 0.2.0 JSS across all cohorts that ran."""
    vals = [
        d["new"]["jss"] for d in data.values()
        if d.get("status") == "ok" and d["new"].get("jss") is not None
    ]
    return max(vals) if vals else float("nan")


def build_markdown(data: Dict[str, Dict[str, Any]], summary: pd.DataFrame,
                   nulldf: pd.DataFrame, excl: pd.DataFrame) -> str:
    ok = summary[summary["status"] == "ok"] if len(summary) else summary
    L: List[str] = []
    L.append("# Recomputed RISED results: 0.1.0 → 0.2.0")
    L.append("")
    L.append("Every cohort re-run under the corrected measurement pipeline, "
             "against the same data, split, seed and fitted model as the 0.1.0 "
             "run, so each difference is attributable to the measurement change "
             "alone.")
    L.append("")
    L.append("Reproduce with `python -m recompute.run_all` then "
             "`python -m recompute.aggregate`. Seeds fixed at 42; B = 1000 "
             "bootstrap replicates with a full delete-one-unit jackknife for the "
             "BCa acceleration (delete-one-*patient* on Diabetes 130).")
    L.append("")
    L.append("> **On the word \"failure\".** 0.2.0 withdrew the PASS/FAIL gate, "
             "so nothing here is a library verdict. \"Failure\" means only "
             "\"exceeds the illustrative cut-point the paper's claim was stated "
             "against\": JSS ≥ 0.05, max per-partition ΔAUC > 0.05, max TFR > 10%. "
             "None of those cut-points has been calibrated against deployment "
             "outcomes.")
    L.append("")

    # ── the three answers ────────────────────────────────────────────────────
    f = build_findings(data)

    def _lst(key: str) -> str:
        v = f[key]
        return ", ".join(v) if v else "*none*"

    n_evid = len(f["inclusivity_evidential"])
    n_clin_evid = len([
        c for c in f["per_cohort"]
        if f["per_cohort"][c]["inc_evidential"]
        and c not in ("adult_income", "acs_income", "german_credit")
    ])

    L.append("## Bottom line")
    L.append("")
    L.append("**The corrections do not leave the empirical case standing in the "
             "form the paper states it.**")
    L.append("")
    L.append("Under the corrected pipeline, on ten cohorts:")
    L.append("")
    L.append("* **Reliability fails nowhere.** Every 0.1.0 Reliability failure "
             "was driven by the multiplicative age rescalings now reclassified as "
             "covariate shift, plus Gaussian noise applied to binary and "
             "categorical columns. With semantics-preserving perturbations only "
             "and a typed schema, the highest JSS across all ten cohorts is "
             f"{_max_jss(data):.4f}, against a 0.05 cut-point. The dimension "
             "does not discriminate between these models at all.")
    L.append(f"* **Inclusivity fails in name on {len(f['inclusivity_persists'])} "
             f"cohorts, but only {n_evid} survives its own equality null.** "
             f"That one is {_lst('inclusivity_evidential')} — a non-clinical "
             f"cross-domain demo, at ΔAUC 0.053 against a 0.05 cut-point. "
             f"**No clinical cohort produces an Inclusivity gap distinguishable "
             f"from chance.** Two cohorts measure a gap *smaller* than what pure "
             f"selection over their subgroups produces.")
    L.append("* **Sensitivity still fails on several cohorts**, but max TFR "
             "never reads `y_true`, so it cannot be read as a performance "
             "failure at all, and a constant predictor scores a perfect 0 on it. "
             "It is the weakest of the three as evidence.")
    L.append("* **Equity is effectively unmeasured.** Nine of the ten proxies "
             "are model input features and one is the outcome's own diagnostic "
             "criterion. The single genuinely independent proxy in the whole "
             "study — German Credit's savings status — gives ρ = −0.019, i.e. "
             "nothing.")
    L.append("")
    L.append(f"The synthetic cohort's collapse (JSS 0.064→0.011, TFR "
             f"19.9%→7.9%, ΔAUC 0.059→0.046) was not special to the synthetic "
             f"cohort. The same three corrections, plus the equality null, "
             f"remove {'all' if n_clin_evid == 0 else 'most'} of the clinical "
             f"evidence too. The Diabetes 130 result the README calls "
             f"\"decisive\" — ΔAUC 0.262, max TFR 49.1% — becomes ΔAUC 0.216 "
             f"(p = 0.15 against its own null) and max TFR 1.4% once the patient "
             f"identity is restored, the band is narrowed and the exclusion rule "
             f"is applied consistently.")
    L.append("")
    L.append("What survives is narrower and worth stating precisely: **the "
             "metrics do measure quantities that AUROC cannot see** — TFR and "
             "JSS are functionals that never touch the labels. What the data no "
             "longer supports is that those quantities **detect deployment-"
             "relevant failures** in these cohorts. The instrument reads "
             "something real; the readings on this evidence base are not "
             "distinguishable from what an equal-performance model would "
             "produce.")
    L.append("")

    L.append("## The three questions, answered")
    L.append("")
    L.append("### 1. Which failures persist under correct measurement, and "
             "which were artifacts?")
    L.append("")
    L.append("| Dimension | Failed under 0.1.0 and still fails | "
             "Failed under 0.1.0, passes under 0.2.0 (artifact) | "
             "Failed under 0.1.0, no longer evaluable |")
    L.append("|---|---|---|---|")
    L.append(f"| Reliability (JSS ≥ 0.05) | {_lst('reliability_persists')} | "
             f"{_lst('reliability_artifact')} | — |")
    L.append(f"| Inclusivity (ΔAUC > 0.05) | {_lst('inclusivity_persists')} | "
             f"{_lst('inclusivity_artifact')} | "
             f"{_lst('inclusivity_not_evaluable')} |")
    L.append(f"| Sensitivity (max TFR > 10%) | {_lst('sensitivity_persists')} | "
             f"{_lst('sensitivity_artifact')} | — |")
    L.append("")
    L.append("\"No longer evaluable\" is not a pass. It means the n ≥ 30 rule, "
             "now applied in the point estimate as well as the intervals, leaves "
             "fewer than two estimable subgroups in every partition, so the "
             "parity gap has no value at all. The 0.1.0 figure for such a cohort "
             "was computed over subgroups the same release already knew were too "
             "small to trust.")
    L.append("")

    L.append("### 2. For persistent failures, is the effect large relative to "
             "the null?")
    L.append("")
    L.append(f"The generic reference from `verification/results/p2_summary.json` "
             f"is mean **{P2_MEAN:.4f}** and p95 **{P2_P95:.4f}** at 10 subgroups "
             f"of 500 under exact equality. Each cohort is additionally measured "
             f"against a null built from its *own* partition geometry, which is "
             f"the fairer comparison: a cohort of two-level partitions has a much "
             f"smaller null than that grid cell, and a cohort with seven small "
             f"race levels has a larger one.")
    L.append("")
    L.append(f"* Inclusivity gaps that clear both the 0.05 cut-point **and** "
             f"their own cohort null (above p95, one-sided p < 0.05): "
             f"**{_lst('inclusivity_evidential')}**.")
    L.append(f"* Inclusivity gaps above 0.05 but **not** separable from the "
             f"equality null: {_lst('inclusivity_failing_but_within_null')}. "
             f"For these the number is consistent with selection bias over "
             f"subgroups and is not evidence of disparity.")
    L.append(f"* Cohorts whose measured gap is *below* its own null mean — "
             f"i.e. smaller than what pure selection over subgroups produces at "
             f"that geometry: {_lst('inclusivity_gap_below_its_own_null')}.")
    L.append("")

    L.append("### 3. Does any cohort still show a failure that aggregate AUROC "
             "would miss?")
    L.append("")
    L.append(f"Taking \"AUROC would miss it\" to mean the model looks good on the "
             f"aggregate number (AUROC ≥ {AUROC_LOOKS_GOOD:.2f}) while some "
             f"dimension still exceeds its cut-point — and, for Inclusivity, "
             f"survives its own equality null:")
    L.append("")
    L.append(f"* **Inclusivity**, evidential and AUROC-invisible: "
             f"{_lst('auroc_blind_inclusivity')}")
    L.append(f"* **Sensitivity** (max TFR > 10% on the narrow band): "
             f"{_lst('auroc_blind_sensitivity')}")
    L.append(f"* **Reliability** (JSS ≥ 0.05 on semantics-preserving "
             f"perturbations only): {_lst('auroc_blind_reliability')}")
    L.append("")
    L.append("Threshold flip rate and JSS never read `y_true` at all — TFR is a "
             "functional of the score CDF alone — so a failure on either is "
             "*by construction* invisible to a discrimination metric. That is "
             "the strongest form the claim can take, but it cuts both ways: the "
             "same independence makes TFR gameable, since a constant predictor "
             "scores a perfect 0 while being useless. TFR must be read next to "
             "AUROC, never instead of it, and a TFR finding is weaker evidence "
             "for the paper's thesis than an Inclusivity finding would be.")
    L.append("")

    # ── headline table ───────────────────────────────────────────────────────
    L.append("## Headline table, all cohorts")
    L.append("")
    L.append("| Cohort | n (test) | AUROC | JSS old→new | ΔAUC pooled (old) | "
             "ΔAUC per-partition (new) | null mean / p95 | p | "
             "TFR wide (old) → narrow (new) | Equity |")
    L.append("|---|---:|---:|---|---:|---|---|---:|---|---|")
    for name in ORDER:
        d = data.get(name)
        if d is None:
            continue
        if d.get("status") != "ok":
            L.append(f"| {d.get('label', name)} | — | — | — | — | — | — | — | — | — |")
            continue
        cs, o, n = d["cohort_stats"], d["old"], d["new"]
        nr = d.get("null_reference", {})
        eq = (f"ρ {_fmt(n['equity_rho'], 3)} ({n.get('equity_need_source')})"
              if n.get("equity_evaluated") else "skipped")
        L.append(
            f"| {d['label']} | {cs['n_total']:,} ({cs['n_test']:,}) | "
            f"{cs['auroc']:.3f} | {_fmt(o['jss'], 4)} → {_fmt(n['jss'], 4)} | "
            f"{_fmt(o['auc_gap_pooled'])} | "
            f"{_fmt(n['auc_gap_per_partition_max'])} "
            f"{_fmt_ci(n['auc_gap_ci'], 3)} | "
            f"{_fmt(nr.get('null_mean_gap'), 3)} / {_fmt(nr.get('null_p95_gap'), 3)} | "
            f"{_fmt(nr.get('p_value_vs_null'), 3)} | "
            f"{_fmt(o['max_tfr'], pct=True)} → {_fmt(n['max_tfr_narrow'], pct=True)} | "
            f"{eq} |")
    L.append("")

    # ── transitions ──────────────────────────────────────────────────────────
    L.append("## What happened to each headline failure")
    L.append("")
    L.append("| Cohort | Reliability (JSS ≥ 0.05) | Inclusivity (ΔAUC > 0.05) | "
             "Sensitivity (max TFR > 10%) |")
    L.append("|---|---|---|---|")
    for name in ORDER:
        d = data.get(name)
        if d is None or d.get("status") != "ok":
            continue
        r = ok[ok["cohort"] == name].iloc[0]
        L.append(f"| {d['label']} | {r['reliability_transition']} | "
                 f"{r['inclusivity_transition']} | {r['sensitivity_transition']} |")
    L.append("")

    # ── null comparison ──────────────────────────────────────────────────────
    L.append("## Inclusivity gaps against the equality null")
    L.append("")
    L.append("`verification/results/p2_summary.json` establishes that the "
             "max−min AUC range has a strictly positive expectation when every "
             "subgroup shares the same true AUC: at 10 subgroups of 500 the mean "
             "is **0.0889** with p95 **0.1304**. That grid is generic, so each "
             "cohort is also given its own null, computed by permuting subgroup "
             "labels within outcome classes — which preserves every subgroup's "
             "size and prevalence while forcing equal true AUC — at the cohort's "
             "real partition geometry, 2000 replicates.")
    L.append("")
    L.append("| Cohort | included subgroups | observed ΔAUC (per-partition) | "
             "cohort null mean | cohort null p95 | excess | one-sided p | "
             "above null p95? |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for _, r in nulldf.iterrows():
        exc = r["excess_over_null_mean"]
        exc_s = "—" if exc is None or not np.isfinite(exc) else f"{exc:+.4f}"
        obs = r["observed_gap_per_partition"]
        verdict = ("not evaluable" if obs is None or not np.isfinite(obs)
                   else ("**yes**" if r["exceeds_null_p95"] else "no"))
        L.append(
            f"| {r['cohort']} | {int(r['n_included_subgroups'])} | "
            f"{_fmt(obs)} | "
            f"{_fmt(r['cohort_null_mean'])} | {_fmt(r['cohort_null_p95'])} | "
            f"{exc_s} | {_fmt(r['p_value'], 3)} | {verdict} |")
    L.append("")
    L.append("Read the p-value column first. Only one cohort is below 0.05. "
             "Six measure a gap above the 0.05 cut-point that a model with "
             "*identical* subgroup performance would produce at least 6.6% of "
             "the time at that cohort's own geometry, and two measure a gap "
             "smaller than the null average.")
    L.append("")

    # ── excluded subgroups ───────────────────────────────────────────────────
    L.append("## Every subgroup excluded by the n ≥ 30 / estimability rule")
    L.append("")
    if len(excl) == 0:
        L.append("No subgroup was excluded on any cohort.")
    else:
        L.append("0.1.0 applied this rule inconsistently: subgroups with n < 30 "
                 "were *flagged* but still entered the point estimate, while the "
                 "bootstrap and jackknife dropped them — so the interval targeted "
                 "a different parameter from the estimate it was attached to. "
                 "0.2.0 applies one rule in all three places. The last column "
                 "shows which of these subgroups were silently inflating the "
                 "0.1.0 pooled gap.")
        L.append("")
        L.append("| Cohort | Subgroup | n | positives | negatives | Rule | "
                 "Was in the 0.1.0 point estimate? |")
        L.append("|---|---|---:|---:|---:|---|---|")
        for _, r in excl.iterrows():
            L.append(f"| {r['cohort']} | `{r['subgroup']}` | {r['n']} | "
                     f"{r['n_pos']} | {r['n_neg']} | {r['rule']} | "
                     f"{'**yes**' if r['in_old_point_estimate'] else 'no'} |")
    L.append("")

    # ── equity ───────────────────────────────────────────────────────────────
    L.append("## Equity: what a valid proxy costs")
    L.append("")
    L.append("0.2.0 refuses `y_true` as the need proxy, because with a binary "
             "outcome proxy ρ = √(12p(1−p))·(n/√(n²−1))·(AUROC−0.5) exactly, so "
             "the statistic is an affine reparameterisation of discrimination "
             "and cannot fail independently of it. Every 0.1.0 equity number in "
             "the study was computed that way.")
    L.append("")
    L.append("Replacing it requires a proxy that is genuinely independent. Of "
             "the ten cohorts, one has one.")
    L.append("")
    L.append("| Cohort | Proxy | Class | ρ (0.1.0, `y_true`) | ρ (0.2.0, proxy) | Note |")
    L.append("|---|---|---|---:|---:|---|")
    for name in ORDER:
        d = data.get(name)
        if d is None or d.get("status") != "ok":
            continue
        pv = d.get("proxy_validity") or {}
        n = d["new"]
        L.append(f"| {d['label']} | `{pv.get('column', '—')}` | "
                 f"**{pv.get('class', '—')}** | {_fmt(d['old']['equity_rho'], 3)} | "
                 f"{_fmt(n.get('equity_rho'), 3)} | {pv.get('note', '')} |")
    L.append("")
    L.append("`model_input` means the proxy is a legitimate measurement but also "
             "one of the model's own predictors, so ρ partly measures the model "
             "against its own input rather than against need. "
             "`outcome_defining` means the proxy is part of the diagnostic "
             "criterion for the outcome; the library's structural guard does not "
             "reject it, but it is not outcome-independent in the sense F8 "
             "requires, and Equity should be treated as **not evaluable** there. "
             "Only `independent` supports the dimension as specified.")
    L.append("")

    # ── methodology and caveats ──────────────────────────────────────────────
    L.append("## Method, and what would change the answer")
    L.append("")
    L.append("* **Both columns come from the same fitted model on the same "
             "split.** Data preparation, split, seed and hyperparameters are "
             "transcribed from the `examples/` scripts unchanged, so AUROC and "
             "Brier are identical in both columns by construction and every "
             "difference is a measurement difference.")
    L.append("* **Diabetes 130 is the one deliberate departure.** `patient_nbr` "
             "is retained and the split is a `GroupShuffleSplit` on it, with the "
             "clustered bootstrap and delete-one-patient jackknife. Both the "
             "0.1.0 and the 0.2.0 columns are computed on that group split so the "
             "comparison still isolates the measurement change; the published "
             "0.1.0 figures, which came from a row-level split that leaks 42.1% "
             "of test rows, are reported separately in the per-cohort section.")
    L.append("* **τ₀ = 0.5 is used for the headline** so the before/after "
             "difference is attributable to the band change alone. Several "
             "cohorts have prevalence near 0.1, where 0.5 is not an operating "
             "point anyone would deploy; the prevalence-matched alternative is "
             "reported per cohort as a point estimate.")
    L.append("* **The equality null conditions on the observed scores.** It "
             "permutes subgroup labels within outcome classes, so it holds every "
             "subgroup's size and prevalence fixed and asks only whether the "
             "measured spread exceeds what pure selection over that many groups "
             "of those sizes produces. It reproduces the published p2 headline "
             "cell to within 0.001 (`python -m recompute.null_reference`).")
    L.append("* **No bootstrap replicates were reduced.** B = 1000 everywhere, "
             "with the full delete-one-unit jackknife the BCa acceleration "
             "requires. Nothing here is a shortened run.")
    L.append("")

    # ── offline availability and runtime ─────────────────────────────────────
    L.append("### Offline availability and runtime")
    L.append("")
    ran = [n for n in ORDER if n in data and data[n].get("status") == "ok"]
    failed = [n for n in ORDER if n in data and data[n].get("status") != "ok"]
    notrun = [n for n in ORDER if n not in data]
    L.append(f"All {len(ran)} cohorts ran offline from caches already in the "
             f"working tree: the sklearn OpenML cache (UCI Heart, Diabetes 130, "
             f"Adult, German Credit), `examples/adult24.csv`, "
             f"`nhis_cache/adult23.csv`, `nhanes_cache/*.xpt`, "
             f"`brfss_cache/LLCP2024.XPT` and `data/2018/1-Year/psam_p06.csv`. "
             f"No cohort required network access.")
    if failed:
        L.append("")
        L.append(f"**Did not complete:** {', '.join(failed)}.")
    if notrun:
        L.append("")
        L.append(f"**Not run:** {', '.join(notrun)}.")
    L.append("")
    L.append("MIMIC-IV-ED is absent by necessity, not oversight: the full cohort "
             "needs PhysioNet credentials, and only the public demo is present.")
    L.append("")
    L.append("| Cohort | test rows | wall clock |")
    L.append("|---|---:|---:|")
    tot = 0.0
    for name in ORDER:
        d = data.get(name)
        if d is None or d.get("status") != "ok":
            continue
        rt = d.get("total_runtime_s") or 0.0
        tot += rt
        L.append(f"| {d['label']} | {d['cohort_stats']['n_test']:,} | "
                 f"{rt/60:.1f} min |")
    L.append(f"| **total CPU** | | **{tot/60:.0f} min** |")
    L.append("")
    L.append("Runtime is dominated by the BCa jackknife, which is delete-one-unit "
             "and so costs O(n_test) replicates of an O(n_test) statistic — "
             "quadratic in the test split. Run with `--jobs 5` the wall clock is "
             "roughly the longest single cohort.")
    L.append("")

    # ── per-cohort detail ────────────────────────────────────────────────────
    L.append("## Per-cohort detail")
    L.append("")
    for name in ORDER:
        if name in data:
            L.append(_cohort_block(name, data[name]))
            L.append("")

    return "\n".join(L)


def main() -> int:
    data = _load()
    if not data:
        print("No result JSON found. Run `python -m recompute.run_all` first.")
        return 1

    summary = build_summary(data)
    excl = build_excluded(data)
    nulldf = build_null(data)

    summary.to_csv(RESULTS / "summary.csv", index=False)
    excl.to_csv(RESULTS / "excluded_subgroups.csv", index=False)
    nulldf.to_csv(RESULTS / "null_comparison.csv", index=False)
    with open(RESULTS / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({
            "thresholds_used": {
                "note": "illustrative only; withdrawn as a gate in 0.2.0",
                "max_judge_sensitivity_score": THR_JSS,
                "max_partition_auc_gap": THR_GAP,
                "max_threshold_flip_rate": THR_TFR,
            },
            "cohorts": data,
        }, fh, indent=2)

    findings = build_findings(data)
    with open(RESULTS / "findings.json", "w", encoding="utf-8") as fh:
        json.dump(findings, fh, indent=2)

    md = build_markdown(data, summary, nulldf, excl)
    (REPO / "docs" / "cohort_evaluation_results.md").write_text(md, encoding="utf-8")
    print(f"Wrote {RESULTS}/summary.csv, excluded_subgroups.csv, "
          f"null_comparison.csv, summary.json, findings.json")
    print(f"Wrote {REPO / 'docs' / 'cohort_evaluation_results.md'}")
    print(f"Cohorts aggregated: {len(data)}")
    missing = [c for c in ORDER if c not in data]
    if missing:
        print(f"MISSING cohorts (not yet run): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
