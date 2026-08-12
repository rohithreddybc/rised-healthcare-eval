"""Guards for the comparator reporting layer.

One test (or group) per invariant the reported tables depend on:

  * the p-value floor is rendered as an inequality, never as an attained value;
  * the Lum bootstrap-CI rule runs at the same family-wise level as the z-test
    and Cochran's Q, and both call paths agree;
  * the variance floor's dropped pairs are counted and reported;
  * the rule-stability metric detects flagged-set churn that a count misses;
  * cross-cohort Holm and BH adjustment is attached to the main table;
  * the scheme-provenance and model-provenance artefacts say what they found.

Permutation-null reproducibility has its own file,
``tests/test_type1_reproducibility.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recompute.comparators.core import MethodResult, p_report

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "recompute" / "results"


# ── defect 7d: p-values censored at the 1/(B+1) floor ────────────────────────
def test_p_report_renders_the_floor_as_an_inequality():
    # Rendered with the same two-decimal exponent format as
    # recompute.null_reference.mc_pvalue, rounded UP so the inequality holds.
    assert p_report(1.0 / 10001, 10_000, True) == "<= 1.00e-04"
    assert float(p_report(1.0 / 10001, 10_000, True)[3:]) >= 1.0 / 10001
    assert p_report(1.0 / 1000, 999, True) == "<= 1.00e-03"
    # A value at or below the floor is rendered as an inequality even if the
    # caller forgot to set the flag: the floor is a property of B, not of a bool.
    assert p_report(1.0 / 10001, 10_000, False).startswith("<=")
    # Ordinary values are unaffected.
    assert p_report(0.0432, 10_000, False) == "0.0432"
    assert p_report(None, 10_000, False) == "n/a"
    assert p_report(float("nan"), 10_000, False) == "n/a"


def test_p_report_never_emits_a_string_that_rounds_to_zero():
    for b in (99, 999, 9_999, 10_000, 99_999):
        s = p_report(1.0 / (b + 1), b, True)
        assert s.startswith("<= "), s
        assert float(s[3:]) > 0.0
        # The old behaviour formatted the floor with %.4f, i.e. "0.0001" at
        # B=10,000 and "0.0000" at B=99,999. Neither may reappear.
        assert s not in ("0.0000", "0.000", "0")


def test_method_result_carries_the_floor_into_the_csv_row():
    r = MethodResult("m", "m30", "flag", p_value=1.0 / 10001, p_is_floor=True,
                     n_perm=10_000)
    row = r.as_row()
    assert row["p_value_report"] == "<= 1.00e-04"
    assert row["p_floor"] == pytest.approx(1.0 / 10001)
    # Deterministic methods have no floor and must not pretend to one.
    d = MethodResult("four_fifths", "m30", "no_flag")
    assert d.as_row()["p_floor"] is None
    assert d.as_row()["p_value_report"] == "n/a"


# ── defect 6: equal nominal level across comparators ─────────────────────────
def test_bootstrap_ci_one_sided_level_is_explicit_and_honoured():
    from recompute.comparators.lum import bootstrap_ci

    th = [0.70, 0.74, 0.78, 0.81]
    v = [4e-4, 3e-4, 5e-4, 6e-4]
    a = bootstrap_ci(th, v, one_sided_alpha=0.0125, n_boot=200_000, seed=1)
    assert a["level_lo"] == pytest.approx(0.0125)
    # `conf` alone puts the lower bound at (1-conf)/2, HALF the nominal level --
    # which is precisely the bug. The two must therefore disagree, and the
    # explicit one-sided bound must be the looser (higher) of the two.
    b = bootstrap_ci(th, v, conf=1.0 - 0.0125, n_boot=200_000, seed=1)
    assert b["level_lo"] == pytest.approx(0.00625)
    assert a["lo"] > b["lo"]


def test_lum_bootstrap_uses_alpha_over_P_in_both_call_paths():
    """The cohort path and the Type I path must run at the same level."""
    import inspect

    from recompute.comparators import lum

    for fn in (lum.run_cohort, lum.decide):
        src = inspect.getsource(fn)
        assert "one_sided_alpha" in src, fn.__name__
        assert "conf=1.0 - alpha / P" not in src, fn.__name__
        assert "alpha / P" in src or "alpha / p_parts" in src, fn.__name__


def test_lum_bootstrap_level_is_stated_in_the_output():
    """The level must be auditable from the artefact, not only the source."""
    from recompute.comparators.lum import run_cohort

    rng = np.random.default_rng(3)
    n = 4000
    y = (rng.random(n) < 0.3).astype(int)
    s = rng.random(n)
    codes = {"a": rng.integers(0, 3, n), "b": rng.integers(0, 2, n)}

    class _D:
        pass

    d = _D()
    d.y, d.s, d.codes_by_col = y, s, codes
    out = run_cohort(d, rules=["m30"], alpha=0.05, n_boot=500, seed=1)
    detail = out["variants"]["lum2022_bootstrapCI"]["m30"].detail
    assert "one-sided level alpha/P" in detail
    assert "family-wise level is alpha" in detail
    # Two usable partitions, so the one-sided level is alpha/P = 0.025 in every
    # one of them -- not 0.05/2/2 = 0.0125, which is what `conf = 1 - alpha/P`
    # produced, and not 0.025-with-no-adjustment, which is what the cohort path
    # produced before the fix (those two coincide only at P = 2, which is why
    # this test uses P = 2 and checks the recorded level rather than the flag).
    diag = out["diagnostics"]["m30"]
    assert len(diag) == 2
    for col, res in diag.items():
        assert res["ci_level_lo"] == pytest.approx(0.05 / len(diag)), col


# ── defect 3: rule stability ─────────────────────────────────────────────────
def test_jaccard_definition():
    from recompute.comparators.rule_stability import jaccard

    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard(set(), set()) == 1.0                # vacuous, but correct
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b", "c"}, {"b", "c", "d"}) == pytest.approx(0.5)


def test_stability_metric_catches_churn_that_a_count_misses():
    """The exact failure mode the previous claim was blind to.

    Three cohorts flagged under every rule, but a different three each time.
    A count says 'perfectly stable'; the verdict-concordance metric says three
    cohorts change verdict, which is the truth.
    """
    from recompute.comparators.rule_stability import (RULES, cohort_frame,
                                                      method_frame)

    flagged = {"m20": {"A", "B", "C"}, "m30": {"A", "B", "C"},
               "m50": {"A", "B", "C"}, "m100": {"A", "C", "D"},
               "ev10": {"A", "B", "D"}}
    rows = []
    for rule in RULES:
        for c in "ABCD":
            rows.append({
                "method": "toy", "permutation_scheme": "joint", "cohort": c,
                "cohort_label": c, "is_clinical": True, "n_test": 100,
                "rule": rule,
                "conclusion": "flag" if c in flagged[rule] else "no_flag",
                "p_value": 0.01 if c in flagged[rule] else 0.5,
                "statistic": 1.0,
            })
    df = pd.DataFrame(rows)
    pc = cohort_frame(df)
    mf = method_frame(df, pc)
    row = mf[mf["cohort_set"] == "all"].iloc[0]

    assert row["n_flag_is_constant"], "the count is constant -- that is the trap"
    assert not row["flagged_set_is_constant"]
    assert row["n_cohorts_nonconstant_verdict"] == 3
    assert set(row["cohorts_nonconstant"].split(";")) == {"B", "C", "D"}
    assert row["cohorts_flagged_under_all_rules"] == "A"
    assert row["mean_jaccard_vs_m30"] < 1.0


def test_stability_metric_reports_a_genuinely_stable_method_as_stable():
    from recompute.comparators.rule_stability import (RULES, cohort_frame,
                                                      method_frame)

    rows = [{
        "method": "toy", "permutation_scheme": "", "cohort": c,
        "cohort_label": c, "is_clinical": True, "n_test": 100, "rule": rule,
        "conclusion": "flag" if c in {"A", "B"} else "no_flag",
        "p_value": 0.01 if c in {"A", "B"} else 0.5, "statistic": 1.0,
    } for rule in RULES for c in "ABCD"]
    mf = method_frame(pd.DataFrame(rows), cohort_frame(pd.DataFrame(rows)))
    row = mf[mf["cohort_set"] == "all"].iloc[0]
    assert row["n_cohorts_nonconstant_verdict"] == 0
    assert row["flagged_set_is_constant"]
    assert row["mean_jaccard_vs_m30"] == 1.0
    assert not row["jaccard_vacuous"]


def test_not_evaluable_is_reported_under_both_conventions():
    from recompute.comparators.rule_stability import RULES, cohort_frame

    # Evaluable and no_flag under three rules, non-evaluable under two.
    seq = {"m20": "no_flag", "m30": "no_flag", "m50": "no_flag",
           "m100": "not_evaluable", "ev10": "not_evaluable"}
    df = pd.DataFrame([{
        "method": "toy", "permutation_scheme": "", "cohort": "A",
        "cohort_label": "A", "is_clinical": True, "n_test": 10, "rule": r,
        "conclusion": seq[r], "p_value": np.nan, "statistic": np.nan,
    } for r in RULES])
    row = cohort_frame(df).iloc[0]
    assert row["verdict_constant"]                 # decisions issued all agree
    assert not row["verdict_constant_strict"]      # but evaluability moved


# ── defect 7b: cross-cohort multiplicity ─────────────────────────────────────
def test_cross_cohort_multiplicity_is_attached_and_is_never_anti_conservative():
    from recompute.comparators.run import add_cross_cohort_multiplicity

    df = pd.DataFrame([
        {"method": "m", "rule": "m30", "permutation_scheme": "joint",
         "cohort": c, "p_value": p, "conclusion": "flag" if p < 0.05 else "no_flag"}
        for c, p in zip("ABCDE", [0.001, 0.02, 0.04, 0.30, 0.90])])
    out = add_cross_cohort_multiplicity(df, alpha=0.05)
    assert (out["p_holm_across_cohorts"] >= out["p_value"]).all()
    assert (out["p_bh_across_cohorts"] >= out["p_value"]).all()
    assert (out["p_holm_across_cohorts"] >= out["p_bh_across_cohorts"]).all()
    # 0.001*5 = 0.005 survives; 0.02*4 = 0.08 and 0.04*3 = 0.12 do not.
    assert int(out["flag_raw"].sum()) == 3
    assert int(out["flag_holm"].sum()) == 1
    assert out["multiplicity_applies"].all()


def test_multiplicity_is_marked_not_applicable_without_a_p_value():
    from recompute.comparators.run import add_cross_cohort_multiplicity

    df = pd.DataFrame([
        {"method": "four_fifths", "rule": "m30", "permutation_scheme": "",
         "cohort": c, "p_value": np.nan, "conclusion": v}
        for c, v in zip("AB", ["flag", "no_flag"])])
    out = add_cross_cohort_multiplicity(df, alpha=0.05)
    assert not out["multiplicity_applies"].any()
    assert list(out["flag_holm"]) == [True, False]
    assert list(out["flag_bh"]) == [True, False]


# ── defect 2 and 7f: the provenance artefacts ────────────────────────────────
@pytest.mark.skipif(not (RESULTS / "scheme_provenance.csv").exists(),
                    reason="run python -m recompute.scheme_provenance")
def test_scheme_provenance_is_conclusive_everywhere():
    df = pd.read_csv(RESULTS / "scheme_provenance.csv")
    assert not df["evidence"].str.contains("INCONCLUSIVE").any()
    comp = df[df["result_file"].str.contains("comparator_", na=False)]
    assert (comp["scheme_used"].str.contains("joint")
            | comp["scheme_used"].str.contains("n/a")).all()
    older = df[df["result_file"].str.contains("summary.csv", na=False)]
    assert (older["scheme_used"] == "independent").all()


def test_stouffer_assumes_independence_in_both_scheme_rows():
    """`scheme` is the permutation scheme, NOT a combination assumption."""
    from recompute.aggregate_null_joint import stouffer

    out = stouffer([0.01, 0.2, 0.5])
    assert out["combination_assumes"] == "independent p-values across cohorts"
    # z = sum(z_i)/sqrt(k) exactly -- the independent-p Stouffer, whatever the
    # permutation scheme that produced the inputs.
    from scipy.stats import norm

    z = [norm.isf(p) for p in (0.01, 0.2, 0.5)]
    assert out["stouffer_z"] == pytest.approx(sum(z) / np.sqrt(3))


@pytest.mark.skipif(not (RESULTS / "model_provenance.json").exists(),
                    reason="run python -m recompute.model_provenance")
def test_model_provenance_records_the_gradient_boosted_tree_claim():
    d = json.loads((RESULTS / "model_provenance.json").read_text(
        encoding="utf-8"))
    v = d["verdict"]
    # The claim is false as written and true when restricted to the clinical
    # cohorts. Both must stay recorded; if either flips, the manuscript's
    # sentence has to change with it.
    assert v["holds_for_all_cohorts"] is False
    assert set(v["non_tree_cohorts"]) == {"acs_income", "adult_income",
                                          "german_credit"}
    assert v["holds_for_clinical_cohorts"] is True
    assert v["synthea_uses_a_different_builder"] is True
    xgb = [m for m in d["models"] if m["is_gradient_boosted_tree"]]
    for m in xgb:
        assert m["hp_n_estimators"] == 200
        assert m["hp_max_depth"] == 4
        assert m["hp_learning_rate"] == pytest.approx(0.05)
        assert m["hp_subsample"] == pytest.approx(0.8)
        assert m["hp_colsample_bytree"] == pytest.approx(0.8)
    lr = [m for m in d["models"] if not m["is_gradient_boosted_tree"]]
    for m in lr:
        assert m["estimator_class"] == "LogisticRegression"
        assert m["hp_max_iter"] == 1000
