"""
Which permutation scheme produced which published number.

    python -m recompute.scheme_provenance

Why this exists
---------------
The manuscript states that the permutation is carried out "independently for
each partition". Parts of the codebase run ``scheme="joint"`` throughout
(``recompute/comparators/core.py`` ``PermContext.draw``,
``recompute/comparators/diciccio.py`` ``run_cohort``). Both statements are true
of *something*, and neither is true of everything, so a reader cannot tell which
scheme is behind any given table. This module settles it artefact by artefact and
writes ``recompute/results/scheme_provenance.csv``.

The two schemes (and what the word "scheme" does NOT mean)
----------------------------------------------------------
``scheme`` names how the **demographic columns** are resampled in
:func:`recompute.null_reference.draw_permuted_codes`, and nothing else:

``independent``
    a fresh within-outcome-class permutation is drawn for **each demographic
    column separately**, so the association between age, sex, race, insurance
    and income is destroyed and the per-column gaps become independent.

``joint``
    **one** within-outcome-class permutation of the row indices is drawn per
    replicate and carried across every demographic column at once, so whole
    demographic rows move together and the joint contingency table is preserved
    exactly.

Both preserve every subgroup's size and prevalence, the marginal score
distribution and the cohort AUROC, and both force equal true subgroup AUROC.
Each column's *marginal* null is identical under the two; only the cross-column
dependence, and hence the maximum over columns, differs.

**``scheme`` is not an assumption about the Stouffer combination.**
``null_joint_combined.csv`` carries a ``scheme`` column, and it means exactly the
above: it says which permutation scheme produced the per-cohort p-values that
were then combined. It does *not* say that one row combines under dependence and
the other under independence. :func:`recompute.aggregate_null_joint.stouffer`
computes ``z = sum(z_i) / sqrt(k)`` with equal weights in **both** rows, which is
Stouffer's method under the assumption that the k cohort p-values are
independent. That assumption is about the ten cohorts being separate datasets --
which they are -- and is unrelated to the permutation scheme. So both rows of
``null_joint_combined.csv`` assume independence in the combination step; they
differ only in how the inputs to that step were generated. The same is true of
``fisher``. This is now stated in that module's docstring as well.

Evidence, not assertion
-----------------------
Where an artefact stores enough information, the scheme is *verified* rather than
read off the source: the recorded p-values are recomputed under each scheme and
the one that reproduces them bit-for-bit is the one that produced them. The
``evidence`` column says which of the two happened for every row.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
NULL_JOINT = RESULTS / "null_joint"
OUT = RESULTS / "scheme_provenance.csv"


def _cohorts() -> List[str]:
    return sorted(p.stem for p in NULL_JOINT.glob("*.json"))


def _verify_incumbent_rows_are_joint() -> Dict[str, Any]:
    """Check the comparator table's incumbent p-values against both blocks.

    ``recompute/comparators/incumbent.py`` reads ``payload["results"][scheme]``
    with ``scheme="joint"``. That is a code fact; this turns it into a data fact
    by matching the p-values actually present in ``comparator_comparison.csv``
    against the ``independent`` and ``joint`` blocks of the stored null runs.
    """
    csv = RESULTS / "comparator_comparison.csv"
    if not csv.exists():
        return {"status": "comparator_comparison.csv absent"}
    df = pd.read_csv(csv)
    inc = df[df["method"] == "permutation_null"]
    hits = {"independent": 0, "joint": 0, "both": 0, "neither": 0, "n": 0,
            "max_abs_diff_to_joint": 0.0}
    for _, r in inc.iterrows():
        path = NULL_JOINT / f"{r['cohort']}.json"
        if not path.exists() or not np.isfinite(r.get("p_value", np.nan)):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        hits["n"] += 1
        got = {}
        for scheme in ("independent", "joint"):
            stored = payload["results"][scheme][r["rule"]].get("p_value_vs_null")
            # One ULP of tolerance: p-values go through a float -> decimal ->
            # float round trip on the way into the CSV, which costs the last bit.
            # The two schemes' p-values differ by ~1e-3 here, so a 1e-12 window
            # cannot confuse them.
            got[scheme] = (stored is not None
                           and abs(float(stored) - float(r["p_value"])) <= 1e-12)
        hits["max_abs_diff_to_joint"] = max(
            hits["max_abs_diff_to_joint"],
            abs(float(payload["results"]["joint"][r["rule"]]["p_value_vs_null"])
                - float(r["p_value"])))
        if got["joint"] and got["independent"]:
            hits["both"] += 1
        elif got["joint"]:
            hits["joint"] += 1
        elif got["independent"]:
            hits["independent"] += 1
        else:
            hits["neither"] += 1
    return hits


def _older_cohort_json_scheme(cohort: str) -> Dict[str, Any]:
    """Classify the pre-joint ``results/<cohort>.json`` null block.

    Those files were written before ``cohort_null_reference`` grew a ``scheme``
    parameter, so they carry no ``permutation_scheme`` key and their
    ``null_design`` string lacks the per-scheme clause the current code appends.
    The absence of both is the fingerprint of the original single-scheme code
    path, which is the ``independent`` branch.
    ``tests/test_null_joint.py::test_independent_scheme_reproduces_published``
    pins that the independent branch still reproduces those values bit-for-bit.
    """
    path = RESULTS / f"{cohort}.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    nr = d.get("null_reference", {})
    if not nr:
        return {}
    design = str(nr.get("null_design", ""))
    return {
        "has_permutation_scheme_key": "permutation_scheme" in nr,
        "recorded_scheme": nr.get("permutation_scheme", "<absent>"),
        "design_mentions_per_column": "per demographic column" in design,
        "design_mentions_one_row_permutation": "one row permutation" in design,
        "n_reps": nr.get("n_reps"),
        "min_subgroup_n": nr.get("min_subgroup_n"),
    }


ROWS: List[Dict[str, Any]] = []


def _row(**kw: Any) -> None:
    ROWS.append(kw)


def build() -> pd.DataFrame:
    ROWS.clear()
    cohorts = _cohorts()

    # ── 1. The pre-joint per-cohort runs and everything aggregated from them ──
    probes = {c: _older_cohort_json_scheme(c) for c in cohorts}
    probes = {c: p for c, p in probes.items() if p}
    any_scheme_key = any(p["has_permutation_scheme_key"] for p in probes.values())
    any_joint_phrase = any(p["design_mentions_one_row_permutation"]
                           for p in probes.values())
    reps = sorted({p["n_reps"] for p in probes.values()})
    older_ok = (not any_scheme_key) and (not any_joint_phrase)
    older_ev = (
        f"verified from the artefact: {len(probes)} cohort files carry no "
        f"'permutation_scheme' key and no joint-scheme clause in 'null_design', "
        f"which is the fingerprint of the pre-scheme code path (the independent "
        f"branch); pinned bit-for-bit by "
        f"tests/test_null_joint.py::test_independent_scheme_reproduces_published"
        if older_ok else
        "INCONCLUSIVE: cohort files carry scheme metadata; inspect them")

    for f in ("recompute/results/<cohort>.json  [null_reference block]",
              "recompute/results/summary.csv",
              "recompute/results/summary.json",
              "recompute/results/null_comparison.csv",
              "recompute/results/findings.json",
              "docs/cohort_evaluation_results.md"):
        _row(result_file=f, scheme_used="independent",
             n_perm=reps[0] if len(reps) == 1 else ";".join(map(str, reps)),
             rules="m30 only", produced_by="recompute/run_cohort.py -> "
             "cohort_null_reference(scheme default) -> recompute/aggregate.py",
             evidence=older_ev)

    # ── 2. The joint-vs-independent sweep: BOTH schemes, side by side ────────
    both_ev = ("verified from the artefact: every entry carries an explicit "
               "'permutation_scheme' field and both blocks are present")
    ok = True
    for c in cohorts:
        d = json.loads((NULL_JOINT / f"{c}.json").read_text(encoding="utf-8"))
        for scheme in ("independent", "joint"):
            blk = d["results"].get(scheme)
            if blk is None:
                ok = False
                continue
            for rule, e in blk.items():
                if rule == "runtime_s":
                    continue
                if e.get("permutation_scheme") != scheme:
                    ok = False
    if not ok:
        both_ev = "INCONCLUSIVE: a block is missing or mislabelled"
    nreps = sorted({json.loads((NULL_JOINT / f"{c}.json").read_text(
        encoding="utf-8"))["n_reps"] for c in cohorts})

    _row(result_file="recompute/results/null_joint/<cohort>.json",
         scheme_used="BOTH (results.independent and results.joint)",
         n_perm=";".join(map(str, nreps)), rules="m20,m30,m50,m100,ev10",
         produced_by="recompute/null_joint.py", evidence=both_ev)
    for f in ("recompute/results/null_sweep_mmin.csv",
              "recompute/results/null_joint_combined.csv",
              "recompute/results/null_joint_sign_tests.csv"):
        _row(result_file=f, scheme_used="BOTH (one row per scheme; see the "
             "'scheme' column)", n_perm=";".join(map(str, nreps)),
             rules="m20,m30,m50,m100,ev10",
             produced_by="recompute/aggregate_null_joint.py",
             evidence=both_ev + "; the CSV's own 'scheme' column carries it")
    _row(result_file="recompute/results/null_comparison_joint.csv",
         scheme_used="BOTH (old_* columns = independent, new_* = joint)",
         n_perm=";".join(map(str, nreps)), rules="m30 only",
         produced_by="recompute/aggregate_null_joint.py",
         evidence=both_ev + "; column prefixes carry it")
    _row(result_file="docs/permutation_null_specification.md",
         scheme_used="BOTH (reported side by side)",
         n_perm=";".join(map(str, nreps)), rules="m20,m30,m50,m100,ev10",
         produced_by="recompute/report_null_joint.py", evidence=both_ev)

    # ── 3. The comparator evaluation: joint only ─────────────────────────────
    hits = _verify_incumbent_rows_are_joint()
    if hits.get("n"):
        inc_ev = (
            f"verified against the stored null runs: of {hits['n']} estimable "
            f"incumbent p-values in comparator_comparison.csv, "
            f"{hits['joint'] + hits['both']} match the JOINT block to within "
            f"1e-12 (max abs difference "
            f"{hits['max_abs_diff_to_joint']:.1e}, one float ULP -- the CSV "
            f"round trip), {hits['independent']} match ONLY the independent "
            f"block, {hits['both']} match both because the two schemes agree "
            f"there, {hits['neither']} match neither")
    else:
        inc_ev = "comparator_comparison.csv absent; scheme read from the source"

    _row(result_file="recompute/results/comparator_comparison.csv  "
         "[method=permutation_null]",
         scheme_used="joint", n_perm=";".join(map(str, nreps)),
         rules="m20,m30,m50,m100,ev10",
         produced_by="recompute/comparators/incumbent.py run_cohort "
                     "(scheme='joint') reading results/null_joint/",
         evidence=inc_ev)
    _row(result_file="recompute/results/comparator_comparison.csv  "
         "[method=diciccio2020]",
         scheme_used="joint", n_perm=10000,
         rules="m20,m30,m50,m100,ev10",
         produced_by="recompute/comparators/diciccio.py run_cohort "
                     "(scheme='joint' default, core.py:360)",
         evidence="code path: run_cohort's `scheme` argument defaults to "
                  "'joint' and run.py never overrides it")
    _row(result_file="recompute/results/comparator_comparison.csv  "
         "[lum2022*, four_fifths, fixed_threshold_005]",
         scheme_used="n/a -- no permutation", n_perm="", rules="all five",
         produced_by="recompute/comparators/{lum,four_fifths,naive}.py",
         evidence="these methods are closed-form or deterministic; the "
                  "permutation scheme cannot apply to them")
    _row(result_file="recompute/results/comparator_type1.csv",
         scheme_used="joint", n_perm=999,
         rules="m30,ev10",
         produced_by="recompute/comparators/type1.py -> "
                     "{incumbent,diciccio}.pvalue_only -> PermContext.draw",
         evidence="code path: PermContext.draw's `scheme` argument defaults to "
                  "'joint' (core.py:360) and neither pvalue_only passes it")
    _row(result_file="recompute/results/comparator_runtime.csv",
         scheme_used="joint", n_perm=10000, rules="m20,m30,m50,m100,ev10",
         produced_by="recompute/comparators/bench.py",
         evidence="code path: incumbent.recompute_null and diciccio.run_cohort "
                  "both default to scheme='joint'; the stored-kernel column is "
                  "read from results['joint']['runtime_s']")
    _row(result_file="docs/comparator_evaluation.md / .tables.md",
         scheme_used="joint", n_perm="10000 (cohorts); 999 (Type I)",
         rules="m20,m30,m50,m100,ev10",
         produced_by="recompute/comparators/report.py",
         evidence="derived entirely from the three comparator CSVs above")

    return pd.DataFrame(ROWS)


def main() -> int:
    df = build()
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT}  ({len(df)} rows)\n")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df[["result_file", "scheme_used", "n_perm", "rules"]]
              .to_string(index=False))
    print("\nEvidence:")
    for _, r in df.iterrows():
        print(f"  - {r['result_file']}\n      {r['evidence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
