"""
Guards for the joint-permutation null and the swept subgroup-inclusion rule.

The load-bearing one is
``test_independent_scheme_reproduces_published``: the new code path must not
have disturbed the old one, because the manuscript's published p-values were
produced by it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
for p in (str(REPO), str(REPO / "recompute" / "_vendor")):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompute.null_reference import (  # noqa: E402
    INCLUSION_RULES,
    _rule_admits,
    code_columns,
    cohort_null_reference,
    draw_permuted_codes,
    mc_pvalue,
    partition_gaps_by_rule,
)

PUBLISHED = REPO / "recompute" / "results" / "german_credit.json"


def _toy(n=1200, seed=7):
    """A cohort whose demographic columns are deliberately collinear."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.25).astype(int)
    s = rng.normal(loc=0.8 * y, scale=1.0, size=n)
    age = rng.integers(0, 4, size=n)
    # income is a noisy copy of age -> strong association
    income = np.where(rng.random(n) < 0.85, age, rng.integers(0, 4, size=n))
    sex = rng.integers(0, 2, size=n)
    demo = pd.DataFrame({"age": age, "income": income, "sex": sex})
    return y, s, demo


def test_independent_scheme_reproduces_published():
    """The old default must still give the published numbers bit-for-bit."""
    if not PUBLISHED.exists():
        pytest.skip("published results not present")
    pub = json.loads(PUBLISHED.read_text(encoding="utf-8"))["null_reference"]

    from recompute.cohorts import LOADERS

    b = LOADERS["german_credit"]()
    scores = b["model"].predict_proba(np.asarray(b["X_test"], float))[:, 1]
    got = cohort_null_reference(
        b["y_test"], scores, b["demo_test"],
        subgroup_columns=b["subgroup_columns"],
        min_subgroup_n=30, n_reps=2000, random_state=42,
        observed_gap=pub["observed_gap"],
    )
    for k in ("null_mean_gap", "null_sd_gap", "null_median_gap",
              "null_p95_gap", "null_p99_gap", "p_value_vs_null"):
        assert got[k] == pytest.approx(pub[k], rel=0, abs=0), k


def test_joint_permutation_preserves_the_contingency_table():
    """A joint draw is a relabelling of rows, so the cross-tab is invariant."""
    y, _s, demo = _toy()
    codes = code_columns(demo)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng = np.random.default_rng(0)

    def xtab(c):
        return pd.crosstab(c["age"], c["income"]).values

    joint = draw_permuted_codes(codes, pos, neg, rng, scheme="joint")
    assert np.array_equal(np.sort(xtab(joint), axis=None),
                          np.sort(xtab(codes), axis=None))
    assert np.array_equal(xtab(joint), xtab(codes))

    indep = draw_permuted_codes(codes, pos, neg, rng, scheme="independent")
    # The independent scheme breaks it (overwhelmingly likely at this strength).
    assert not np.array_equal(xtab(indep), xtab(codes))


def test_both_schemes_preserve_size_and_prevalence():
    y, _s, demo = _toy()
    codes = code_columns(demo)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    for scheme in ("independent", "joint"):
        rng = np.random.default_rng(1)
        perm = draw_permuted_codes(codes, pos, neg, rng, scheme=scheme)
        for col in codes:
            a, b = codes[col], perm[col]
            assert np.array_equal(np.bincount(a), np.bincount(b)), scheme
            assert np.array_equal(np.bincount(a[y == 1]),
                                  np.bincount(b[y == 1])), scheme


def test_joint_null_is_not_wider_than_independent_when_columns_are_collinear():
    """The whole argument for the fix, on a cohort built to exhibit it."""
    y, s, demo = _toy()
    kw = dict(min_subgroup_n=30, n_reps=1500, random_state=3)
    ind = cohort_null_reference(y, s, demo, scheme="independent", **kw)
    joi = cohort_null_reference(y, s, demo, scheme="joint", **kw)
    assert joi["null_p95_gap"] < ind["null_p95_gap"]
    assert joi["null_mean_gap"] < ind["null_mean_gap"]


def test_marginal_per_column_null_is_scheme_invariant():
    """Only the cross-column dependence changes, never a column's margin."""
    y, s, demo = _toy()
    one = demo[["age"]]
    kw = dict(min_subgroup_n=30, n_reps=800, random_state=11)
    a = cohort_null_reference(y, s, one, scheme="independent", **kw)
    b = cohort_null_reference(y, s, one, scheme="joint", **kw)
    assert a["null_mean_gap"] == pytest.approx(b["null_mean_gap"], abs=0.01)


def test_rule_sweep_is_monotone_in_admissions():
    y, s, demo = _toy()
    codes = code_columns(demo)
    for col, c in codes.items():
        for lvl in np.unique(c):
            m = c == lvl
            n, n_pos = int(m.sum()), int(y[m].sum())
            adm = [k for k in ("m20", "m30", "m50", "m100")
                   if _rule_admits(INCLUSION_RULES[k], n, n_pos, n - n_pos)]
            # A stricter size threshold can never admit more.
            for tight, loose in (("m30", "m20"), ("m50", "m30"),
                                 ("m100", "m50")):
                if tight in adm:
                    assert loose in adm, (col, lvl, tight, loose)


def test_all_rules_share_one_pass_and_agree_with_single_rule():
    y, s, demo = _toy()
    codes = code_columns(demo)
    multi = partition_gaps_by_rule(y, s, codes)
    from recompute.null_reference import _max_partition_gap
    for k in ("m20", "m30", "m50", "m100"):
        single = _max_partition_gap(
            y, s, codes, int(INCLUSION_RULES[k]["min_n"]))
        assert multi[k] == pytest.approx(single, abs=1e-12), k


def test_pvalue_floor_is_reported_as_an_inequality():
    v = np.linspace(0.0, 0.1, 2000)
    out = mc_pvalue(v, 0.5)          # observed beyond every null draw
    assert out["p_is_floor"] is True
    assert out["p_value_vs_null"] == pytest.approx(1 / 2001)
    assert out["p_report"].startswith("<=")
    out2 = mc_pvalue(v, 0.05)
    assert out2["p_is_floor"] is False
    assert not out2["p_report"].startswith("<=")


def test_mde_equals_the_null_95th_percentile():
    v = np.random.default_rng(0).gamma(2.0, 0.05, size=5000)
    out = mc_pvalue(v, 0.2)
    assert out["minimum_detectable_gap_p05"] == pytest.approx(
        float(np.percentile(v, 95)))
