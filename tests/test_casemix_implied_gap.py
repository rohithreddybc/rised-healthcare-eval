"""Tests for the implied case-mix gap, its refit grid and its sampling null.

``recompute/casemix_implied_gap.py`` had no test file. These cover the three
things the manuscript now leans on: that the identity is distribution free, that
the Gaussian plug-in is the only place a shape assumption enters, and that the
refit path reproduces the published table exactly through the published fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recompute.casemix_implied_gap import compute
from recompute.casemix_implied_gap_robustness import (
    RESULTS,
    headline_stability,
    implied_gap_rows_for_fit,
    summarise,
)
from recompute.casemix_theory import (
    auroc_definition_from_risk,
    auroc_from_risk,
    auroc_gaussian_lp,
)
from recompute.refit import build_full_cohort, published_fit

PUBLISHED_CSV = RESULTS / "casemix_implied_gap.csv"
ROBUSTNESS_CSV = RESULTS / "casemix_implied_gap_robustness.csv"
NULL_CSV = RESULTS / "casemix_implied_gap_null.csv"

#: Small enough to refit inside a test suite.
FAST_COHORT = "german_credit"


# ── the identity is distribution free ────────────────────────────────────────
@pytest.mark.parametrize("shape", ["gaussian", "laplace", "lognormal",
                                   "uniform", "bimodal", "t3"])
def test_gini_form_equals_the_definition_for_every_shape(shape):
    """Proposition 1 is an identity, not a Gaussian approximation."""
    rng = np.random.default_rng(20240810)
    n = 20000
    if shape == "gaussian":
        lp = rng.normal(0.0, 1.0, n)
    elif shape == "laplace":
        lp = rng.laplace(0.0, 1.0 / np.sqrt(2.0), n)
    elif shape == "lognormal":
        lp = rng.lognormal(0.0, 0.6, n) - 1.2
    elif shape == "uniform":
        lp = rng.uniform(-1.7, 1.7, n)
    elif shape == "bimodal":
        lp = np.where(rng.random(n) < 0.5, -1.2, 1.2) + rng.normal(0, 0.3, n)
    else:
        lp = rng.standard_t(3, n) / np.sqrt(3.0)
    risk = 1.0 / (1.0 + np.exp(-lp))
    assert auroc_from_risk(risk) == pytest.approx(
        auroc_definition_from_risk(risk), abs=1e-12)


def test_gaussian_plugin_is_the_only_shape_assumption():
    """Two distributions at one SD give different AUROC, so (mean, SD) is not
    a sufficient summary and the Gaussian plug-in carries a real error."""
    sd = 0.6
    gauss = auroc_gaussian_lp(0.0, sd)
    two_atom = auroc_from_risk(1.0 / (1.0 + np.exp(-np.array([-sd, sd] * 5000))))
    assert gauss > two_atom
    assert gauss - two_atom > 0.005


# ── the refit path reproduces the published table ────────────────────────────
def test_published_spec_reproduces_the_published_implied_gap_table():
    """The published fit, through the refit code path, must match the CSV."""
    if not (PUBLISHED_CSV.exists() and ROBUSTNESS_CSV.exists()):
        pytest.skip("result files not built")
    old = pd.read_csv(PUBLISHED_CSV)
    new = pd.read_csv(ROBUSTNESS_CSV)
    new = new[(new["spec_id"] == "published") & (new["rule"] == "m30")
              & (new["is_clinical"])]
    m = old.merge(new, on=["cohort", "partition"], suffixes=("_o", "_n"))
    assert len(m) == len(old)
    assert np.abs(m["implied_casemix_gap"]
                  - m["implied_casemix_gap_gaussian"]).max() < 1e-5
    assert np.abs(m["partition_sd_ratio_o"]
                  - m["partition_sd_ratio_n"]).max() < 1e-5


def test_row_builder_runs_on_a_real_fit():
    fc = build_full_cohort(FAST_COHORT)
    rows = implied_gap_rows_for_fit(published_fit(fc), fc.subgroup_columns)
    assert rows
    for r in rows:
        assert r["implied_casemix_gap_gaussian"] >= 0.0
        assert r["implied_casemix_gap_empirical"] >= 0.0
        assert r["max_level_lp_sd"] >= r["min_level_lp_sd"] > 0.0


# ── the grid is not silently incomplete ──────────────────────────────────────
def test_headline_stability_reports_its_own_denominator():
    """A count of partitions reaching a threshold is meaningless without the
    number of partitions the specification actually measured."""
    if not ROBUSTNESS_CSV.exists():
        pytest.skip("result files not built")
    df = pd.read_csv(ROBUSTNESS_CSV)
    h = headline_stability(df)
    assert "n_partitions" in h.columns
    assert (h["n_partitions"] > 0).all()
    assert h["n_at_0.05"].le(h["n_partitions"]).all()


def test_summary_has_one_row_per_partition():
    if not ROBUSTNESS_CSV.exists():
        pytest.skip("result files not built")
    s = summarise(pd.read_csv(ROBUSTNESS_CSV))
    assert s["partition_key"].is_unique
    assert (s["min_gaussian"] <= s["median_gaussian"]).all()
    assert (s["median_gaussian"] <= s["max_gaussian"]).all()


# ── the sampling null ────────────────────────────────────────────────────────
def test_null_is_strictly_above_the_no_effect_value():
    """rho-hat is a max over a min and the implied gap a max minus a min, so
    both exceed their no-effect value under a null with no case mix at all."""
    if not NULL_CSV.exists():
        pytest.skip("result files not built")
    n = pd.read_csv(NULL_CSV)
    assert (n["null_sd_ratio_median"] > 1.0).all()
    assert (n["null_implied_gap_median"] > 0.0).all()


def test_published_compute_still_runs():
    if not PUBLISHED_CSV.exists():
        pytest.skip("result files not built")
    recs = compute()
    assert len(recs) == 21
    assert all(r["implied_casemix_gap"] >= 0 for r in recs)
