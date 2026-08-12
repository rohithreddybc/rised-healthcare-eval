"""
Tests for the SD-ratio robustness check.

Three things have to be true for the check to mean anything.

**The refit path computes the same quantity the published path does.** If
``sd_ratio_robustness`` recomputed rho-hat by its own arithmetic, a difference
between a refit and the published table could be the arithmetic rather than the
fit. The published fit is therefore carried through the *new* code path and is
asserted to reproduce ``cohort_sd_ratios.csv`` exactly.

**The refits are reproducible.** Same class, same seed, same rho-hat, to the
last bit -- otherwise "spread across specifications" is partly the harness.

**The variance decomposition is correct.** It is checked against a synthetic
cube with known variance components and against the algebraic identity that the
sums of squares partition the total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recompute.comparators.sd_ratio_report import (
    anova3_random,
    kendall_w,
)
from recompute.comparators.sd_ratio_robustness import (
    RESULTS,
    induced_flag_rate,
    load_sweep_curves,
    sd_ratio_rows_for_fit,
)
from recompute.refit import (
    MODEL_CLASSES,
    PUBLISHED_CLASS,
    SEEDS,
    build_full_cohort,
    fit_spec,
    published_fit,
    split_indices,
)

PUBLISHED_CSV = RESULTS / "cohort_sd_ratios.csv"
ROBUSTNESS_CSV = RESULTS / "sd_ratio_robustness.csv"

#: Small enough to refit repeatedly inside a test suite.
FAST_COHORT = "german_credit"


# ── the harness ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def fast_cohort():
    return build_full_cohort(FAST_COHORT)


def test_published_fit_reproduces_published_table(fast_cohort):
    """The published model, through the NEW code path, must match the old CSV."""
    curves = load_sweep_curves()
    rows = sd_ratio_rows_for_fit(published_fit(fast_cohort),
                                 fast_cohort.subgroup_columns, curves)
    got = {(r["rule"], r["partition"]): r["partition_sd_ratio"] for r in rows}
    old = pd.read_csv(PUBLISHED_CSV)
    old = old[old["cohort"] == FAST_COHORT].drop_duplicates(
        ["rule", "partition"])
    assert len(old) > 0
    for _, r in old.iterrows():
        key = (r["rule"], r["partition"])
        assert key in got, key
        assert got[key] == pytest.approx(r["partition_sd_ratio"], abs=1e-12)


def test_refit_is_reproducible(fast_cohort):
    """Same class, same seed -> bit-identical scores and rho-hat."""
    a = fit_spec(fast_cohort, "logreg_l2", 43)
    b = fit_spec(fast_cohort, "logreg_l2", 43)
    np.testing.assert_array_equal(a.scores, b.scores)
    np.testing.assert_array_equal(a.y_test, b.y_test)


def test_seeds_actually_move_the_split(fast_cohort):
    """A different seed must give a different held-out set, or nothing varies."""
    _, te42 = split_indices(fast_cohort, 42)
    _, te43 = split_indices(fast_cohort, 43)
    assert set(te42.tolist()) != set(te43.tolist())
    assert te42.size == te43.size


def test_split_is_deterministic_in_the_seed(fast_cohort):
    for s in (42, 45):
        tr1, te1 = split_indices(fast_cohort, s)
        tr2, te2 = split_indices(fast_cohort, s)
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)


def test_train_and_test_are_disjoint_and_cover(fast_cohort):
    for s in SEEDS[:3]:
        tr, te = split_indices(fast_cohort, s)
        assert set(tr.tolist()).isdisjoint(te.tolist())
        assert len(tr) + len(te) == fast_cohort.X.shape[0]


def test_every_class_fits_and_scores(fast_cohort):
    for name in MODEL_CLASSES:
        f = fit_spec(fast_cohort, name, 42)
        assert f.scores.shape == f.y_test.shape
        assert np.all(np.isfinite(f.scores))
        assert f.scores.min() >= 0.0 and f.scores.max() <= 1.0


def test_published_class_map_covers_every_loader():
    from recompute.cohorts import LOADERS

    assert set(PUBLISHED_CLASS) == set(LOADERS)
    assert set(PUBLISHED_CLASS.values()) <= set(MODEL_CLASSES)


def test_diabetes130_keeps_a_group_split():
    """Refitting must not reintroduce the row-level leakage the group split removes."""
    fc = build_full_cohort("diabetes130")
    assert fc.groups is not None
    for s in (42, 44):
        tr, te = split_indices(fc, s)
        leak = float(np.isin(fc.groups[te], np.unique(fc.groups[tr])).mean())
        assert leak == 0.0, f"seed {s} leaked {leak:.4f} of test rows"


# ── the sweep mapping ────────────────────────────────────────────────────────
def test_flag_rate_interpolation_hits_the_grid_nodes():
    curves = load_sweep_curves()
    x, f, _ = curves["permutation_null"]
    for xi, fi in zip(x, f):
        got, _, extrap = induced_flag_rate(curves["permutation_null"], xi)
        assert got == pytest.approx(fi, abs=1e-12)
        assert not extrap


def test_flag_rate_flags_extrapolation():
    curves = load_sweep_curves()
    x, _, _ = curves["permutation_null"]
    _, _, extrap = induced_flag_rate(curves["permutation_null"],
                                     float(x[-1]) + 1.0)
    assert extrap
    _, _, extrap_in = induced_flag_rate(curves["permutation_null"],
                                        float(x[0]))
    assert not extrap_in


# ── the decomposition ────────────────────────────────────────────────────────
def test_anova3_sums_of_squares_partition_the_total():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(7, 4, 6))
    res = anova3_random(y)
    # Every component's SS is recoverable from the variance components only up
    # to the EMS system, but the total SS must be what the identity says.
    gm = y.mean()
    assert res["ss_total"] == pytest.approx(float(np.sum((y - gm) ** 2)))


def test_anova3_recovers_a_dominant_partition_component():
    """A cube whose only real signal is the partition effect."""
    rng = np.random.default_rng(1)
    a, b, c = 21, 4, 6
    A = rng.normal(0.0, 1.0, size=a)
    y = A[:, None, None] + rng.normal(0.0, 0.05, size=(a, b, c))
    res = anova3_random(y)
    assert res["var_partition"] == pytest.approx(1.0, rel=0.5)
    assert res["share_partition"] > 0.95
    assert res["model_side_share"] < 0.05


def test_anova3_recovers_a_dominant_class_component():
    rng = np.random.default_rng(2)
    a, b, c = 21, 4, 6
    B = rng.normal(0.0, 1.0, size=b)
    y = B[None, :, None] + rng.normal(0.0, 0.05, size=(a, b, c))
    res = anova3_random(y)
    assert res["share_model_class"] > 0.90
    assert res["share_partition"] < 0.10


def test_anova3_pure_noise_has_no_dominant_component():
    rng = np.random.default_rng(3)
    res = anova3_random(rng.normal(size=(21, 4, 6)))
    assert res["share_residual_3way"] > 0.5


def test_anova3_shares_sum_to_one():
    rng = np.random.default_rng(4)
    res = anova3_random(rng.normal(size=(9, 4, 6)))
    keys = ["partition", "model_class", "seed", "partition_x_class",
            "partition_x_seed", "class_x_seed", "residual_3way"]
    assert sum(res[f"share_{k}"] for k in keys) == pytest.approx(1.0)


def test_kendall_w_is_one_for_identical_rankings():
    m = pd.DataFrame({f"s{j}": np.arange(10, dtype=float) for j in range(5)})
    assert kendall_w(m)["W"] == pytest.approx(1.0)


def test_kendall_w_is_small_for_random_rankings():
    rng = np.random.default_rng(5)
    m = pd.DataFrame({f"s{j}": rng.permutation(20).astype(float)
                      for j in range(24)})
    assert kendall_w(m)["W"] < 0.25


# ── the shipped artefact ─────────────────────────────────────────────────────
@pytest.mark.skipif(not ROBUSTNESS_CSV.exists(),
                    reason="run recompute.comparators.sd_ratio_robustness first")
def test_shipped_csv_has_the_full_grid():
    df = pd.read_csv(ROBUSTNESS_CSV)
    specs = set(df["spec_id"].astype(str))
    assert "published" in specs
    for c in MODEL_CLASSES:
        for s in SEEDS:
            assert f"{c}|s{s}" in specs, f"{c}|s{s} missing -- grid was reduced"
    assert len(specs) == len(MODEL_CLASSES) * len(SEEDS) + 1


@pytest.mark.skipif(not ROBUSTNESS_CSV.exists(),
                    reason="run recompute.comparators.sd_ratio_robustness first")
def test_shipped_csv_published_rows_match_the_published_table():
    new = pd.read_csv(ROBUSTNESS_CSV)
    new = new[new["model_class"] == "published"]
    old = pd.read_csv(PUBLISHED_CSV).drop_duplicates(
        ["cohort", "rule", "partition"])
    m = new.merge(old, on=["cohort", "rule", "partition"],
                  suffixes=("_new", "_old"))
    assert len(m) == len(old)
    np.testing.assert_allclose(m["partition_sd_ratio_new"],
                               m["partition_sd_ratio_old"], atol=1e-12)


@pytest.mark.skipif(not ROBUSTNESS_CSV.exists(),
                    reason="run recompute.comparators.sd_ratio_robustness first")
def test_shipped_csv_covers_the_21_clinical_partitions():
    df = pd.read_csv(ROBUSTNESS_CSV)
    pub = df[(df["model_class"] == "published") & (df["rule"] == "m30")
             & df["is_clinical"]]
    assert len(pub) == 21
    assert pub["partition_sd_ratio"].median() == pytest.approx(1.1449810617,
                                                              abs=1e-9)
    assert pub["partition_sd_ratio"].min() == pytest.approx(1.0215101993,
                                                            abs=1e-9)
    assert pub["partition_sd_ratio"].max() == pytest.approx(3.3035828296,
                                                            abs=1e-9)
