"""Regression tests for the model-based-concordance case-mix bootstrap.

The defect these guard against (``MBC_FIX.md``): the shipped
``_bootstrap_fraction`` resampled only the *raw* model-based concordance, while
``mbc_rows`` reported that single interval next to both the raw and the
*recalibrated* point estimate. 26 of 64 partitions ended up with a 95% interval
that did not contain its own point estimate -- the signature of a point estimate
and a resampling distribution targeting different estimands.

The invariant asserted here is the one that was violated: **every reported 95%
interval must contain its own point estimate.**
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from recompute.comparators.cohort_casemix import (
    MBC_CSV,
    _bootstrap_partition,
    _cox_fit,
    _levels,
    _partition_seed,
    _recalibrated,
    mbc_rows,
)
from recompute.comparators.core import auc_delong


# ── a self-contained cohort stand-in, so these tests need no dataset ─────────
class _FakeCohort:
    def __init__(self, seed=0, n=1200):
        rng = np.random.default_rng(seed)
        # Three levels with deliberately different case mix (different linear
        # predictor spread), which is exactly the situation MBC is meant to
        # detect, plus one level where the model genuinely under-performs.
        codes, lp = [], []
        for k, (scale, shift) in enumerate([(1.6, 0.0), (0.8, -0.5), (1.1, 0.4)]):
            x = rng.normal(shift, scale, n // 3)
            codes.append(np.full(n // 3, k))
            lp.append(x)
        codes = np.concatenate(codes)
        lp = np.concatenate(lp)
        p = 1.0 / (1.0 + np.exp(-lp))
        y = (rng.random(lp.size) < p).astype(int)
        # Degrade the scores in level 2 so a real (non-case-mix) gap exists.
        noisy = lp + rng.normal(0, 1.2, lp.size)
        lp_obs = np.where(codes == 2, noisy, lp)
        self.name = "fake"
        self.y = y
        self.s = 1.0 / (1.0 + np.exp(-lp_obs))
        self.codes_by_col = {"grp": codes}
        self.is_clinical = False
        self.n_test = int(lp.size)


@pytest.fixture(scope="module")
def fake_rows():
    return mbc_rows(_FakeCohort(), n_boot=600, seed=7)


# ── 1. the invariant that failed ─────────────────────────────────────────────
def _containment_failures(part_rows: pd.DataFrame):
    """Every (point estimate, interval) pair that must be self-consistent."""
    checks = [
        ("residual_gap", "residual_gap_lo95", "residual_gap_hi95", None),
        ("residual_gap_recalibrated", "residual_gap_recalibrated_lo95",
         "residual_gap_recalibrated_hi95", None),
        ("casemix_attributable_fraction",
         "casemix_attributable_fraction_lo95",
         "casemix_attributable_fraction_hi95", "fraction_reportable"),
        ("casemix_attributable_fraction_recalibrated",
         "casemix_attributable_fraction_recalibrated_lo95",
         "casemix_attributable_fraction_recalibrated_hi95",
         "fraction_reportable"),
    ]
    bad = []
    for _, r in part_rows.iterrows():
        for pt_c, lo_c, hi_c, gate in checks:
            if gate is not None and not bool(r[gate]):
                continue
            pt, lo, hi = r[pt_c], r[lo_c], r[hi_c]
            if not (np.isfinite(pt) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            if not (lo <= pt <= hi):
                bad.append(
                    f"{r.get('cohort','?')}/{r.get('rule','?')}/"
                    f"{r.get('partition','?')} {pt_c}={pt:.4f} "
                    f"not in [{lo:.4f}, {hi:.4f}]"
                )
    return bad


def test_every_interval_contains_its_point_estimate_synthetic(fake_rows):
    part = pd.DataFrame(fake_rows)
    part = part[part.row_type == "partition"]
    assert len(part) > 0
    bad = _containment_failures(part)
    assert not bad, "interval does not contain its point estimate:\n" + "\n".join(bad)


@pytest.mark.slow
def test_every_interval_contains_its_point_estimate_shipped_csv():
    """The regenerated results file must satisfy the same invariant."""
    if not MBC_CSV.exists():
        pytest.skip(f"{MBC_CSV} not generated")
    d = pd.read_csv(MBC_CSV)
    part = d[d.row_type == "partition"]
    assert len(part) > 0
    bad = _containment_failures(part)
    assert not bad, "interval does not contain its point estimate:\n" + "\n".join(bad)


# ── 2. the specific mechanism: recalibration must be refit per replicate ─────
def test_bootstrap_recalibrated_differs_from_raw():
    """``mbc_rc`` must not be a copy of ``mbc_raw``.

    If recalibration were skipped inside replicates (the shipped bug) the two
    replicate vectors would coincide and the recalibrated interval would silently
    be the raw one.
    """
    c = _FakeCohort()
    lv = _levels(c.y, c.s, c.codes_by_col["grp"], "m30")
    per = {k: auc_delong(c.y[m], c.s[m])[0] for k, m, *_ in lv}
    k_hi, k_lo = max(per, key=per.get), min(per, key=per.get)
    bt = _bootstrap_partition(c, lv, k_hi, k_lo, np.random.default_rng(1), 200)
    assert bt["n_ok"] == 200
    assert not np.allclose(bt["mbc_raw"], bt["mbc_rc"])


def test_bootstrap_holds_the_extreme_pair_fixed():
    """The re-selected gap is winner's-curse inflated relative to the fixed one.

    This is the second defect: re-picking argmax/argmin per replicate inflates
    the ratio's denominator, dragging the fraction toward zero.
    """
    c = _FakeCohort()
    lv = _levels(c.y, c.s, c.codes_by_col["grp"], "m30")
    per = {k: auc_delong(c.y[m], c.s[m])[0] for k, m, *_ in lv}
    k_hi, k_lo = max(per, key=per.get), min(per, key=per.get)
    bt = _bootstrap_partition(c, lv, k_hi, k_lo, np.random.default_rng(2), 400)
    assert bt["gap_sel"].mean() > bt["gap"].mean()
    # The fixed-pair gap is centred on the observed gap; the re-selected one is not.
    obs_gap = per[k_hi] - per[k_lo]
    assert abs(bt["gap"].mean() - obs_gap) < abs(bt["gap_sel"].mean() - obs_gap)


def test_replicates_are_not_filtered_on_positive_gap():
    """Dropping sign-reversed replicates would condition the interval."""
    c = _FakeCohort()
    lv = _levels(c.y, c.s, c.codes_by_col["grp"], "m30")
    per = {k: auc_delong(c.y[m], c.s[m])[0] for k, m, *_ in lv}
    # Force a near-zero true gap by comparing a level with itself-like twin:
    # instead assert the count is exact, i.e. nothing was silently discarded.
    k_hi, k_lo = max(per, key=per.get), min(per, key=per.get)
    bt = _bootstrap_partition(c, lv, k_hi, k_lo, np.random.default_rng(3), 300)
    assert bt["gap"].size == 300 == bt["n_ok"]


# ── 3. the tautology that makes the recalibrated fraction uninformative ──────
def test_recalibration_is_monotone_and_preserves_auroc():
    """Cox recalibration cannot change a level's AUROC, so MBC collapses onto it.

    This is why ``casemix_attributable_fraction_recalibrated`` sits near 1 by
    construction and must not be reported as the headline.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=900)
    p = 1.0 / (1.0 + np.exp(-(0.4 + 1.3 * x)))
    y = (rng.random(900) < p).astype(int)
    s = 1.0 / (1.0 + np.exp(-x))
    assert auc_delong(y, s)[0] == pytest.approx(
        auc_delong(y, _recalibrated(y, s))[0], abs=1e-12)


def test_cox_fit_recovers_known_coefficients():
    rng = np.random.default_rng(4)
    x = rng.normal(size=20000)
    a_true, b_true = 0.35, 1.4
    y = (rng.random(20000) < 1.0 / (1.0 + np.exp(-(a_true + b_true * x)))).astype(float)
    a, b = _cox_fit(y, x)
    assert a == pytest.approx(a_true, abs=0.06)
    assert b == pytest.approx(b_true, abs=0.08)


def test_cox_fit_survives_separation():
    """Perfectly separated data must fall back, not diverge or raise."""
    x = np.linspace(-3, 3, 60)
    y = (x > 0).astype(float)
    a, b = _cox_fit(y, x)
    assert np.all(np.isfinite([a, b]))


# ── 4. reproducibility ───────────────────────────────────────────────────────
def test_partition_seed_is_pinned_to_the_cell():
    """Seeds depend on identity only, not on iteration order or --only."""
    s1 = _partition_seed(42, "brfss2024", "m30", "race")
    s2 = _partition_seed(42, "brfss2024", "m30", "race")
    s3 = _partition_seed(42, "brfss2024", "ev10", "race")
    s4 = _partition_seed(42, "nhis2023", "m30", "race")
    assert s1 == s2
    assert len({s1, s3, s4}) == 3


def test_mbc_rows_is_deterministic():
    a = pd.DataFrame(mbc_rows(_FakeCohort(), n_boot=150, seed=11))
    b = pd.DataFrame(mbc_rows(_FakeCohort(), n_boot=150, seed=11))
    pd.testing.assert_frame_equal(a, b)


# ── 5. the ratio gate ────────────────────────────────────────────────────────
def test_fraction_is_suppressed_when_the_denominator_cannot_carry_it(fake_rows):
    part = pd.DataFrame(fake_rows)
    part = part[part.row_type == "partition"]
    for _, r in part.iterrows():
        if not r["fraction_reportable"]:
            assert not np.isfinite(r["casemix_attributable_fraction_lo95"])
        else:
            # gate means the bootstrap gap is separated from zero
            assert r["boot_gap_lo95"] > 0


@pytest.mark.slow
def test_shipped_csv_reports_monte_carlo_error():
    if not MBC_CSV.exists():
        pytest.skip(f"{MBC_CSV} not generated")
    d = pd.read_csv(MBC_CSV)
    part = d[d.row_type == "partition"]
    assert (part["n_boot"] >= 2000).all(), "bootstrap replicates were reduced"
    rep = part[part.fraction_reportable.astype(bool)]
    if len(rep):
        assert rep["casemix_attributable_fraction_mc_se_lo95"].notna().any()
