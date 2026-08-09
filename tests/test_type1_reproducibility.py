"""Regression tests for the Type I simulation's reproducibility.

The defect
----------
``recompute/comparators/simulate.py`` seeded each replicate with

    np.random.default_rng([seed, rep, abs(hash(geom.name)) % 2**31])

CPython salts ``str.__hash__`` with a per-process random value unless
``PYTHONHASHSEED`` is set before interpreter start. The Type I study runs its
cells across a ``ProcessPoolExecutor``, so every worker had a different salt and
therefore drew a *different* dataset for the same ``(geometry, replicate,
seed)``. Two consequences, both fatal for a published table:

  * two cells of the same geometry (m30 and ev10) were not run on the same data,
    so they were not paired even though the study is designed to pair them;
  * no run could be reproduced, by anyone, ever -- including by this repository.

The module docstring's claim that ``make_dataset`` is "a pure function of
``(geometry, replicate index, seed)``" was false.

The tests below are the ones that would have caught it. ``test_draws_are
_identical_across_processes`` is the direct regression test: it re-derives the
datasets in genuinely separate interpreter processes started with *different*
``PYTHONHASHSEED`` values and requires bit-identical output. Run against the old
code it fails; against the fix it passes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from recompute.comparators.simulate import (
    GEOMETRIES,
    GEOMETRY_BY_NAME,
    apply_monotone,
    geometry_seed_word,
    make_dataset,
    true_subgroup_auc,
    verify_null,
)

REPO = Path(__file__).resolve().parent.parent

#: A geometry from each family, cheap enough to run in a unit test.
SAMPLE = ("balanced_3x1000", "composite_shift_4", "composite_logit_4",
          "composite_pwl_4", "casemix_moderate_3")


# ── the seed word itself ─────────────────────────────────────────────────────
def test_seed_word_is_a_fixed_digest_not_a_salted_hash():
    """The seed word must be a specified function of the bytes, forever."""
    for name in ("balanced_3x1000", "composite_shift_4", "casemix_moderate_3"):
        assert geometry_seed_word(name) == zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF
    # Pinned literals: if someone swaps the digest, every published Type I number
    # silently changes and this is the tripwire.
    assert geometry_seed_word("balanced_3x1000") == 2056423863
    assert geometry_seed_word("composite_shift_4") == 1121839185
    assert geometry_seed_word("casemix_moderate_3") == 44411793


def test_seed_word_is_stable_in_a_fresh_interpreter_with_a_hostile_hashseed():
    """crc32 must not move when the string-hash salt moves."""
    code = (
        "from recompute.comparators.simulate import geometry_seed_word as g;"
        "print(g('balanced_3x1000'), g('casemix_moderate_3'))"
    )
    seen = set()
    for hs in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(REPO))
        out = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                             capture_output=True, text=True, check=True)
        seen.add(out.stdout.strip())
    assert len(seen) == 1, f"seed word varies with PYTHONHASHSEED: {seen}"


# ── the datasets ─────────────────────────────────────────────────────────────
def test_make_dataset_is_deterministic_within_a_process():
    for name in SAMPLE:
        g = GEOMETRY_BY_NAME[name]
        a = make_dataset(g, rep=3, seed=42)
        b = make_dataset(g, rep=3, seed=42)
        assert np.array_equal(a[0], b[0])
        assert np.array_equal(a[1], b[1])
        for c in a[2]:
            assert np.array_equal(a[2][c], b[2][c])


def test_different_geometries_get_different_streams():
    """The seed word must still separate geometries -- crc32 is not a constant."""
    words = {g.name: geometry_seed_word(g.name) for g in GEOMETRIES}
    assert len(set(words.values())) == len(words), "seed-word collision"


@pytest.mark.parametrize("hashseed", ["0", "1", "999"])
def test_draws_are_identical_across_processes(hashseed, tmp_path):
    """THE regression test for the reported defect.

    A separate interpreter, started with a different ``PYTHONHASHSEED``, must
    produce bit-identical draws. Under the old ``abs(hash(name))`` seeding this
    fails for every geometry; under ``crc32`` it passes for all of them.
    """
    code = f"""
import sys, numpy as np
sys.path.insert(0, {str(REPO)!r})
from recompute.comparators.simulate import GEOMETRY_BY_NAME, make_dataset
out = []
for name in {list(SAMPLE)!r}:
    y, s, codes = make_dataset(GEOMETRY_BY_NAME[name], rep=7, seed=42)
    out.append(float(np.sum(y)))
    out.append(float(np.sum(s)))
    out.append(float(np.sum(s * np.arange(len(s)))))
    for c in sorted(codes):
        out.append(float(np.dot(codes[c], np.arange(len(codes[c])))))
print(" ".join(repr(v) for v in out))
"""
    env = dict(os.environ, PYTHONHASHSEED=hashseed, PYTHONPATH=str(REPO))
    got = subprocess.run([sys.executable, "-c", code], env=env, cwd=REPO,
                         capture_output=True, text=True, check=True).stdout.strip()

    here = []
    for name in SAMPLE:
        y, s, codes = make_dataset(GEOMETRY_BY_NAME[name], rep=7, seed=42)
        here += [float(np.sum(y)), float(np.sum(s)),
                 float(np.sum(s * np.arange(len(s))))]
        here += [float(np.dot(codes[c], np.arange(len(codes[c]))))
                 for c in sorted(codes)]
    assert got == " ".join(repr(v) for v in here)


def test_pooled_workers_see_one_dataset():
    """End-to-end: the same thing again, through the executor the study uses."""
    from concurrent.futures import ProcessPoolExecutor

    os.environ["PYTHONHASHSEED"] = "0"
    with ProcessPoolExecutor(max_workers=3) as ex:
        got = list(ex.map(_digest, ["composite_shift_4"] * 3
                          + ["casemix_moderate_3"] * 3))
    assert len(set(got[:3])) == 1
    assert len(set(got[3:])) == 1
    assert got[0] != got[3]


def _digest(name: str) -> str:
    """Module-level so it is picklable by the spawn-based executor."""
    import hashlib

    from recompute.comparators.simulate import GEOMETRY_BY_NAME, make_dataset

    y, s, codes = make_dataset(GEOMETRY_BY_NAME[name], rep=11, seed=42)
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(s, dtype=np.float64).tobytes())
    for c in sorted(codes):
        h.update(np.ascontiguousarray(codes[c], dtype=np.int64).tobytes())
    return h.hexdigest()


# ── the nulls the new geometries claim to satisfy ────────────────────────────
@pytest.mark.parametrize(
    "name", [g.name for g in GEOMETRIES if not g.is_case_mix])
def test_equal_auroc_geometries_really_do_hold_true_auroc_equal(name):
    """No geometry may smuggle a real effect into the equal-AUROC null."""
    res = verify_null(GEOMETRY_BY_NAME[name], n_check=120_000)
    assert res["max_studentized"] < 4.0, res
    assert res["max_gap"] < 0.05, res


#: Parameters each family is actually used at, plus headroom either side. The
#: logit-scaling map is representable only while ``a * |logit(s)|`` stays inside
#: roughly +-36.7 -- past that ``expit`` rounds to 1.0 in float64 and distinct
#: scores tie. The geometries use a <= 3.0 on scores with |logit(s)| < 6, so the
#: bound holds with a factor of two to spare; the test exercises the range that
#: is actually reachable, and `test_no_geometry_saturates_its_transform` below
#: checks the real datasets directly.
_FAMILY_PARAMS = {
    "power": (0.2, 0.45, 1.0, 3.0, 12.0),
    "logit_scale": (0.2, 0.45, 1.0, 1.8, 3.0),
    "piecewise": (0.2, 0.45, 1.0, 3.0, 12.0),
}


@pytest.mark.parametrize("family", ["power", "logit_scale", "piecewise"])
def test_every_transform_family_is_strictly_increasing(family):
    """Strict monotonicity is the whole basis of the composite null."""
    s = np.linspace(1e-4, 1 - 1e-4, 20_001)
    for a in _FAMILY_PARAMS[family]:
        t = apply_monotone(s, np.full_like(s, a), family)
        assert np.all(np.diff(t) > 0), (family, a)
        assert t.min() > 0.0 and t.max() < 1.0, (family, a)


@pytest.mark.parametrize("family", ["power", "logit_scale", "piecewise"])
def test_monotone_transform_leaves_auroc_exactly_unchanged(family):
    from recompute.comparators.core import auc_delong

    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.3).astype(int)
    s = 0.001 + 0.998 * rng.random(4000)
    before = auc_delong(y, s)[0]
    for a in _FAMILY_PARAMS[family]:
        after = auc_delong(y, apply_monotone(s, np.full_like(s, a), family))[0]
        assert abs(after - before) < 1e-12, (family, a)


def test_logit_scale_saturation_bound_is_where_it_is_documented_to_be():
    """Pin the known float64 limit rather than leaving it implicit.

    ``expit`` rounds to exactly 1.0 for arguments past about 36.7, so
    ``logit_scale`` ties scores once ``a * |logit(s)| `` crosses that. The
    geometries stay well inside it (a <= 3.0, |logit(s)| < 6 on the scores this
    is applied to); this test states the boundary so that anyone who raises
    ``a`` finds out here instead of in a silently invalid Type I cell.
    """
    s = np.linspace(1e-4, 1 - 1e-4, 20_001)            # |logit(s)| up to 9.21
    assert np.all(np.diff(apply_monotone(s, np.full_like(s, 3.0),
                                         "logit_scale")) > 0)
    ties = np.diff(apply_monotone(s, np.full_like(s, 5.0), "logit_scale")) == 0
    assert ties.any(), "saturation bound has moved; revisit the comment"
    # And the guard in make_dataset is what actually protects the study.
    from dataclasses import replace

    g = replace(GEOMETRY_BY_NAME["composite_logit_4"],
                monotone_exponents=((0.45, 1.0, 1.8, 40.0),))
    with pytest.raises(AssertionError, match="introduced ties"):
        make_dataset(replace(g, n=200_000), rep=0, seed=42)


@pytest.mark.parametrize(
    "name", [g.name for g in GEOMETRIES if g.is_composite])
def test_no_geometry_saturates_its_transform(name):
    """A saturating map would tie scores and move the true AUROC silently.

    ``make_dataset`` raises if the transform loses any distinct score, so this
    exercises that guard on large draws from every composite geometry -- the
    condition the composite null's validity actually rests on.
    """
    from dataclasses import replace

    g = GEOMETRY_BY_NAME[name]
    for rep in range(3):
        y, s, _ = make_dataset(replace(g, n=200_000), rep=rep, seed=42)
        assert np.unique(s).size == s.size          # no ties at all, in fact
        assert 0.0 < s.min() and s.max() < 1.0


@pytest.mark.parametrize(
    "name", [g.name for g in GEOMETRIES if g.is_case_mix])
def test_case_mix_geometries_have_genuinely_unequal_true_auroc(name):
    """The case-mix null is the OPPOSITE claim and must be checked as such."""
    g = GEOMETRY_BY_NAME[name]
    truth = true_subgroup_auc(g)
    assert truth["max_gap"] > 0.01, truth
    assert 0.6 < truth["mean_auc"] < 0.9, truth
    with pytest.raises(ValueError):
        verify_null(g)


@pytest.mark.parametrize(
    "name", [g.name for g in GEOMETRIES if g.is_case_mix])
def test_case_mix_simulated_auroc_matches_the_quadrature(name):
    """The exact per-level AUROC must match what the generator actually draws."""
    from dataclasses import replace

    from recompute.comparators.core import auc_delong

    g = GEOMETRY_BY_NAME[name]
    truth = true_subgroup_auc(g)
    y, s, codes = make_dataset(replace(g, n=400_000), rep=0, seed=5)
    for k in np.unique(codes["p0"]):
        m = codes["p0"] == k
        emp = auc_delong(y[m], s[m])[0]
        assert abs(emp - truth[f"level_{k}"]) < 0.01, (k, emp, truth)
    assert abs(float(y.mean()) - g.prevalence) < 0.01


@pytest.mark.parametrize(
    "name", [g.name for g in GEOMETRIES if g.is_case_mix])
def test_case_mix_model_is_perfectly_calibrated_in_every_subgroup(name):
    """No unfairness is present: the score IS the true event probability."""
    g = GEOMETRY_BY_NAME[name]
    from dataclasses import replace

    y, s, codes = make_dataset(replace(g, n=400_000), rep=0, seed=6)
    for k in np.unique(codes["p0"]):
        m = codes["p0"] == k
        # Mean predicted probability equals observed event rate, in every
        # subgroup and in every decile of predicted risk within it.
        assert abs(s[m].mean() - y[m].mean()) < 0.01, k
        q = np.quantile(s[m], np.linspace(0, 1, 6))
        for lo, hi in zip(q[:-1], q[1:]):
            b = m & (s >= lo) & (s <= hi)
            if b.sum() > 2000:
                assert abs(s[b].mean() - y[b].mean()) < 0.03, (k, lo, hi)
