"""
The SD-ratio sweep and the positive controls.

Why this module exists
----------------------
The six-geometry case-mix study in ``simulate.py`` reports the case-mix
false-alarm rate at **three** hand-chosen geometries -- per-level
linear-predictor SD ratios 1.22, 2.00 and 3.17, giving flag rates 0.162, 0.920
and 0.999 for the incumbent -- and the manuscript headlines the middle one,
describing that geometry as "a moderate and clinically ordinary amount of
case-mix heterogeneity". No citation, no empirical anchor, and ``simulate.py``
simply asserts "the realistic clinical case". The headline number is therefore a
*choice*, and it sits on the steepest part of a curve: the flag rate moves from
0.162 to 0.920 across a stretch of the SD-ratio axis sampled at exactly two
points.

Two things follow, and this module supplies both.

**A curve, not a point** (:data:`SWEEP_GEOMETRIES`). The flag rate is reported as
a function of the SD ratio on a grid dense enough to resolve the interval where
it actually moves. The grid deliberately includes ratio 1.0 -- an exact
equal-AUROC null with the case-mix machinery still switched on -- so the sweep
carries its own negative control, and includes the three hand-chosen anchors so
the published numbers appear on the curve rather than beside it.

**A positive control** (:data:`POSITIVE_CONTROL_GEOMETRIES`). All 23 geometries
in the equal-AUROC and case-mix studies are nulls: no unfairness of any kind is
present anywhere in the study.
A study with no positive control cannot distinguish "these procedures flag case
mix" from "these procedures flag everything", and cannot say anything at all
about whether an auditor could tell the two apart. Three unfairness mechanisms
are added:

``unfair_coef``
    A second covariate ``x2`` genuinely drives the outcome in one subgroup, and
    the deployed model does not use it. The model is therefore worse *for that
    subgroup than it could be*: an oracle model fitted for that subgroup alone
    would discriminate strictly better. The per-level norm is held fixed
    (``scale^2 + w^2`` constant), so the achievable AUROC is identical in every
    level and the only thing that differs is what is delivered.
``unfair_calib``
    A subgroup-specific miscalibration: the deployed score is a strictly
    increasing distortion of the model's implied risk in one level only. The
    model is genuinely wrong for that subgroup -- and every subgroup's AUROC is
    exactly unchanged, so this is real unfairness that no discrimination-based
    procedure can see even in principle.
``matched pairs``
    The question the manuscript asserts an answer to without evidence: *can any
    procedure distinguish a case-mix-driven gap from an unfairness-driven gap of
    the same magnitude?* :func:`matched_pair` builds two geometries whose **true
    per-level AUROC vectors are identical**, whose **level prevalences are
    identical** (0.20 in every level of both arms, by the per-level intercept
    solve), and which differ only in mechanism: in the case-mix arm the model is
    Bayes-optimal in every level, in the unfairness arm it throws away exactly
    ``delta`` of achievable discrimination in one level. If a procedure's
    behaviour is the same on both arms, it cannot tell them apart -- and its
    output is uninformative about fairness however well calibrated its Type I
    error is.

Pinned parameters
-----------------
The matched-pair parameters are solved by quadrature and **pinned as literals**
below rather than solved at import, so that every worker process, every rerun and
every reader gets the same geometry without depending on the local scipy's root
finder. :func:`solve_matched_pair` is the solver that produced them, and
re-running it should reproduce the pinned values below to the stated tolerance.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from recompute.comparators.simulate import (
    CaseMix,
    Geometry,
    _equal,
    true_subgroup_auc,
)

# ── (a) the SD-ratio sweep ───────────────────────────────────────────────────
#: The grid. 1.0 is the exact null; 1.222, 2.000 and 3.167 are the three
#: hand-chosen geometries (casemix_mild_3, casemix_moderate_3, casemix_strong_4)
#: so the published points land on the curve. The interval 1.25-2.0 is sampled
#: densely because that is where the flag rate travels from 0.16 to 0.92, and
#: where a reader will want the true gaps near 0.06 and 0.09 that the
#: hand-chosen anchors skipped over.
SD_RATIO_GRID: Tuple[float, ...] = (
    1.0, 1.1, 1.222, 1.25, 1.35, 1.45, 1.5, 1.6, 1.75, 1.9, 2.0, 2.25, 2.5,
    2.75, 3.0, 3.167,
)


def sweep_scales(ratio: float) -> Tuple[float, float, float]:
    """Three per-level SDs with max/min = ``ratio``, geometric about 1.0.

    ``(r**-0.5, 1, r**0.5)``. At ratio 2.0 this is (0.707, 1.0, 1.414), which
    reproduces ``casemix_moderate_3``'s (0.7, 1.0, 1.4) to within a percent, so
    the sweep is a genuine one-parameter extension of the published geometry
    rather than a different construction that happens to pass through it.
    """
    r = float(ratio)
    return (r ** -0.5, 1.0, r ** 0.5)


def sweep_name(ratio: float) -> str:
    return f"sweep_sdr_{int(round(ratio * 1000)):04d}"


SWEEP_GEOMETRIES: List[Geometry] = [
    Geometry(
        sweep_name(r), n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=sweep_scales(r)),
        description=(f"SD-ratio sweep at {r:.3f}: per-level linear-predictor SD "
                     + "/".join(f"{v:.3f}" for v in sweep_scales(r))))
    for r in SD_RATIO_GRID
]

#: The same sweep with every level's prevalence pinned at 0.20, so that the curve
#: can be read as a function of spread alone. Level prevalence otherwise moves
#: with spread by construction (M3), and the two mechanisms are confounded along
#: the whole of the main sweep.
SWEEP_PREVFIXED_GEOMETRIES: List[Geometry] = [
    Geometry(
        sweep_name(r) + "_pf", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=sweep_scales(r),
                         equalize_prevalence=True),
        description=(f"SD-ratio sweep at {r:.3f}, every level's prevalence held "
                     "at 0.20 by a per-level intercept: spread mechanism only"))
    for r in SD_RATIO_GRID
]


# ── (b) the positive controls ────────────────────────────────────────────────
def _pair_geometry(name: str, scale0: float, w0: float, description: str,
                   n: int = 2000, prevalence: float = 0.20) -> Geometry:
    """One arm of a matched pair: level 0 perturbed, levels 1 and 2 the reference.

    ``equalize_prevalence`` is on in both arms, so all six levels of a pair carry
    exactly the target event prevalence and no procedure can be responding to a
    prevalence difference instead of to the mechanism.
    """
    return Geometry(
        name, n=n, prevalence=prevalence, partitions=(_equal(3),),
        case_mix=CaseMix(
            locs=(0.0,) * 3, scales=(scale0, 1.0, 1.0),
            unfair_w=((w0, 0.0, 0.0) if w0 else None),
            equalize_prevalence=True),
        description=description)


def _gap_of(scale0: float, w0: float) -> float:
    """True max-min AUROC gap of the arm with these level-0 parameters."""
    t = true_subgroup_auc(_pair_geometry("_probe", scale0, w0, ""))
    return float(t["max_gap"])


def solve_matched_pair(delta: float) -> Dict[str, float]:
    """Solve both arms so each has true max-min AUROC gap exactly ``delta``.

    Case-mix arm: shrink level 0's predictor SD (``w = 0``, model stays
    Bayes-optimal). Unfairness arm: move variance out of the covariate the model
    sees and into the one it does not, holding ``scale^2 + w^2 = 1`` so the
    *achievable* AUROC is unchanged in every level and the entire loss is model
    failure. Returns the two parameters and the realised gaps.
    """
    from scipy.optimize import brentq

    scale0 = float(brentq(lambda c: _gap_of(c, 0.0) - delta, 0.05, 0.999,
                          xtol=1e-10, rtol=1e-12))
    w0 = float(brentq(lambda w: _gap_of(np.sqrt(max(1.0 - w * w, 1e-12)), w)
                      - delta, 1e-4, 0.995, xtol=1e-10, rtol=1e-12))
    return {
        "delta": float(delta),
        "casemix_scale0": scale0,
        "unfair_w0": w0,
        "unfair_scale0": float(np.sqrt(1.0 - w0 * w0)),
        "casemix_gap": _gap_of(scale0, 0.0),
        "unfair_gap": _gap_of(float(np.sqrt(1.0 - w0 * w0)), w0),
    }


#: Pinned solutions of :func:`solve_matched_pair`, to 1e-10 in the realised gap.
#: 0.123 is the headline case-mix geometry's gap; 0.05 is the conventional
#: "material difference" threshold the manuscript's own rules use; 0.20 is the
#: strong case. Regenerate with ``python -m recompute.comparators.positive_control
#: --retune`` and pin the printed values.
MATCHED_PAIRS: Dict[str, Dict[str, float]] = {
    "d050": {"delta": 0.050, "casemix_scale0": 0.7514484343,
             "unfair_w0": 0.6056718788},
    "d123": {"delta": 0.123, "casemix_scale0": 0.4450103090,
             "unfair_w0": 0.8674210824},
    "d200": {"delta": 0.200, "casemix_scale0": 0.1589065521,
             "unfair_w0": 0.9831688205},
}

POSITIVE_CONTROL_GEOMETRIES: List[Geometry] = []
for _tag, _par in MATCHED_PAIRS.items():
    _d = _par["delta"]
    POSITIVE_CONTROL_GEOMETRIES += [
        _pair_geometry(
            f"pc_casemix_{_tag}", _par["casemix_scale0"], 0.0,
            (f"MATCHED PAIR {_tag}, CASE-MIX arm: level 0's predictor SD is "
             f"{_par['casemix_scale0']:.4f} against 1.0 elsewhere, giving a true "
             f"AUROC gap of {_d:.3f}. The model is the exact data-generating "
             "probability and is Bayes-optimal in every level; zero unfairness")),
        _pair_geometry(
            f"pc_unfair_{_tag}", float(np.sqrt(1.0 - _par["unfair_w0"] ** 2)),
            _par["unfair_w0"],
            (f"MATCHED PAIR {_tag}, UNFAIRNESS arm: level 0's outcome is partly "
             f"driven by a covariate the model does not use (w="
             f"{_par['unfair_w0']:.4f}), giving the SAME true AUROC gap of "
             f"{_d:.3f} and the same level prevalences. The achievable AUROC is "
             f"identical in all three levels; the model throws {_d:.3f} of it "
             "away for level 0 alone")),
    ]

POSITIVE_CONTROL_GEOMETRIES += [
    # Genuine unfairness that is invisible to AUROC by construction.
    Geometry(
        "pc_calib_slope", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=(1.0, 1.0, 1.0),
                         miscal_slope=(0.5, 1.0, 1.0),
                         equalize_prevalence=True),
        description=("SUBGROUP-SPECIFIC MISCALIBRATION: level 0's score is "
                     "expit(0.5 * lp) while its outcome is drawn from expit(lp) "
                     "-- a calibration slope of 2.0 in that subgroup and 1.0 "
                     "elsewhere. Strictly increasing, so every subgroup's true "
                     "AUROC is EXACTLY equal; the model is nonetheless genuinely "
                     "and materially wrong for one subgroup")),
    Geometry(
        "pc_calib_intercept", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=(1.0, 1.0, 1.0),
                         miscal_intercept=(1.0, 0.0, 0.0),
                         equalize_prevalence=True),
        description=("SUBGROUP-SPECIFIC MISCALIBRATION IN THE LARGE: level 0's "
                     "risks are inflated by one logit. True AUROC exactly equal "
                     "in every subgroup; systematic over-prediction for one")),
    # A geometry with case mix AND unfairness at once: the realistic worst case,
    # where an auditor sees a gap that is part mechanism and part failure.
    Geometry(
        "pc_both_d123", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3,
                         scales=(0.4711829305, 1.0, 1.0),
                         unfair_w=(0.6056718788, 0.0, 0.0),
                         equalize_prevalence=True),
        description=("BOTH mechanisms at once: level 0 has both a narrower "
                     "predictor distribution and a covariate the model ignores. "
                     "Total true AUROC gap 0.123, of which 0.050 is genuine "
                     "unfairness (the excess of achievable over delivered) and "
                     "the remaining 0.073 is case mix")),
]

ALL_SWEEP_GEOMETRIES: List[Geometry] = (
    SWEEP_GEOMETRIES + SWEEP_PREVFIXED_GEOMETRIES + POSITIVE_CONTROL_GEOMETRIES)

SWEEP_GEOMETRY_BY_NAME: Dict[str, Geometry] = {g.name: g
                                       for g in ALL_SWEEP_GEOMETRIES}
