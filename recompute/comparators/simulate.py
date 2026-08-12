"""
Data-generating processes for the Type I error study.

The null being simulated
------------------------
In every geometry below, **subgroup membership is independent of the score given
the outcome**. That is the exact null all five procedures claim to test: every
subgroup has the same true AUROC. Any flag raised on this data is a false
positive, whatever the method, so the flag rate at nominal 0.05 is directly
comparable across methods. None of these procedures -- including the incumbent --
has been checked this way in this project.

Two families of geometry
------------------------
**Simple null.** Subgroup labels are drawn uniformly at random and independently
of everything. Here the score distribution is *identical* across subgroups, so
full exchangeability holds and a naive permutation test is exact. These
geometries vary what is actually thought to drive the incumbent's behaviour: the
number of levels, how unequal they are, the outcome prevalence, and the number of
partitions the maximum is taken over.

**Composite null.** Subgroup ``g``'s scores are passed through a strictly
increasing map on (0, 1), applied identically to that subgroup's positives and
negatives. AUROC is rank-based within a subgroup, so a strictly monotone
within-subgroup transform leaves every subgroup's true AUROC *exactly* unchanged
at its common value -- while the subgroups' score distributions now differ
substantially. This is precisely the composite null DiCiccio et al. studentize
against, and it is the only place where the studentization can earn its keep.
Under it, exchangeability fails and an unstudentized permutation test has no
validity guarantee. Three transform families are used -- ``s -> s**a``,
``s -> expit(a * logit(s))`` and a two-segment piecewise-linear map -- so the
result does not rest on one algebraic form; the composite cells also span 1, 3
and 5 partitions, n in {500, 2000, 10000} and prevalence in {0.05, 0.20, 0.50},
which is what exercises the maximum-over-partitions coupling the mechanism claim
depends on.

**Case-mix null.** The two families above both hold every subgroup's *true*
AUROC equal. That is the null the five procedures nominally test, but it is not
the null a clinical prediction model actually satisfies. Subgroup AUROC is
case-mix dependent: discrimination measures how well a model separates the cases
it is shown, and a subgroup whose predictor values are more spread out is easier
to separate. A single shared model with identical coefficients therefore
*expects* unequal subgroup AUROC whenever the subgroups' covariate distributions
differ -- with no unfairness of any kind present (Vergouwe et al. 2010;
van Klaveren et al. 2016; Riley et al. 2021 on the same point for calibration).

The case-mix geometries make this concrete in the strongest possible form. One
linear predictor ``lp = b0 + x`` is drawn with a per-subgroup mean and standard
deviation; the outcome is drawn as ``y ~ Bernoulli(expit(lp))``; and the score
*is* ``expit(lp)``. The model is thus not merely fair -- it is the exact
data-generating probability, perfectly calibrated in every subgroup, with one
coefficient vector for everybody. Nothing in it treats any subgroup differently.
Yet the subgroups' true AUROCs differ, by construction and by an amount
:func:`true_subgroup_auc` computes exactly. Every flag raised on this data is a
false alarm about fairness, and the flag rate is the quantity of interest.

Everything is seeded; :func:`make_dataset` is a pure function of ``(geometry,
replicate index, seed)``.

Reproducibility of the seed itself
----------------------------------
The geometry's contribution to the seed is :func:`geometry_seed_word`, a
``zlib.crc32`` digest of the geometry name. It was previously ``abs(hash(name))``,
which is **not** reproducible: CPython salts ``str.__hash__`` per interpreter
process unless ``PYTHONHASHSEED`` is pinned, so every ``ProcessPoolExecutor``
worker drew a different dataset for the same ``(geometry, rep, seed)`` and no run
could be reproduced -- not by a reader, and not by this repository. ``crc32`` is a
fixed, specified function of the bytes and carries no process state.
``recompute.comparators.type1`` additionally pins ``PYTHONHASHSEED=0`` for the
whole process tree as a belt-and-braces measure, so that any *other* incidental
use of ``hash`` in a dependency is also pinned.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class CaseMix:
    """Case-mix heterogeneity: one shared model, unequal covariate spread.

    See the "Case-mix null" section of the module docstring. ``locs`` and
    ``scales`` are the per-level mean and standard deviation of the *single*
    linear predictor. The model has one coefficient vector for everybody; only
    the population it is applied to differs.

    Structural limitation (not lifted)
    ---------------------------------
    ``CaseMix`` attaches to partition 0 only: ``locs`` and ``scales`` are indexed
    by the level codes of ``geom.partitions[0]``, and every other partition is a
    pure noise column drawn independently. The class therefore **cannot express
    case mix on two partitions at once**, which is what a real cohort has (age
    and race both shift the predictor distribution, and they are correlated).
    Lifting it needs a joint covariate model per cell of the partition cross,
    which is a different DGP; until then every simulated case-mix result is a
    *single*-partition result and the multi-partition cells vary only the number
    of noise columns the maximum runs over.

    Extensions to the DGP (all default to the original case-mix behaviour, so
    every pre-existing geometry draws bit-identical data)
    ----------------------------------------------------------------------------
    ``unfair_w``
        Per-level coefficient on a second covariate ``x2 ~ N(0, 1)`` that drives
        the outcome but that **the deployed model does not use**. With ``w_g > 0``
        level ``g``'s outcome is genuinely partly determined by something the
        model ignores, so the model is worse for that subgroup *than it could
        be*: an oracle model fitted for that subgroup alone would discriminate
        better. This is the positive control -- genuine, subgroup-specific
        differential model performance -- and it is the one thing the original
        six-geometry case-mix study contained none of. ``None`` (the default) means ``w = 0``
        everywhere, no second covariate is drawn, and the model is Bayes-optimal
        in every level.
    ``miscal_intercept`` / ``miscal_slope``
        Per-level ``a_g`` and ``lambda_g`` applied to the *deployed* linear
        predictor only: ``score = expit(a_g + lambda_g * lp)``. The outcome is
        still drawn from ``expit(lp)``, so the model is genuinely miscalibrated
        in level ``g`` -- and because ``a_g + lambda_g * lp`` is strictly
        increasing in ``lp`` for ``lambda_g > 0``, every subgroup's AUROC is
        exactly unchanged. This is the *other* positive control: real unfairness
        that a discrimination-based procedure is structurally blind to.
    ``lp_dist``
        Which standardised (zero mean, unit variance) family the linear
        predictor's noise is drawn from: ``normal``, ``t5`` (heavy tails),
        ``laplace`` (peaked) or ``skewnorm5`` (asymmetric).
    ``equalize_prevalence``
        Solve a per-level intercept so that **every level has the target event
        prevalence**, instead of solving one intercept for the mixture. Level
        prevalence and predictor spread otherwise move together by construction
        (``casemix_location_3`` has level prevalences 0.022 / 0.129 / 0.449), and
        nothing in the original case-mix design separates them. A subgroup intercept is not
        unfairness: sex and age are model *inputs* in every real cohort here, so
        a model with a subgroup-specific intercept is still one correctly
        specified shared model.
    """

    locs: Tuple[float, ...]
    scales: Tuple[float, ...]
    unfair_w: Optional[Tuple[float, ...]] = None
    miscal_intercept: Optional[Tuple[float, ...]] = None
    miscal_slope: Optional[Tuple[float, ...]] = None
    lp_dist: str = "normal"
    equalize_prevalence: bool = False

    @property
    def has_unfair_coef(self) -> bool:
        return self.unfair_w is not None and any(w != 0.0 for w in self.unfair_w)

    @property
    def has_miscalibration(self) -> bool:
        return (self.miscal_intercept is not None
                or self.miscal_slope is not None)

    @property
    def sd_ratio(self) -> float:
        """max/min of the per-level linear-predictor SD -- the headline knob.

        This is the quantity the manuscript's "moderate and clinically ordinary"
        claim is really about, and the quantity
        ``recompute/results/cohort_sd_ratios.csv`` measures in the ten real
        cohorts. With ``unfair_w`` present the *total* per-level SD of the true
        linear predictor is ``sqrt(scale^2 + w^2)``; ``scales`` alone is the SD
        of the part the deployed model sees.
        """
        s = np.asarray(self.scales, dtype=float)
        return float(s.max() / s.min())


@dataclass(frozen=True)
class Geometry:
    """One subgroup geometry for the Type I error study."""

    name: str
    n: int
    prevalence: float
    #: One entry per demographic partition: the level proportions of that column.
    partitions: Tuple[Tuple[float, ...], ...]
    #: True cohort AUROC. Held equal across subgroups by construction.
    auc: float = 0.75
    #: Per-level parameters of the strictly monotone score map, per partition.
    #: ``None`` means the identity map, i.e. the simple (exchangeable) null.
    #: Only the FIRST partition's entry is used; see :func:`make_dataset`.
    monotone_exponents: Optional[Tuple[Tuple[float, ...], ...]] = None
    #: Which strictly increasing family the parameters above index.
    #: ``power``       s -> s ** a                       (a > 0)
    #: ``logit_scale`` s -> expit(a * logit(s))          (a > 0)
    #: ``piecewise``   two-segment piecewise-linear on (0, 1) with knot at 0.5
    #:                 and slope ratio a; strictly increasing for every a > 0.
    #: All three are strictly increasing on (0, 1) and therefore leave every
    #: subgroup's true AUROC exactly unchanged. ``power`` is the default so the
    #: pre-existing geometries are bit-for-bit unchanged.
    transform: str = "power"
    #: Case-mix heterogeneity spec; mutually exclusive with the transform above.
    case_mix: Optional[CaseMix] = None
    description: str = ""

    @property
    def is_composite(self) -> bool:
        return self.monotone_exponents is not None

    @property
    def is_case_mix(self) -> bool:
        return self.case_mix is not None

    @property
    def null_family(self) -> str:
        if self.case_mix is not None:
            return "case_mix"
        return "composite" if self.is_composite else "simple"


def _equal(k: int) -> Tuple[float, ...]:
    return tuple([1.0 / k] * k)


def geometry_seed_word(name: str) -> int:
    """Stable 31-bit seed word for a geometry name.

    ``zlib.crc32`` is a specified function of the input bytes with no process
    state, unlike ``hash``, whose string salt is randomised per interpreter
    unless ``PYTHONHASHSEED`` is pinned. Using ``hash`` here meant that the two
    ``ProcessPoolExecutor`` workers that ran two cells of the same geometry drew
    *different* data for the same ``(geometry, rep, seed)``, and that no run was
    reproducible. See ``tests/test_type1_reproducibility.py``.
    """
    return int(zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF)


#: The geometries. Sizes are chosen to bracket the real cohorts: n from 1,000 to
#: 5,000, prevalence 0.075 (NHIS 2024) to 0.30, 2 to 10 levels, 1 to 5
#: partitions, balanced through to severely skewed level sizes.
GEOMETRIES: List[Geometry] = [
    Geometry(
        "balanced_3x1000", n=3000, prevalence=0.20, partitions=(_equal(3),),
        description="3 equal levels of 1000, one partition -- the easy case"),
    Geometry(
        "balanced_5x200", n=1000, prevalence=0.20, partitions=(_equal(5),),
        description="5 equal levels of 200 -- small but balanced"),
    Geometry(
        "skewed_5", n=2000, prevalence=0.20,
        partitions=((0.55, 0.25, 0.10, 0.07, 0.03),),
        description="5 levels, sizes 1100/500/200/140/60 -- realistic skew"),
    Geometry(
        "many_10", n=2000, prevalence=0.20, partitions=(_equal(10),),
        description="10 equal levels of 200 -- maximum-of-many pressure"),
    Geometry(
        "rare_outcome", n=2000, prevalence=0.075,
        partitions=((0.55, 0.25, 0.10, 0.07, 0.03),),
        description="NHIS 2024's prevalence: the smallest level carries ~4 events"),
    Geometry(
        "multi_partition", n=2000, prevalence=0.20,
        partitions=(_equal(2), _equal(3), (0.5, 0.3, 0.2),
                    (0.4, 0.3, 0.2, 0.1), _equal(5)),
        description="5 partitions -- exercises the maximum over columns"),
    Geometry(
        "composite_shift_4", n=2000, prevalence=0.20, partitions=(_equal(4),),
        monotone_exponents=((0.4, 1.0, 2.0, 5.0),),
        description=("4 equal levels, per-level strictly monotone score map: "
                     "equal true AUROC, very different score distributions")),
    Geometry(
        "composite_shift_skewed", n=2000, prevalence=0.20,
        partitions=((0.55, 0.25, 0.10, 0.07, 0.03),),
        monotone_exponents=((0.4, 0.7, 1.0, 2.5, 5.0),),
        description="composite null with skewed level sizes -- the hardest cell"),
]

# ── Composite-null geometries, thick enough to test the mechanism ─────────────
# The original composite null was two cells, both single-partition, both n=2000
# at prevalence 0.20, both using s -> s**a. It therefore never exercised the
# maximum-over-partitions coupling the paper's mechanism claim rests on, never
# varied the sample size or the event count, and could not distinguish a property
# of the composite null from a property of the power family. These cells do.
_C4 = (0.4, 1.0, 2.0, 5.0)                     # the established 4-level shift
_MULTI3 = (_equal(4), _equal(3), (0.5, 0.3, 0.2))
_MULTI5 = (_equal(4), _equal(2), _equal(3), (0.5, 0.3, 0.2), (0.4, 0.3, 0.2, 0.1))

GEOMETRIES += [
    # -- multiple partitions under the composite null (the coupling cells) -----
    Geometry(
        "composite_3part", n=2000, prevalence=0.20, partitions=_MULTI3,
        monotone_exponents=(_C4,),
        description=("3 partitions, composite shift on the first -- the max "
                     "runs over transformed and untransformed columns")),
    Geometry(
        "composite_5part", n=2000, prevalence=0.20, partitions=_MULTI5,
        monotone_exponents=(_C4,),
        description=("5 partitions, composite shift on the first -- maximum "
                     "over-partition pressure under the composite null")),
    # -- sample size ----------------------------------------------------------
    Geometry(
        "composite_n500", n=500, prevalence=0.20, partitions=(_equal(4),),
        monotone_exponents=(_C4,),
        description="composite shift at n=500 -- 25 events per level"),
    Geometry(
        "composite_n10000", n=10_000, prevalence=0.20, partitions=(_equal(4),),
        monotone_exponents=(_C4,),
        description="composite shift at n=10,000 -- the asymptotic end"),
    # -- prevalence -----------------------------------------------------------
    Geometry(
        "composite_prev005", n=2000, prevalence=0.05, partitions=(_equal(4),),
        monotone_exponents=(_C4,),
        description="composite shift at prevalence 0.05 -- ~25 events per level"),
    Geometry(
        "composite_prev050", n=2000, prevalence=0.50, partitions=(_equal(4),),
        monotone_exponents=(_C4,),
        description="composite shift at prevalence 0.50 -- balanced outcome"),
    # -- a monotone map that is not a power ------------------------------------
    Geometry(
        "composite_logit_4", n=2000, prevalence=0.20, partitions=(_equal(4),),
        monotone_exponents=((0.45, 1.0, 1.8, 3.0),), transform="logit_scale",
        description=("4 levels, logistic (logit-scaling) monotone map -- "
                     "non-power check on the composite-null result")),
    Geometry(
        "composite_pwl_4", n=2000, prevalence=0.20, partitions=(_equal(4),),
        monotone_exponents=((0.2, 1.0, 4.0, 12.0),), transform="piecewise",
        description=("4 levels, two-segment piecewise-linear monotone map -- "
                     "a kinked, non-smooth transform")),
    Geometry(
        "composite_logit_5part", n=2000, prevalence=0.20, partitions=_MULTI5,
        monotone_exponents=((0.45, 1.0, 1.8, 3.0),), transform="logit_scale",
        description=("5 partitions with a logistic monotone map on the first -- "
                     "coupling and transform family varied together")),
]

# ── Case-mix geometries: unequal true subgroup AUROC with no unfairness ───────
# The realistic null for a clinical prediction model. One shared, perfectly
# calibrated, correctly specified model; subgroups differ only in the spread and
# location of the predictor. ``scales`` is the per-level SD of the single linear
# predictor -- wider spread means an easier separation problem and a genuinely
# higher true AUROC, with identical coefficients throughout.
GEOMETRIES += [
    Geometry(
        "casemix_mild_3", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.9, 1.0, 1.1)),
        description=("3 levels, predictor SD 0.9/1.0/1.1 (SD ratio 1.22): true "
                     "AUROC 0.726/0.745/0.762, gap 0.036 -- mild case-mix "
                     "heterogeneity, BELOW the 0.05 convention")),
    Geometry(
        "casemix_moderate_3", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.7, 1.0, 1.4)),
        description=("3 levels, predictor SD 0.7/1.0/1.4 (SD ratio 2.00): true "
                     "AUROC 0.684/0.745/0.807, gap 0.123. This SD ratio is "
                     "called 'the realistic clinical case'; the empirical SD "
                     "ratios measured from the ten fitted cohorts (see "
                     "recompute/results/cohort_sd_ratios.csv and "
                     "docs/sd_ratio_robustness.md) show it is NOT the median "
                     "real geometry")),
    Geometry(
        "casemix_strong_4", n=2000, prevalence=0.20, partitions=(_equal(4),),
        case_mix=CaseMix(locs=(0.0,) * 4, scales=(0.6, 0.9, 1.3, 1.9)),
        description=("4 levels, predictor SD 0.6 to 1.9 (SD ratio 3.17): true "
                     "AUROC 0.661 to 0.860, gap 0.199 -- strong case-mix "
                     "heterogeneity")),
    Geometry(
        "casemix_location_3", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(-2.0, 0.0, 2.0), scales=(1.0, 1.0, 1.0)),
        description=("3 levels differing in predictor LOCATION only, equal "
                     "spread: true AUROC gap 0.017 -- separates case-mix "
                     "location from case-mix spread. CAUTION: the induced level "
                     "prevalences are 0.022/0.129/0.449, a 20-fold ratio, so "
                     "level 0 carries ~15 expected events and is DROPPED by the "
                     "ev10 rule in a few percent of replicates; m30 and ev10 do "
                     "NOT agree on this geometry. See casemix_location_mild_3")),
    Geometry(
        "casemix_moderate_3_n10000", n=10_000, prevalence=0.20,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.7, 1.0, 1.4)),
        description=("moderate case-mix at n=10,000 -- five times the data. The "
                     "true gap is 0.122986 against 0.122985 at n=2,000: level "
                     "sizes round differently, so the solved intercept differs "
                     "in the fifth decimal (-1.6714541 vs -1.6713141) and the "
                     "gaps are equal to 1e-6, not identical")),
    Geometry(
        "casemix_moderate_3part", n=2000, prevalence=0.20,
        partitions=(_equal(3), _equal(2), (0.5, 0.3, 0.2)),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.7, 1.0, 1.4)),
        description=("moderate case-mix on the first of 3 partitions -- the "
                     "other two are pure noise columns")),
]

# ── Case-mix family, swept as hard as the composite one ───────────────────────
# Round 2 expanded the composite null over n, prevalence, partitions and
# transform family, and left the case-mix family at a single n, a single
# prevalence, equal level sizes and a Gaussian linear predictor -- while the
# case-mix family is the one the paper's conclusion rests on. Every cell below is
# EXPECTED to lower the flag rate relative to casemix_moderate_3 (less data,
# fewer events, or a level that the inclusion rule can drop), which is precisely
# why they have to be reported: an untested axis that can only move the headline
# down is the same failure mode this project has now hit three times.
_CM_MOD = (0.7, 1.0, 1.4)                       # the headline SD ratio of 2.00

GEOMETRIES += [
    # ── M1: sample size, level balance, prevalence, predictor family ─────────
    Geometry(
        "casemix_moderate_n500", n=500, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD),
        description=("moderate case mix at n=500 -- ~33 events per level, the "
                     "smallest cohort size the study brackets")),
    Geometry(
        "casemix_moderate_unequal", n=2000, prevalence=0.20,
        partitions=((0.55, 0.30, 0.15),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD),
        description=("moderate case mix with unequal level sizes 1100/600/300 "
                     "-- the widest-spread level is also the smallest")),
    Geometry(
        "casemix_moderate_prev005", n=2000, prevalence=0.05,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD),
        description="moderate case mix at prevalence 0.05 -- ~33 events/level"),
    Geometry(
        "casemix_moderate_prev050", n=2000, prevalence=0.50,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD),
        description="moderate case mix at prevalence 0.50 -- balanced outcome"),
    Geometry(
        "casemix_moderate_t5", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD, lp_dist="t5"),
        description=("moderate case mix with a heavy-tailed (standardised t_5) "
                     "linear predictor -- same per-level SD, different shape")),
    Geometry(
        "casemix_moderate_laplace", n=2000, prevalence=0.20,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD, lp_dist="laplace"),
        description=("moderate case mix with a peaked (standardised Laplace) "
                     "linear predictor")),
    Geometry(
        "casemix_moderate_skew", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD, lp_dist="skewnorm5"),
        description=("moderate case mix with an asymmetric (standardised "
                     "skew-normal, shape 5) linear predictor")),
    # ── M3: spread and prevalence separated ──────────────────────────────────
    Geometry(
        "casemix_spread_prevfixed", n=2000, prevalence=0.20,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0,) * 3, scales=_CM_MOD,
                         equalize_prevalence=True),
        description=("SPREAD ONLY: predictor SD 0.7/1.0/1.4 with a per-level "
                     "intercept holding every level's prevalence at 0.20 "
                     "exactly. The other half of the M3 pair with "
                     "casemix_location_3 (prevalence only, spread fixed); "
                     "casemix_moderate_3 confounds the two")),
    Geometry(
        "casemix_location_mild_3", n=2000, prevalence=0.20,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(-1.0, 0.0, 1.0), scales=(1.0, 1.0, 1.0)),
        description=("PREVALENCE ONLY, mild: locations -1/0/+1, equal spread. "
                     "Level prevalences ~0.09/0.20/0.38, a 4-fold rather than "
                     "20-fold ratio, so no level is near the ev10 boundary and "
                     "the two inclusion rules can be compared without the "
                     "level-dropping confound that afflicts casemix_location_3")),
]

GEOMETRY_BY_NAME = {g.name: g for g in GEOMETRIES}

#: Grouping used by the reporting layer.
FAMILY_BY_NAME = {g.name: g.null_family for g in GEOMETRIES}


def _level_codes(n: int, props: Sequence[float],
                 rng: np.random.Generator) -> np.ndarray:
    """Fixed level sizes, assigned to rows uniformly at random.

    Sizes are fixed by construction (rather than multinomial) so that the
    geometry is exactly what the table says it is in every replicate; *which*
    rows get which label is random and independent of score and outcome, which
    is what makes this the null.
    """
    counts = np.floor(np.asarray(props, dtype=float) * n).astype(int)
    counts[0] += n - counts.sum()
    codes = np.repeat(np.arange(len(counts)), counts).astype(np.int32)
    return rng.permutation(codes)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_monotone(s: np.ndarray, a: np.ndarray, family: str) -> np.ndarray:
    """Apply a strictly increasing per-row map to scores in (0, 1).

    ``a`` is the per-row parameter (already expanded from the level codes). Each
    family is strictly increasing on (0, 1) for every positive parameter, which
    is the only property the composite null needs: a strictly increasing map
    applied uniformly within a subgroup leaves that subgroup's AUROC exactly
    unchanged, because AUROC depends on the scores only through their ranks.
    """
    if family == "power":
        return np.power(s, a)
    if family == "logit_scale":
        # expit(a * logit(s)): the natural logistic-family analogue of a power
        # map, strictly increasing for a > 0, and not a power of s.
        #
        # Representability. The map is strictly increasing as a real function,
        # but its OUTPUT is a float64 in (0, 1), and expit(x) rounds to exactly
        # 1.0 once x exceeds about 36.7. So a * logit(s) must stay inside roughly
        # +-36.7 or distinct scores collapse onto 1.0, creating ties that would
        # move the AUROC. The scores this is applied to are expit of a standard
        # normal shifted by at most ~1.5, so |logit(s)| stays under ~6 even in
        # the tail of ten million draws, and the largest ``a`` in use is 3.0 --
        # a factor of two of headroom. :func:`make_dataset` asserts on every
        # dataset that no ties were introduced, so this bound is checked rather
        # than assumed.
        return _expit(a * (np.log(s) - np.log1p(-s)))
    if family == "piecewise":
        # Two segments meeting at (0.5, 0.5). Below the knot the slope is
        # a / (a + 1) * 2 relative to identity, above it the complement, so the
        # map is a bijection of (0, 1) onto itself, strictly increasing, and has
        # a kink -- deliberately not smooth.
        lo = np.broadcast_to(1.0 / (1.0 + np.asarray(a, dtype=float)), s.shape)
        below = s < 0.5
        out = np.empty_like(s)
        out[below] = (s[below] / 0.5) * lo[below]
        out[~below] = lo[~below] + ((s[~below] - 0.5) / 0.5) * (1.0 - lo[~below])
        return out
    raise ValueError(f"unknown monotone transform family {family!r}")


def _mixture_weights(geom: Geometry) -> np.ndarray:
    """Exact level proportions of partition 0, as :func:`_level_codes` builds them."""
    props = geom.partitions[0]
    counts = np.floor(np.asarray(props, dtype=float) * geom.n).astype(int)
    counts[0] += geom.n - counts.sum()
    return counts / float(geom.n)


_GH_Z = np.linspace(-9.0, 9.0, 3601)
_GH_W = np.exp(-0.5 * _GH_Z ** 2) / np.sqrt(2.0 * np.pi)


def _hermite_nodes(n: int = 96):
    """Probabilists' Gauss-Hermite nodes and normalised weights for N(0, 1)."""
    from numpy.polynomial.hermite_e import hermegauss

    x, w = hermegauss(n)
    return x, w / w.sum()


_HERMITE = _hermite_nodes()

#: The standardised (zero mean, unit variance) linear-predictor noise families.
LP_FAMILIES = ("normal", "t5", "laplace", "skewnorm5")

_SKEW_A = 5.0
_SKEW_D = _SKEW_A / np.sqrt(1.0 + _SKEW_A ** 2)
_SKEW_M = _SKEW_D * np.sqrt(2.0 / np.pi)
_SKEW_S = np.sqrt(1.0 - 2.0 * _SKEW_D ** 2 / np.pi)


def lp_standard_pdf(z: np.ndarray, family: str) -> np.ndarray:
    """Density of the standardised linear-predictor noise, mean 0 variance 1."""
    z = np.asarray(z, dtype=float)
    if family == "normal":
        return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi)
    if family == "t5":
        from scipy.stats import t as _t
        c = np.sqrt(5.0 / 3.0)                 # SD of a t_5 variate
        return _t.pdf(z * c, 5) * c
    if family == "laplace":
        b = 1.0 / np.sqrt(2.0)
        return np.exp(-np.abs(z) / b) / (2.0 * b)
    if family == "skewnorm5":
        from scipy.stats import skewnorm
        return skewnorm.pdf(z * _SKEW_S + _SKEW_M, _SKEW_A) * _SKEW_S
    raise ValueError(f"unknown linear-predictor family {family!r}")


def lp_standard_draw(rng: np.random.Generator, family: str,
                     size: int) -> np.ndarray:
    """Draw from the same standardised family :func:`lp_standard_pdf` describes."""
    if family == "normal":
        return rng.standard_normal(size)
    if family == "t5":
        return rng.standard_t(5, size) / np.sqrt(5.0 / 3.0)
    if family == "laplace":
        return rng.laplace(0.0, 1.0 / np.sqrt(2.0), size)
    if family == "skewnorm5":
        # Azzalini's construction: delta |U0| + sqrt(1 - delta^2) U1, standardised.
        u0 = np.abs(rng.standard_normal(size))
        u1 = rng.standard_normal(size)
        v = _SKEW_D * u0 + np.sqrt(1.0 - _SKEW_D ** 2) * u1
        return (v - _SKEW_M) / _SKEW_S
    raise ValueError(f"unknown linear-predictor family {family!r}")


def lp_standard_sf(z: float, family: str) -> float:
    """Upper-tail mass of the standardised family beyond ``z`` (``z >= 0``)."""
    if family == "normal":
        return float(norm.sf(z))
    if family == "t5":
        from scipy.stats import t as _t
        return float(_t.sf(z * np.sqrt(5.0 / 3.0), 5))
    if family == "laplace":
        return float(0.5 * np.exp(-z * np.sqrt(2.0)))
    if family == "skewnorm5":
        from scipy.stats import skewnorm
        return float(skewnorm.sf(z * _SKEW_S + _SKEW_M, _SKEW_A))
    raise ValueError(f"unknown linear-predictor family {family!r}")


#: Tail mass a quadrature grid is allowed to omit. An earlier version hardcoded
#: ``t = +-40`` with no check at all, which is safe for a standard normal at
#: scale <= 2 and silently wrong for a heavy-tailed family or a wide scale: a
#: truncated tail biases both the level prevalence and the level AUROC, and
#: neither error is visible in the output it corrupts. Everything below goes
#: through :func:`_lp_grid`, which widens until this bound is met and raises if it
#: cannot be -- the case-mix analogue of the composite path's tie assertion.
#: Truncation is bounded by the family's own survival function rather than by a
#: numerical mass check, so the guard measures truncation only and is not
#: confounded with the trapezoid rule's discretisation error (which is O(dz^2)
#: and, for the cusped Laplace density, is the larger of the two at any usable
#: grid size).
_TAIL_TOL = 1e-10

#: Grid points per level. Non-smooth or heavy-tailed families get a finer grid;
#: ``normal`` keeps 40,001, the original value, so its numbers do not move.
_GRID_POINTS = {"normal": 40_001, "skewnorm5": 80_001,
                "t5": 120_001, "laplace": 160_001}


def _lp_grid(centre: float, scale: float, family: str, w: float = 0.0,
             n_points: Optional[int] = None) -> np.ndarray:
    """A grid whose omitted tail mass is at most :data:`_TAIL_TOL`."""
    if n_points is None:
        n_points = _GRID_POINTS.get(family, 40_001)
    span = 12.0
    for _ in range(40):
        # The extra 8w covers the N(0, w^2) the unfair covariate convolves in.
        tail = lp_standard_sf(span, family) + norm.sf(8.0) if w else (
            lp_standard_sf(span, family))
        if tail <= _TAIL_TOL:
            half = span * scale + 8.0 * w + 1.0
            return np.linspace(centre - half, centre + half, n_points)
        span *= 1.5
    raise AssertionError(
        f"no usable quadrature grid for family {family!r}: the tail beyond "
        f"{span} standardised units still holds {tail!r} of the mass, so the "
        "reported AUROC and prevalence would be biased by the truncation.")


def _outcome_prob_given_deployed(u: np.ndarray, w: float) -> np.ndarray:
    """``E[expit(u + w Z)]`` with ``Z ~ N(0,1)``: P(Y=1 | deployed lp = u).

    With ``w = 0`` the deployed linear predictor *is* the true one and this is
    just ``expit(u)``. With ``w > 0`` the outcome also depends on a covariate the
    model never sees, so the model's implied risk is an attenuated version of the
    truth -- which is exactly what makes the deployed score sub-optimal in that
    subgroup rather than merely differently distributed.
    """
    if w == 0.0:
        return _expit(u)
    # Gauss-Hermite, not the trapezoid grid used elsewhere: ``u`` has tens of
    # thousands of points and the outer product with a 3,601-point grid is a
    # gigabyte per call. 96 probabilists' nodes agree with that grid to 2e-13 on
    # this integrand at 1/40th of the memory and about 500x the speed.
    x, wt = _HERMITE
    return _expit(u[:, None] + w * x[None, :]) @ wt


def _auc_from_density(grid: np.ndarray, dens: np.ndarray,
                      p_event: np.ndarray) -> float:
    """AUROC of ranking by ``grid`` when P(Y=1|grid) = ``p_event``."""
    f_pos = dens * p_event
    f_neg = dens * (1.0 - p_event)
    m_pos = float(np.trapz(f_pos, grid))
    m_neg = float(np.trapz(f_neg, grid))
    if m_pos <= 0 or m_neg <= 0:
        return float("nan")
    f_pos = f_pos / m_pos
    f_neg = f_neg / m_neg
    dt = grid[1] - grid[0]
    # Midpoint-corrected CDF: the coincident mass contributes one half, which
    # removes the O(dt) bias of a plain cumulative sum.
    cdf_neg = np.cumsum(f_neg) * dt - 0.5 * f_neg * dt
    return float(np.trapz(f_pos * cdf_neg, grid))


def _case_mix_is_original_shaped(cm: "CaseMix") -> bool:
    """True when the original case-mix code path must be reproduced bit-for-bit.

    The intercept ``b0`` feeds straight into the simulated linear predictor, so
    changing how it is solved changes every drawn dataset. Every geometry that
    predates the ``unfair_w`` / ``miscal_*`` / ``equalize_prevalence`` extensions
    therefore keeps the original solve, to the last bit, and only the new knobs
    take the generalised path.
    """
    return (cm.lp_dist == "normal" and not cm.equalize_prevalence
            and not cm.has_unfair_coef)


def case_mix_intercept(geom: Geometry) -> float:
    """Intercept ``b0`` of the shared model that hits the target prevalence.

    Solved deterministically by quadrature on the level-size-weighted mixture of
    the linear predictor, so the same geometry always yields the same intercept
    and the prevalence column of the results table is exact rather than nominal.

    Note that this matches the *mixture* prevalence only. Each level's own
    prevalence is then whatever the geometry implies, and they can differ by an
    order of magnitude -- ``casemix_location_3``'s levels sit at 0.022 / 0.129 /
    0.449. Use ``CaseMix(equalize_prevalence=True)`` for the per-level solve;
    :func:`case_mix_offsets` returns both.
    """
    return case_mix_offsets(geom)[0]


@lru_cache(maxsize=None)
def _case_mix_offsets_cached(geom: Geometry) -> Tuple[float, Tuple[float, ...]]:
    return _case_mix_offsets(geom)


def case_mix_offsets(geom: Geometry) -> Tuple[float, np.ndarray]:
    """``(b0, d)``: the shared intercept and the per-level intercept offsets.

    Memoised on the geometry, which is a frozen dataclass of tuples and
    therefore hashable. This is not an optimisation of convenience: the solve
    runs inside :func:`make_dataset`, i.e. once per simulated dataset, and the
    per-level solve required by ``equalize_prevalence`` costs more than the
    permutation test that consumes its output. Memoising it cuts the case-mix
    sweep from roughly 33 core-hours to 13. The value is a deterministic
    function of the geometry, so caching cannot change any result.
    """
    b0, d = _case_mix_offsets_cached(geom)
    return b0, np.asarray(d, dtype=float)


def _case_mix_offsets(geom: Geometry) -> Tuple[float, Tuple[float, ...]]:
    """``(b0, d)``: the shared intercept and the per-level intercept offsets.

    ``d`` is all zeros unless ``equalize_prevalence`` is set, in which case
    ``b0`` is fixed at zero and each ``d_g`` is solved so that level ``g``'s own
    event prevalence equals ``geom.prevalence`` exactly. That is what separates
    the spread mechanism from the prevalence mechanism: with the offsets in place
    the levels differ *only* in the spread of the linear predictor.
    """
    from scipy.optimize import brentq

    cm = geom.case_mix
    assert cm is not None
    locs = np.asarray(cm.locs, dtype=float)
    scales = np.asarray(cm.scales, dtype=float)
    ws = (np.asarray(cm.unfair_w, dtype=float) if cm.unfair_w is not None
          else np.zeros_like(scales))

    if cm.equalize_prevalence:
        d = np.empty_like(scales)
        for k in range(len(scales)):
            def lvl_p(dk: float, k: int = k) -> float:
                return _level_prevalence(0.0, float(locs[k] + dk),
                                         float(scales[k]), float(ws[k]),
                                         cm.lp_dist)
            d[k] = brentq(lambda x: lvl_p(x) - geom.prevalence, -40.0, 40.0,
                          xtol=1e-12, rtol=1e-14)
        return 0.0, tuple(float(v) for v in d)

    if _case_mix_is_original_shaped(cm):
        # ── the original solve, preserved verbatim: it feeds the DGP ─────────
        w = _mixture_weights(geom)

        def mean_p(b0: float) -> float:
            # E[expit(b0 + loc_g + scale_g Z)] averaged over levels, weights w.
            lp = b0 + locs[:, None] + scales[:, None] * _GH_Z[None, :]
            per_level = np.trapz(_expit(lp) * _GH_W[None, :], _GH_Z, axis=1)
            return float(np.dot(w, per_level))

        b0 = float(brentq(lambda b: mean_p(b) - geom.prevalence, -30.0, 30.0,
                          xtol=1e-12, rtol=1e-14))
        return b0, tuple(0.0 for _ in scales)

    wts = _mixture_weights(geom)

    def mixture_p(b0: float) -> float:
        return float(sum(
            wts[k] * _level_prevalence(b0, float(locs[k]), float(scales[k]),
                                       float(ws[k]), cm.lp_dist)
            for k in range(len(scales))))

    b0 = float(brentq(lambda b: mixture_p(b) - geom.prevalence, -30.0, 30.0,
                      xtol=1e-12, rtol=1e-14))
    return b0, tuple(0.0 for _ in scales)


def _level_prevalence(b0: float, loc: float, scale: float, w: float,
                      family: str) -> float:
    """``E[expit(b0 + loc + scale Z + w Z2)]`` for one level, by quadrature."""
    centre = b0 + loc
    grid = _lp_grid(centre, scale, family, w)
    dens = lp_standard_pdf((grid - centre) / scale, family) / scale
    return float(np.trapz(dens * _outcome_prob_given_deployed(grid, w), grid))


def true_subgroup_auc(geom: Geometry) -> Dict[str, float]:
    """Exact true behaviour of every level of partition 0, by quadrature.

    For a fair case-mix geometry the deployed score is a strictly increasing
    function of the linear predictor ``L``, so within level ``g``

        AUROC_g = P(L_case > L_control) = int f_pos(t) F_neg(t) dt

    with ``f_pos(t) prop. f_g(t) p(t)`` and ``f_neg(t) prop. f_g(t)(1 - p(t))``,
    ``p(t) = P(Y = 1 | deployed lp = t)``. Computed on a checked grid rather than
    simulated, so the reported magnitudes carry no Monte-Carlo error.

    Keys
    ----
    ``level_k``      true AUROC of the **deployed** score in level ``k``
    ``oracle_k``     true AUROC of the best possible score in level ``k``, i.e.
                     of ranking by the true linear predictor. Equal to
                     ``level_k`` whenever the model is Bayes-optimal there;
                     strictly larger when ``unfair_w`` is on.
    ``excess_k``     ``oracle_k - level_k``: the discrimination the deployed
                     model *throws away* in level ``k``. This is the definition
                     of genuine unfairness used by the positive control -- a gap
                     between what is achievable for a subgroup and what is
                     delivered to it -- and it is exactly zero for every
                     case-mix geometry, however large that geometry's AUROC gap.
    ``prev_k``       level ``k``'s own event prevalence (M3: reported for every
                     geometry, because it moves with the spread by construction).
    ``max_gap``      max-min of ``level_k`` -- the observable AUROC gap.
    ``max_excess``   max of ``excess_k`` -- the unobservable unfairness.
    ``prev_ratio``   max/min of ``prev_k``.
    ``sd_ratio``     max/min of the per-level deployed linear-predictor SD.
    """
    if geom.case_mix is None:
        return {}
    cm = geom.case_mix
    b0, d = case_mix_offsets(geom)
    locs = np.asarray(cm.locs, dtype=float)
    scales = np.asarray(cm.scales, dtype=float)
    ws = (np.asarray(cm.unfair_w, dtype=float) if cm.unfair_w is not None
          else np.zeros_like(scales))

    out: Dict[str, float] = {}
    for k in range(len(scales)):
        centre = b0 + d[k] + locs[k]
        scale = float(scales[k])
        w = float(ws[k])
        grid = _lp_grid(centre, scale, cm.lp_dist, w)
        dens = lp_standard_pdf((grid - centre) / scale, cm.lp_dist) / scale
        p_event = _outcome_prob_given_deployed(grid, w)
        out[f"level_{k}"] = _auc_from_density(grid, dens, p_event)
        out[f"prev_{k}"] = float(np.trapz(dens * p_event, grid))
        if w == 0.0:
            out[f"oracle_{k}"] = out[f"level_{k}"]
        else:
            # The oracle ranks by the TRUE linear predictor v = u + w Z2, whose
            # density is the convolution of the level's density with N(0, w^2).
            sd_v = float(np.sqrt(scale ** 2 + w ** 2))
            vgrid = _lp_grid(centre, sd_v, "normal" if cm.lp_dist == "normal"
                             else cm.lp_dist, 0.0)
            dv = vgrid[1] - vgrid[0]
            kern = np.exp(-0.5 * ((vgrid - vgrid.mean()) / w) ** 2) / (
                w * np.sqrt(2 * np.pi))
            base = lp_standard_pdf((vgrid - centre) / scale, cm.lp_dist) / scale
            dens_v = np.convolve(base, kern, mode="same") * dv
            dens_v = dens_v / float(np.trapz(dens_v, vgrid))
            out[f"oracle_{k}"] = _auc_from_density(vgrid, dens_v, _expit(vgrid))
        out[f"excess_{k}"] = out[f"oracle_{k}"] - out[f"level_{k}"]

    vals = [out[f"level_{k}"] for k in range(len(scales))
            if np.isfinite(out[f"level_{k}"])]
    prevs = [out[f"prev_{k}"] for k in range(len(scales))]
    if len(vals) >= 2:
        out["max_gap"] = float(max(vals) - min(vals))
        out["mean_auc"] = float(np.mean(vals))
    out["max_excess"] = float(max(out[f"excess_{k}"] for k in range(len(scales))))
    out["prev_min"] = float(min(prevs))
    out["prev_max"] = float(max(prevs))
    out["prev_ratio"] = float(max(prevs) / min(prevs)) if min(prevs) > 0 else (
        float("inf"))
    out["sd_ratio"] = cm.sd_ratio
    out["intercept"] = b0
    return out


def verify_case_mix(geom: Geometry, n_check: int = 400_000, seed: int = 11
                    ) -> Dict[str, float]:
    """Integrity check for a case-mix geometry -- the six-geometry case-mix study lacked one.

    :func:`verify_null` refuses case-mix geometries (their subgroups do *not*
    share one true AUROC, by design) and the test suite skipped them, so the
    family carrying the entire thesis was the only family never checked for
    smuggling in a real effect. The right assertion here is not equal AUROC; it
    is that **no unfairness is present**, which is two checkable claims:

    ``max_calibration_error``
        the deployed score is the true event probability in every level, so
        within-level mean predicted risk matches the observed event rate.
    ``max_excess_auc``
        the deployed score is *Bayes-optimal* in every level: the AUROC obtained
        by ranking on the true linear predictor is no higher than the AUROC of
        the deployed score. A geometry that quietly handicapped one subgroup's
        model would show up here as a positive excess.

    Both are returned rather than asserted, so the caller decides the tolerance.
    """
    from dataclasses import replace

    from recompute.comparators.core import auc_delong

    cm = geom.case_mix
    if cm is None:
        raise ValueError(f"{geom.name} is not a case-mix geometry")
    big = replace(geom, n=n_check)
    y, s, codes, aux = make_dataset(big, rep=0, seed=seed, return_aux=True)
    truth = true_subgroup_auc(geom)

    calib = 0.0
    excess = -np.inf
    prev_err = 0.0
    for k in np.unique(codes["p0"]):
        m = codes["p0"] == k
        calib = max(calib, abs(float(s[m].mean()) - float(y[m].mean())))
        prev_err = max(prev_err,
                       abs(float(y[m].mean()) - truth[f"prev_{k}"]))
        a_dep = auc_delong(y[m], s[m])[0]
        a_orc = auc_delong(y[m], aux["true_lp"][m])[0]
        excess = max(excess, float(a_orc - a_dep))
    return {
        "max_calibration_error": float(calib),
        "max_excess_auc": float(excess),
        "max_level_prevalence_error": float(prev_err),
    }


def make_dataset(geom: Geometry, rep: int, seed: int = 42,
                 return_aux: bool = False):
    """One simulated cohort under the null. Pure in ``(geom, rep, seed)``.

    Returns ``(y, s, codes_by_col)`` in exactly the form every comparator's
    ``decide`` / ``pvalue_only`` entry point expects. With ``return_aux`` a
    fourth element carries the quantities an *auditor* never sees but a
    verification test needs -- the true linear predictor and the true event
    probability -- so that Bayes-optimality can be checked rather than asserted.

    The seed word for the geometry is :func:`geometry_seed_word` (a ``crc32``
    digest), not ``hash``: see the module docstring. This function is a pure
    function of its arguments in the literal sense -- it carries no process
    state and no interpreter-startup state.
    """
    rng = np.random.default_rng([seed, rep, geometry_seed_word(geom.name)])

    if geom.case_mix is not None:
        y, s, codes, aux = _make_case_mix(geom, rng)
        return (y, s, codes, aux) if return_aux else (y, s, codes)

    y = (rng.random(geom.n) < geom.prevalence).astype(int)
    # Binormal scores with the requested cohort AUROC, then squashed to (0, 1)
    # so they look like the predicted probabilities the real path produces. The
    # squash is a single global monotone map and does not change any AUROC.
    mu = norm.ppf(geom.auc) * np.sqrt(2.0)
    latent = rng.normal(loc=mu * y, scale=1.0, size=geom.n)
    s = 1.0 / (1.0 + np.exp(-latent))

    codes_by_col: Dict[str, np.ndarray] = {}
    for c, props in enumerate(geom.partitions):
        codes_by_col[f"p{c}"] = _level_codes(geom.n, props, rng)

    if geom.is_composite:
        # Apply the per-level strictly increasing map. Only the FIRST partition
        # carries the transform; the others (if any) then see a score
        # distribution that varies across their own levels only through their
        # overlap with the first, which is itself random. Within every level of
        # every partition the map is monotone in s, so no true AUROC moves --
        # which would NOT hold if two partitions each carried their own map.
        col0 = "p0"
        expo = np.asarray(geom.monotone_exponents[0], dtype=float)
        s_new = apply_monotone(s, expo[codes_by_col[col0]], geom.transform)
        # The composite null's entire validity rests on the map being strictly
        # increasing *as evaluated in float64*. A transform that saturates would
        # tie distinct scores together and silently move the true AUROC, turning
        # the whole cell into a power calculation dressed up as a null. Checked
        # on every dataset rather than argued for once: ties are cheap to count
        # and this runs once per simulated dataset, not once per permutation.
        if np.unique(s_new).size != np.unique(s).size:
            raise AssertionError(
                f"{geom.name}: the {geom.transform!r} map introduced ties "
                f"({np.unique(s).size} distinct scores in, "
                f"{np.unique(s_new).size} out). The transform has saturated in "
                "float64 and the true subgroup AUROC is no longer preserved.")
        s = s_new

    return (y, s, codes_by_col, {}) if return_aux else (y, s, codes_by_col)


def _make_case_mix(geom: Geometry, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray,
                              Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """One cohort under the case-mix DGP: one shared model, unequal spread.

    Order of draws differs from the equal-AUROC path (the subgroup codes have to
    exist before the predictor can be drawn), which is why the two paths are
    separate functions rather than one with a branch in the middle.

    Draw order is chosen so that every geometry that existed before round 3
    consumes exactly the stream it consumed then: the second covariate is drawn
    only when ``unfair_w`` is on, and the ``normal`` family keeps the original
    ``rng.normal(loc=, scale=)`` call rather than a standardise-and-rescale.
    """
    cm = geom.case_mix
    assert cm is not None
    b0, d = case_mix_offsets(geom)

    codes_by_col: Dict[str, np.ndarray] = {}
    for c, props in enumerate(geom.partitions):
        codes_by_col[f"p{c}"] = _level_codes(geom.n, props, rng)

    g = codes_by_col["p0"]
    locs = np.asarray(cm.locs, dtype=float)[g] + np.asarray(d, dtype=float)[g]
    scales = np.asarray(cm.scales, dtype=float)[g]
    # The single shared linear predictor. One coefficient vector for everyone:
    # the only thing that varies across subgroups is the covariate distribution
    # (and, when equalize_prevalence is on, a subgroup intercept the model has).
    if cm.lp_dist == "normal":
        lp = b0 + rng.normal(loc=locs, scale=scales)
    else:
        lp = b0 + locs + scales * lp_standard_draw(rng, cm.lp_dist, geom.n)

    # The covariate the deployed model does not use. Only drawn when it is
    # actually needed, so that the fair geometries' random stream is untouched.
    if cm.has_unfair_coef:
        ws = np.asarray(cm.unfair_w, dtype=float)[g]
        true_lp = lp + ws * rng.standard_normal(geom.n)
    else:
        true_lp = lp

    p_true = _expit(true_lp)
    y = (rng.random(geom.n) < p_true).astype(int)

    # The deployed score. Without miscalibration it IS the model's implied
    # probability, expit(lp); with it, a strictly increasing per-level distortion
    # of that -- which leaves every subgroup's AUROC exactly unchanged while
    # genuinely breaking calibration for that subgroup.
    if cm.has_miscalibration:
        a = (np.asarray(cm.miscal_intercept, dtype=float)[g]
             if cm.miscal_intercept is not None else 0.0)
        lam = (np.asarray(cm.miscal_slope, dtype=float)[g]
               if cm.miscal_slope is not None else 1.0)
        if np.any(np.asarray(lam) <= 0):
            raise ValueError(f"{geom.name}: miscal_slope must be > 0, else the "
                             "deployed score is not a monotone distortion and "
                             "the subgroup AUROC is not preserved.")
        s = _expit(a + lam * lp)
    else:
        s = _expit(lp)

    aux = {"true_lp": true_lp, "deployed_lp": lp, "p_true": p_true}
    return y, s, codes_by_col, aux


def verify_null(geom: Geometry, n_check: int = 200_000, seed: int = 7
                ) -> Dict[str, float]:
    """Sanity check that every subgroup really does share one true AUROC.

    Draws one very large dataset from ``geom`` and reports both the raw max-min
    subgroup AUROC and the **studentized** version of it, ``max |AUC_i - AUC_j| /
    sqrt(Var_i + Var_j)``. The raw gap is not a usable check on its own: in the
    skewed and rare-outcome geometries the smallest level holds 3% of the rows,
    so even at n = 200,000 its AUROC carries a standard error of a percentage
    point or two and the raw gap has an irreducible floor. The studentized
    version divides that out, is asymptotically standard normal under the null
    whatever the geometry, and is therefore the quantity the test suite asserts
    on. This is the guard against a DGP that silently smuggles in a real effect
    and makes the whole Type I table meaningless.
    """
    from dataclasses import replace

    from recompute.comparators.core import auc_delong

    if geom.case_mix is not None:
        raise ValueError(
            f"{geom.name} is a case-mix geometry: its subgroups do NOT share one "
            "true AUROC, by design. Use true_subgroup_auc() instead.")
    big = replace(geom, n=n_check)
    y, s, codes_by_col = make_dataset(big, rep=0, seed=seed)
    gaps: Dict[str, float] = {}
    max_t = 0.0
    for col, codes in codes_by_col.items():
        est = [auc_delong(y[codes == k], s[codes == k])
               for k in np.unique(codes)]
        est = [(a, v) for a, v in est if np.isfinite(a) and np.isfinite(v)]
        if len(est) < 2:
            gaps[col] = float("nan")
            continue
        gaps[col] = float(max(a for a, _ in est) - min(a for a, _ in est))
        for i in range(len(est)):
            for j in range(i + 1, len(est)):
                denom = est[i][1] + est[j][1]
                if denom > 0:
                    max_t = max(max_t, abs(est[i][0] - est[j][0]) / np.sqrt(denom))
    out: Dict[str, float] = dict(gaps)
    out["max_gap"] = float(np.nanmax(list(gaps.values())))
    out["max_studentized"] = float(max_t)
    return out
