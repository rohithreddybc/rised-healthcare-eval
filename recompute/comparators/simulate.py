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
    """

    locs: Tuple[float, ...]
    scales: Tuple[float, ...]


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

# ── Round-2 additions: a composite null thick enough to test the mechanism ────
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
        description=("3 levels, predictor SD 0.9/1.0/1.1: true AUROC "
                     "0.727/0.745/0.763, gap 0.036 -- mild case-mix "
                     "heterogeneity, BELOW the 0.05 convention")),
    Geometry(
        "casemix_moderate_3", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.7, 1.0, 1.4)),
        description=("3 levels, predictor SD 0.7/1.0/1.4: true AUROC "
                     "0.684/0.745/0.807, gap 0.123 -- moderate case-mix "
                     "heterogeneity, the realistic clinical case")),
    Geometry(
        "casemix_strong_4", n=2000, prevalence=0.20, partitions=(_equal(4),),
        case_mix=CaseMix(locs=(0.0,) * 4, scales=(0.6, 0.9, 1.3, 1.9)),
        description=("4 levels, predictor SD 0.6 to 1.9: true AUROC 0.661 to "
                     "0.860, gap 0.199 -- strong case-mix heterogeneity")),
    Geometry(
        "casemix_location_3", n=2000, prevalence=0.20, partitions=(_equal(3),),
        case_mix=CaseMix(locs=(-2.0, 0.0, 2.0), scales=(1.0, 1.0, 1.0)),
        description=("3 levels differing in predictor LOCATION only, equal "
                     "spread: true AUROC gap 0.017 -- separates case-mix "
                     "location from case-mix spread")),
    Geometry(
        "casemix_moderate_3_n10000", n=10_000, prevalence=0.20,
        partitions=(_equal(3),),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.7, 1.0, 1.4)),
        description=("moderate case-mix at n=10,000 -- the same true gap of "
                     "0.123 with five times the data")),
    Geometry(
        "casemix_moderate_3part", n=2000, prevalence=0.20,
        partitions=(_equal(3), _equal(2), (0.5, 0.3, 0.2)),
        case_mix=CaseMix(locs=(0.0, 0.0, 0.0), scales=(0.7, 1.0, 1.4)),
        description=("moderate case-mix on the first of 3 partitions -- the "
                     "other two are pure noise columns")),
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


def case_mix_intercept(geom: Geometry) -> float:
    """Intercept ``b0`` of the shared model that hits the target prevalence.

    Solved deterministically by quadrature on the level-size-weighted mixture of
    the linear predictor, so the same geometry always yields the same intercept
    and the prevalence column of the results table is exact rather than nominal.
    """
    from scipy.optimize import brentq

    cm = geom.case_mix
    assert cm is not None
    w = _mixture_weights(geom)
    locs = np.asarray(cm.locs, dtype=float)
    scales = np.asarray(cm.scales, dtype=float)

    def mean_p(b0: float) -> float:
        # E[expit(b0 + loc_g + scale_g Z)] averaged over levels with weights w.
        lp = b0 + locs[:, None] + scales[:, None] * _GH_Z[None, :]
        per_level = np.trapz(_expit(lp) * _GH_W[None, :], _GH_Z, axis=1)
        return float(np.dot(w, per_level))

    return float(brentq(lambda b: mean_p(b) - geom.prevalence, -30.0, 30.0,
                        xtol=1e-12, rtol=1e-14))


def true_subgroup_auc(geom: Geometry) -> Dict[str, float]:
    """Exact true AUROC of every level of partition 0, by quadrature.

    For a case-mix geometry the score is a strictly increasing function of the
    linear predictor ``L ~ N(loc_g, scale_g^2)``, so within level ``g``

        AUROC_g = P(L_case > L_control) = int f_pos(t) F_neg(t) dt

    with ``f_pos(t) prop. phi_g(t) p(t)`` and ``f_neg(t) prop. phi_g(t)(1-p(t))``,
    ``p(t) = expit(t)``. This is computed on a fine grid rather than simulated,
    so the reported unequal-AUROC magnitudes carry no Monte-Carlo error.
    """
    if geom.case_mix is None:
        return {}
    cm = geom.case_mix
    b0 = case_mix_intercept(geom)
    out: Dict[str, float] = {}
    t = np.linspace(-40.0, 40.0, 40_001)
    for k, (loc, scale) in enumerate(zip(cm.locs, cm.scales)):
        phi = np.exp(-0.5 * ((t - loc) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
        p = _expit(b0 + t)
        f_pos = phi * p
        f_neg = phi * (1.0 - p)
        m_pos = np.trapz(f_pos, t)
        m_neg = np.trapz(f_neg, t)
        if m_pos <= 0 or m_neg <= 0:
            out[f"level_{k}"] = float("nan")
            continue
        f_pos = f_pos / m_pos
        f_neg = f_neg / m_neg
        # F_neg evaluated at the same grid, midpoint-corrected so the coincident
        # mass at t contributes one half (there is none for continuous L, but the
        # correction removes the O(dt) bias of a plain cumulative sum).
        dt = t[1] - t[0]
        cdf_neg = np.cumsum(f_neg) * dt - 0.5 * f_neg * dt
        out[f"level_{k}"] = float(np.trapz(f_pos * cdf_neg, t))
    vals = [v for v in out.values() if np.isfinite(v)]
    if len(vals) >= 2:
        out["max_gap"] = float(max(vals) - min(vals))
        out["mean_auc"] = float(np.mean(vals))
    out["intercept"] = b0
    return out


def make_dataset(geom: Geometry, rep: int, seed: int = 42
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """One simulated cohort under the null. Pure in ``(geom, rep, seed)``.

    Returns ``(y, s, codes_by_col)`` in exactly the form every comparator's
    ``decide`` / ``pvalue_only`` entry point expects.

    The seed word for the geometry is :func:`geometry_seed_word` (a ``crc32``
    digest), not ``hash``: see the module docstring. This function is a pure
    function of its arguments in the literal sense -- it carries no process
    state and no interpreter-startup state.
    """
    rng = np.random.default_rng([seed, rep, geometry_seed_word(geom.name)])

    if geom.case_mix is not None:
        return _make_case_mix(geom, rng)

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

    return y, s, codes_by_col


def _make_case_mix(geom: Geometry, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """One cohort under the case-mix null: one shared model, unequal spread.

    Order of draws differs from the equal-AUROC path (the subgroup codes have to
    exist before the predictor can be drawn), which is why the two paths are
    separate functions rather than one with a branch in the middle.
    """
    cm = geom.case_mix
    assert cm is not None
    b0 = case_mix_intercept(geom)

    codes_by_col: Dict[str, np.ndarray] = {}
    for c, props in enumerate(geom.partitions):
        codes_by_col[f"p{c}"] = _level_codes(geom.n, props, rng)

    g = codes_by_col["p0"]
    locs = np.asarray(cm.locs, dtype=float)[g]
    scales = np.asarray(cm.scales, dtype=float)[g]
    # The single shared linear predictor. One coefficient vector for everyone:
    # the only thing that varies across subgroups is the covariate distribution.
    lp = b0 + rng.normal(loc=locs, scale=scales)
    p = _expit(lp)
    y = (rng.random(geom.n) < p).astype(int)
    # The score IS the true probability. The model is not merely fair, it is
    # correct and perfectly calibrated in every subgroup.
    return y, p, codes_by_col


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
