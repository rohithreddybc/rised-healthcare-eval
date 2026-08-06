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
increasing map ``s -> s**a_g`` on (0, 1), applied identically to that subgroup's
positives and negatives. AUROC is rank-based within a subgroup, so a strictly
monotone within-subgroup transform leaves every subgroup's true AUROC *exactly*
unchanged at its common value -- while the subgroups' score distributions now
differ substantially. This is precisely the composite null DiCiccio et al.
studentize against, and it is the only place where the studentization can earn
its keep. Under it, exchangeability fails and an unstudentized permutation test
has no validity guarantee.

Everything is seeded; :func:`make_dataset` is a pure function of ``(geometry,
replicate index, seed)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm


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
    #: Per-level exponents of the strictly monotone score map, per partition.
    #: ``None`` means the identity map, i.e. the simple (exchangeable) null.
    monotone_exponents: Optional[Tuple[Tuple[float, ...], ...]] = None
    description: str = ""

    @property
    def is_composite(self) -> bool:
        return self.monotone_exponents is not None


def _equal(k: int) -> Tuple[float, ...]:
    return tuple([1.0 / k] * k)


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

GEOMETRY_BY_NAME = {g.name: g for g in GEOMETRIES}


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


def make_dataset(geom: Geometry, rep: int, seed: int = 42
                 ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """One simulated cohort under the null. Pure in ``(geom, rep, seed)``.

    Returns ``(y, s, codes_by_col)`` in exactly the form every comparator's
    ``decide`` / ``pvalue_only`` entry point expects.
    """
    rng = np.random.default_rng([seed, rep, abs(hash(geom.name)) % (2 ** 31)])

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
        # Apply the per-level strictly increasing map s -> s**a. Only the FIRST
        # partition carries the transform; the others (if any) then see a score
        # distribution that varies across their own levels only through their
        # overlap with the first, which is itself random. Within every level of
        # every partition the map is monotone in s, so no true AUROC moves.
        col0 = f"p0"
        expo = np.asarray(geom.monotone_exponents[0], dtype=float)
        s = np.power(s, expo[codes_by_col[col0]])

    return y, s, codes_by_col


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
    from recompute.comparators.core import auc_delong

    big = Geometry(geom.name, n_check, geom.prevalence, geom.partitions,
                   geom.auc, geom.monotone_exponents, geom.description)
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
