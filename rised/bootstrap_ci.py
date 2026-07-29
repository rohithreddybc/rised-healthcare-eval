"""
Bias-corrected and accelerated (BCa) bootstrap confidence intervals,
plus Holm-Bonferroni family-wise error correction for the RISED test family.

BCa references:
  Efron, B. (1987). Better bootstrap confidence intervals. JASA 82(397):171-185.
  DiCiccio, T. & Efron, B. (1996). Bootstrap confidence intervals. Stat Sci 11(3):189-228.

For metrics bounded on [0,1] near boundaries (JSS, AUC parity gap, max TFR), the
percentile bootstrap is known to undercover; BCa applies bias correction (z0)
and acceleration (a) to produce intervals with correct coverage in finite samples.

Clustered (grouped) resampling
------------------------------
Row-level resampling assumes rows are independent. That assumption fails when a
cohort contains repeated measurements on the same unit (e.g. multiple hospital
encounters per patient), and row-level intervals are then anti-conservative.
:class:`ResamplingPlan` implements the cluster bootstrap of Field & Welsh (2007):
unique group identifiers are resampled with replacement and *all* rows belonging
to a sampled group are taken. The matching jackknife leaves out one whole group
at a time, which is the delete-one-cluster jackknife required for a coherent BCa
acceleration constant under clustering.

  Field, C. A. & Welsh, A. H. (2007). Bootstrapping clustered data.
  JRSS-B 69(3):369-390.
"""

from __future__ import annotations

import warnings
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import norm

#: Below this many usable replicates the BCa tail quantiles are not estimable.
MIN_RELIABLE_BOOTSTRAP = 200


class ResamplingPlan:
    """Row-level or cluster-level resampling plan for the bootstrap/jackknife.

    Parameters
    ----------
    n : int
        Number of rows in the sample.
    groups : array-like of shape (n,), optional
        Cluster identifier per row. When ``None`` (the default) resampling is
        row-level and behaviour is identical to the pre-existing bootstrap.
        When supplied, the resampling unit becomes the unique group id.

    Notes
    -----
    Bootstrap replicates produced under clustering do **not** in general have
    length ``n``; they have the total length of the sampled clusters. Estimators
    must therefore accept an index array of arbitrary length.
    """

    def __init__(self, n: int, groups=None) -> None:
        self.n = int(n)
        if groups is None:
            self.groups = None
            self._members: Optional[List[np.ndarray]] = None
            return

        g = np.asarray(groups)
        if g.ndim != 1 or len(g) != self.n:
            raise ValueError(
                f"groups must be 1-D of length n={self.n}; got shape {g.shape}."
            )
        self.groups = g
        _, inverse = np.unique(g, return_inverse=True)
        n_groups = int(inverse.max()) + 1 if len(inverse) else 0
        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=n_groups)
        bounds = np.concatenate([[0], np.cumsum(counts)])
        self._members = [
            order[bounds[i]:bounds[i + 1]] for i in range(n_groups)
        ]

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def clustered(self) -> bool:
        """True when resampling is at the group level."""
        return self.groups is not None

    @property
    def n_units(self) -> int:
        """Number of independent resampling units (rows, or groups)."""
        return self.n if self._members is None else len(self._members)

    def describe(self) -> dict:
        """Machine-readable description for inclusion in result ``details``."""
        out = {"clustered": self.clustered, "n_rows": self.n, "n_units": self.n_units}
        if self._members is not None:
            sizes = np.array([len(m) for m in self._members], dtype=float)
            out["mean_rows_per_group"] = float(sizes.mean()) if len(sizes) else 0.0
            out["max_rows_per_group"] = int(sizes.max()) if len(sizes) else 0
        return out

    # ── resampling ───────────────────────────────────────────────────────────
    def bootstrap_index(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one bootstrap index array (row-level or cluster-level)."""
        if self._members is None:
            return rng.integers(0, self.n, size=self.n)
        n_g = len(self._members)
        if n_g == 0:
            return np.empty(0, dtype=int)
        picks = rng.integers(0, n_g, size=n_g)
        return np.concatenate([self._members[i] for i in picks])

    def jackknife_index_iter(self) -> Iterator[np.ndarray]:
        """Yield leave-one-unit-out index arrays.

        Leaves out one *row* when unclustered, one whole *group* when clustered.
        """
        if self._members is None:
            full = np.arange(self.n)
            for i in range(self.n):
                yield np.delete(full, i)
        else:
            n_g = len(self._members)
            for i in range(n_g):
                keep = [self._members[j] for j in range(n_g) if j != i]
                yield (
                    np.sort(np.concatenate(keep)) if keep else np.empty(0, dtype=int)
                )

    def full_index(self) -> np.ndarray:
        return np.arange(self.n)


def bootstrap_replicates(
    statistic_fn: Callable[[np.ndarray], float],
    plan: ResamplingPlan,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute ``n_bootstrap`` replicates of ``statistic_fn`` under ``plan``."""
    out = np.empty(int(n_bootstrap), dtype=float)
    for b in range(int(n_bootstrap)):
        out[b] = statistic_fn(plan.bootstrap_index(rng))
    return out


def jackknife_from_plan(
    statistic_fn: Callable[[np.ndarray], float],
    plan: ResamplingPlan,
) -> np.ndarray:
    """Delete-one-unit jackknife replicates of ``statistic_fn`` under ``plan``."""
    return np.array(
        [statistic_fn(idx) for idx in plan.jackknife_index_iter()], dtype=float
    )


def bca_interval(
    theta_hat: float,
    theta_boot: np.ndarray,
    theta_jack: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Compute the BCa 100*(1-alpha)% confidence interval.

    Parameters
    ----------
    theta_hat : float
        Point estimate computed on the full sample.
    theta_boot : np.ndarray
        Bootstrap replicates of the statistic, shape (B,).
    theta_jack : np.ndarray
        Jackknife (leave-one-out) replicates of the statistic, shape (n,).
    alpha : float
        Significance level (default 0.05 -> 95% CI).

    Returns
    -------
    (lower, upper) : tuple of float
        BCa-corrected confidence interval bounds.
    """
    theta_boot = np.asarray(theta_boot, dtype=float)
    theta_jack = np.asarray(theta_jack, dtype=float)
    theta_boot = theta_boot[~np.isnan(theta_boot)]
    theta_jack = theta_jack[~np.isnan(theta_jack)]
    B = len(theta_boot)
    if B == 0 or np.isnan(theta_hat):
        return (float("nan"), float("nan"))
    if len(theta_jack) == 0:
        theta_jack = np.array([theta_hat], dtype=float)
    if B < MIN_RELIABLE_BOOTSTRAP:
        warnings.warn(
            f"BCa interval computed from only {B} usable bootstrap replicates. "
            f"The bias-correction z0 saturates at 1/(2B) and the tail quantiles "
            f"are not estimable at this B; use at least "
            f"{MIN_RELIABLE_BOOTSTRAP} (1000 for published figures).",
            UserWarning,
            stacklevel=2,
        )

    # Bias correction z0
    prop_below = float(np.mean(theta_boot < theta_hat))
    # Avoid Phi^-1(0) and Phi^-1(1)
    prop_below = min(max(prop_below, 1.0 / (2 * B)), 1.0 - 1.0 / (2 * B))
    z0 = norm.ppf(prop_below)

    # Acceleration a from jackknife
    jack_mean = float(np.mean(theta_jack))
    diffs = jack_mean - theta_jack
    denom = 6.0 * (np.sum(diffs ** 2) ** 1.5)
    a = float(np.sum(diffs ** 3) / denom) if denom > 0 else 0.0

    # Adjusted percentiles
    z_alpha_lo = norm.ppf(alpha / 2.0)
    z_alpha_hi = norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        num = z0 + z
        den = 1.0 - a * num
        if den == 0:
            return float(norm.cdf(z0 + num))
        return float(norm.cdf(z0 + num / den))

    p_lo = _adjust(z_alpha_lo)
    p_hi = _adjust(z_alpha_hi)
    p_lo = min(max(p_lo, 1e-6), 1.0 - 1e-6)
    p_hi = min(max(p_hi, 1e-6), 1.0 - 1e-6)

    lower = float(np.quantile(theta_boot, p_lo))
    upper = float(np.quantile(theta_boot, p_hi))
    return (lower, upper)


def jackknife_replicates(
    statistic_fn: Callable[[np.ndarray], float],
    n: int,
    indices: Optional[np.ndarray] = None,
    groups=None,
) -> np.ndarray:
    """
    Compute jackknife replicates of a statistic.

    Leaves out one row at a time by default; when ``groups`` is supplied, leaves
    out one whole group at a time (delete-one-cluster jackknife).

    Parameters
    ----------
    statistic_fn : callable(idx_array) -> float
        Function that takes an array of indices and returns the statistic.
    n : int
        Sample size.
    indices : np.ndarray, optional
        Pre-computed full index array (default: np.arange(n)). Ignored when
        ``groups`` is supplied.
    groups : array-like of shape (n,), optional
        Cluster identifier per row. Default ``None`` reproduces the original
        leave-one-row-out behaviour exactly.

    Returns
    -------
    theta_jack : np.ndarray
        Jackknife replicates, one per resampling unit.
    """
    if groups is not None:
        return jackknife_from_plan(statistic_fn, ResamplingPlan(n, groups))
    if indices is None:
        indices = np.arange(n)
    out = np.empty(n, dtype=float)
    for i in range(n):
        loo = np.delete(indices, i)
        out[i] = statistic_fn(loo)
    return out


def holm_bonferroni(p_values, alpha: float = 0.05):
    """
    Apply Holm-Bonferroni step-down family-wise error correction.

    Parameters
    ----------
    p_values : array-like
        Per-test p-values.
    alpha : float
        Family-wise significance level (default 0.05).

    Returns
    -------
    rejected : np.ndarray of bool
        Whether each test rejects H0 after correction.
    adjusted_alpha : np.ndarray of float
        Per-test alpha thresholds in the original input order.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    rejected = np.zeros(m, dtype=bool)
    adjusted_alpha = np.empty(m, dtype=float)
    cutoff_reached = False
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        adjusted_alpha[idx] = threshold
        if not cutoff_reached and p[idx] <= threshold:
            rejected[idx] = True
        else:
            cutoff_reached = True
    return rejected, adjusted_alpha


def empirical_coverage(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    X: np.ndarray,
    y: np.ndarray,
    bca_fn: Callable[[np.ndarray, np.ndarray], Tuple[float, float]],
    n_resplits: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """Estimate empirical coverage of a bootstrap CI procedure.

    Repeats `n_resplits` independent train/test splits, recomputes the
    statistic on each test split, and reports the proportion of returned
    confidence intervals that cover the statistic computed on a held-out
    reference resampling. Intended as a sanity check for BCa coverage on
    bounded statistics such as PSS and max TFR, as referenced in
    Appendix A of the RISED paper.

    Parameters
    ----------
    statistic_fn : (X_test, y_test) -> float
        Point estimator under the resplit.
    bca_fn : (X_test, y_test) -> (lo, hi)
        CI procedure to be audited (must be deterministic given seed).
    n_resplits : int, default 100
    test_size : float, default 0.2
    random_state : int, default 42

    Returns
    -------
    coverage : float
        Empirical proportion of CIs containing the reference statistic.
    """
    from sklearn.model_selection import train_test_split

    rs = np.random.RandomState(random_state)
    n = X.shape[0]
    # Reference statistic: computed on full sample
    theta_ref = statistic_fn(X, y)

    covered = 0
    for _ in range(n_resplits):
        seed = int(rs.randint(0, 2 ** 31 - 1))
        _, X_te, _, y_te = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )
        lo, hi = bca_fn(X_te, y_te)
        if lo <= theta_ref <= hi:
            covered += 1
    return covered / n_resplits
