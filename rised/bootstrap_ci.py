"""
Bias-corrected and accelerated (BCa) bootstrap confidence intervals,
plus Holm-Bonferroni family-wise error correction for the RISED test family.

BCa references:
  Efron, B. (1987). Better bootstrap confidence intervals. JASA 82(397):171-185.
  DiCiccio, T. & Efron, B. (1996). Bootstrap confidence intervals. Stat Sci 11(3):189-228.

For metrics bounded on [0,1] near boundaries (JSS, AUC parity gap, max TFR), the
percentile bootstrap is known to undercover; BCa applies bias correction (z0)
and acceleration (a) to produce intervals with correct coverage in finite samples.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy.stats import norm


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
    B = len(theta_boot)
    if B == 0:
        return (float("nan"), float("nan"))

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
) -> np.ndarray:
    """
    Compute jackknife (leave-one-out) replicates of a statistic.

    Parameters
    ----------
    statistic_fn : callable(idx_array) -> float
        Function that takes an array of indices and returns the statistic.
    n : int
        Sample size.
    indices : np.ndarray, optional
        Pre-computed full index array (default: np.arange(n)).

    Returns
    -------
    theta_jack : np.ndarray, shape (n,)
        Jackknife replicates.
    """
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
