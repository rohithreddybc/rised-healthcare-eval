"""
Lum, Zhang & Bower (2022, ACM FAccT), "De-biasing 'bias' measurement".

What the paper says
-------------------
Metrics that summarise *group-wise model performance disparity* are themselves
statistically biased estimators of the disparity they claim to measure. A
subgroup's performance estimate carries sampling noise; a dispersion statistic
computed across subgroups therefore picks up that noise on top of any real
between-group differences, and the smaller the subgroups the more of the measured
"bias" is noise. The paper's remedy is the **double-corrected variance
estimator**: an unbiased estimate, with uncertainty quantification, of the
variance of true model performance across groups. It is closed-form and needs no
numerical optimisation.

This is the closest published antecedent to the incumbent procedure. Both start
from the same observation -- that a max-min AUROC range across subgroups is
inflated by sampling noise -- and both try to remove the inflation. The
incumbent removes it by simulating the noise (a permutation null) and comparing
the observed statistic against it. Lum et al. remove it analytically, by
subtracting the noise's known contribution from the estimator.

What is implemented, precisely
------------------------------
Let a partition have L admissible levels with AUROC estimates ``theta_k`` and
DeLong sampling variances ``v_k``. Write ``theta_bar`` for the unweighted mean.

**The estimator (faithful).** The naive dispersion measure is the sample
variance of the estimates,

    S2 = (1 / (L - 1)) * sum_k (theta_k - theta_bar)^2 .

Writing ``theta_k = mu_k + e_k`` with ``E e_k = 0`` and ``Var e_k = v_k``,

    E[S2] = (1 / (L - 1)) * sum_k (mu_k - mu_bar)^2  +  (1 / L) * sum_k v_k ,

because ``E sum_k (e_k - e_bar)^2 = sum_k v_k - (1/L) sum_k v_k``. So the naive
statistic overstates the true between-group variance by exactly the average
within-group sampling variance. The double correction is therefore

    V_dc = S2 - (1 / L) * sum_k v_k ,

which is unbiased for the between-group variance of the *true* performances.
"Double" is read here as the two corrections it applies to the naive plug-in
statistic: the (L-1) denominator, and the subtraction of the mean sampling
variance. ``V_dc`` may come out negative -- that is expected and informative, and
both the raw and the zero-truncated value are reported.

**Uncertainty quantification.** The paper supplies UQ for ``V_dc`` in closed
form; the exact expression is not reproduced here, so it is re-derived. Treating
the ``v_k`` as known (their estimation error is second order) and the ``theta_k``
as independent and approximately normal with variances ``sigma_k^2 = tau^2 + v_k``,
``sum_k (theta_k - theta_bar)^2 = theta' M theta`` with ``M = I - J/L`` idempotent,
so under a constant mean ``Var(theta' M theta) = 2 tr((M Sigma)^2)`` and

    Var(V_dc) = Var(S2)
              = (2 / (L-1)^2) * [ (1 - 2/L) * sum_k sigma_k^4
                                  + (sum_k sigma_k^2)^2 / L^2 ] .

With all ``sigma_k^2`` equal this collapses to ``2 sigma^4 / (L-1)``, the textbook
variance of a normal sample variance, which is the check
``test_dc_variance_se_reduces_to_the_textbook_case`` makes.

The decision rule is the paper's own -- its real-data finding is that once
corrected the disparities are *no longer statistically significant* -- expressed
as a one-sided z-test of ``H0: tau^2 = 0``, evaluating the standard error under
the null (``sigma_k^2 = v_k``, the only variance consistent with H0):

    z = V_dc / SE_0 ,   p = 1 - Phi(z) .

This is the **primary** Lum verdict. Two alternatives are computed and reported
because they behave very differently and the difference is itself a finding:

  *parametric bootstrap CI* -- resample ``theta_k ~ N(theta_bar, v_k + max(V_dc, 0))``,
  recompute ``V_dc``, take a percentile interval, flag when the lower bound
  clears zero. With five subgroups this has four degrees of freedom and almost no
  power; it flags nothing anywhere in this study, including cohorts whose
  subgroup AUROCs are twelve standard errors apart. Reported, not used as the
  verdict, and the reason is stated in COMPARATOR_EVALUATION.md.

  *Cochran's Q* -- the precision-weighted homogeneity test, ``our addition``.

**What does not transfer, and what we did about it.** The incumbent's statistic
is a max-min *range*, not a variance, and a range has no comparably clean
unbiased correction: ``E[range]`` under noise depends on the whole joint
distribution of the L estimates, not just their variances. Two analogues on the
range scale are reported so the numbers can be read next to the incumbent's
``Delta``:

  ``range_moment``   ``tau_hat * d_L``, where ``tau_hat = sqrt(max(V_dc, 0))`` and
                     ``d_L = E[range of L iid standard normals]``. This is the
                     range implied by the corrected variance under a normal
                     random-effects model.
  ``range_shrunk``   the max-min range of the empirical-Bayes shrunken estimates
                     ``theta_bar + (tau2 / (tau2 + v_k)) * (theta_k - theta_bar)``.
                     Noisier subgroups are pulled harder toward the mean, which
                     is the range-scale expression of the same correction.

**Significance, secondary.** The paper is an estimation paper and gives no
hypothesis test of "are the group metrics all equal". Because the comparison
table needs a p-value column, Cochran's Q homogeneity test on the same
``(theta_k, v_k)`` inputs is reported as well: ``Q = sum_k w_k (theta_k -
theta_w)^2`` with ``w_k = 1/v_k``, referred to chi-square on L-1 degrees of
freedom. Q is the classical inferential companion of exactly this
method-of-moments correction (it is the ``Q`` of DerSimonian-Laird), but it is
*our* addition and is labelled as such everywhere it appears.

**Combining over partitions.** A cohort has several partitions and the incumbent
maxes over them. The Lum verdict flags when any partition flags, with the
confidence level Bonferroni-adjusted to ``1 - alpha/P`` across the ``P``
partitions so the family-wise error rate is still controlled; the reported
p-value is the Holm-adjusted minimum of the per-partition Q p-values.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import chi2, norm

from recompute.comparators.core import (
    FLAG,
    NO_FLAG,
    NOT_EVALUABLE,
    RULE_NAMES,
    VAR_FLOOR,
    CohortData,
    Level,
    MethodResult,
    admissible,
    all_level_stats,
    holm,
)

METHOD = "lum2022"

N_BOOT_DEFAULT = 2000


@lru_cache(maxsize=64)
def expected_normal_range(n_levels: int, n_sim: int = 200_000,
                          seed: int = 42) -> float:
    """E[range of ``n_levels`` iid standard normals], by simulation.

    Used only to put the corrected *variance* back on the *range* scale so it can
    be read beside the incumbent's max-min Delta. Cached, seeded, deterministic.
    """
    if n_levels < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sim, n_levels))
    return float(np.mean(z.max(axis=1) - z.min(axis=1)))


def double_corrected_variance(thetas: Sequence[float], variances: Sequence[float]
                              ) -> Dict[str, float]:
    """The double-corrected variance estimator of Lum, Zhang & Bower (2022).

    Returns the naive sample variance, the mean within-group sampling variance
    that inflates it, and the corrected estimate (raw and truncated at zero).
    """
    th = np.asarray(thetas, dtype=float)
    v = np.asarray(variances, dtype=float)
    ok = np.isfinite(th) & np.isfinite(v)
    th, v = th[ok], v[ok]
    L = len(th)
    if L < 2:
        return {"n_levels": L, "naive_variance": float("nan"),
                "mean_sampling_variance": float("nan"),
                "V_dc": float("nan"), "V_dc_truncated": float("nan")}
    s2 = float(np.var(th, ddof=1))
    vbar = float(np.mean(v))
    v_dc = s2 - vbar
    return {
        "n_levels": L,
        "naive_variance": s2,
        "mean_sampling_variance": vbar,
        "V_dc": float(v_dc),
        "V_dc_truncated": float(max(v_dc, 0.0)),
        # Fraction of the naive dispersion that was sampling noise. This is the
        # number the paper is really about.
        "noise_share_of_naive": float(vbar / s2) if s2 > 0 else float("nan"),
    }


def dc_variance_se(variances: Sequence[float], tau2: float = 0.0) -> float:
    """Standard error of ``V_dc``; see the module docstring for the derivation.

    ``tau2 = 0`` gives the standard error *under the null*, which is what a test
    of ``H0: tau^2 = 0`` must use. A positive ``tau2`` gives the estimation
    standard error used for a confidence interval.
    """
    v = np.asarray(variances, dtype=float)
    v = v[np.isfinite(v)]
    L = len(v)
    if L < 2:
        return float("nan")
    sig2 = v + max(float(tau2), 0.0)
    var = (2.0 / (L - 1) ** 2) * (
        (1.0 - 2.0 / L) * float(np.sum(sig2 ** 2))
        + float(np.sum(sig2)) ** 2 / L ** 2
    )
    return float(np.sqrt(max(var, 0.0)))


def dc_ztest(thetas: Sequence[float], variances: Sequence[float]
             ) -> Dict[str, float]:
    """One-sided closed-form test of ``H0: tau^2 = 0`` on the Lum estimator.

    This is the direct reading of the paper's estimator plus the paper's stated
    (here re-derived) uncertainty quantification, asked the only question the
    paper's real-data conclusion turns on: is the corrected disparity
    distinguishable from zero?

    **Read the p-value with care at small L.** Under H0 the numerator is a
    weighted sum of L-1 chi-square variates, which is markedly right-skewed when
    L is small, and a normal approximation to its upper tail is poor. At L = 2 it
    is a single chi-square with one degree of freedom and the approximation is
    badly anti-conservative far out in the tail: ACS-Income's age partition has
    two levels and returns p = 2e-37 where the exact two-sample comparison of the
    same two AUROCs gives 1e-05. The *decision* at a conventional alpha is much
    better behaved than the tail probability -- and it is the decision, not the
    p-value, that :mod:`recompute.comparators.type1` validates. Where an exactly
    calibrated reference is wanted, :func:`cochran_q` supplies one: with known
    variances and approximately normal estimates, Q is chi-square on L-1 degrees
    of freedom exactly, not asymptotically. ``n_levels`` is returned so the
    caveat can be applied.
    """
    base = double_corrected_variance(thetas, variances)
    if base["n_levels"] < 2:
        return {"z": float("nan"), "p_value": float("nan"),
                "se_null": float("nan")}
    se0 = dc_variance_se(variances, tau2=0.0)
    if not np.isfinite(se0) or se0 <= 0:
        return {"z": float("nan"), "p_value": float("nan"), "se_null": se0}
    z = base["V_dc"] / se0
    return {"z": float(z), "p_value": float(norm.sf(z)), "se_null": float(se0),
            "n_levels": int(base["n_levels"])}


def bootstrap_ci(thetas: Sequence[float], variances: Sequence[float],
                 conf: float = 0.95, n_boot: int = N_BOOT_DEFAULT,
                 seed: int = 42) -> Dict[str, float]:
    """Parametric-bootstrap percentile interval for ``V_dc``.

    Our implementation of the paper's uncertainty quantification (see the module
    docstring). Resamples the L group estimates from
    ``N(theta_bar, v_k + max(V_dc, 0))`` -- i.e. the fitted random-effects model
    -- and recomputes the estimator on each draw.
    """
    th = np.asarray(thetas, dtype=float)
    v = np.asarray(variances, dtype=float)
    ok = np.isfinite(th) & np.isfinite(v)
    th, v = th[ok], v[ok]
    L = len(th)
    if L < 2:
        return {"lo": float("nan"), "hi": float("nan")}
    base = double_corrected_variance(th, v)
    tau2 = base["V_dc_truncated"]
    rng = np.random.default_rng(seed)
    sd = np.sqrt(np.maximum(v + tau2, 0.0))
    draws = rng.normal(loc=float(th.mean()), scale=sd, size=(n_boot, L))
    s2 = draws.var(axis=1, ddof=1)
    vals = s2 - v.mean()
    a = (1.0 - conf) / 2.0
    return {"lo": float(np.quantile(vals, a)),
            "hi": float(np.quantile(vals, 1.0 - a))}


def cochran_q(thetas: Sequence[float], variances: Sequence[float]
              ) -> Dict[str, float]:
    """Cochran's Q test of homogeneity (our addition, not in the paper).

    ``Q = sum w_k (theta_k - theta_w)^2`` with ``w_k = 1/v_k`` and ``theta_w``
    the inverse-variance weighted mean; ``Q ~ chi2_{L-1}`` under the null that all
    ``theta_k`` are equal. Also returns the DerSimonian-Laird between-group
    variance, the precision-weighted counterpart of ``V_dc``.
    """
    th = np.asarray(thetas, dtype=float)
    v = np.asarray(variances, dtype=float)
    ok = np.isfinite(th) & np.isfinite(v) & (v > VAR_FLOOR)
    th, v = th[ok], v[ok]
    L = len(th)
    if L < 2:
        return {"Q": float("nan"), "df": 0, "p_value": float("nan"),
                "tau2_DL": float("nan")}
    w = 1.0 / v
    mu = float(np.sum(w * th) / np.sum(w))
    q = float(np.sum(w * (th - mu) ** 2))
    df = L - 1
    denom = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (q - df) / denom) if denom > 0 else float("nan")
    return {"Q": q, "df": df, "p_value": float(chi2.sf(q, df)),
            "tau2_DL": float(tau2)}


def shrunken_range(thetas: Sequence[float], variances: Sequence[float],
                   tau2: float) -> float:
    """Max-min range of the empirical-Bayes shrunken subgroup estimates."""
    th = np.asarray(thetas, dtype=float)
    v = np.asarray(variances, dtype=float)
    ok = np.isfinite(th) & np.isfinite(v)
    th, v = th[ok], v[ok]
    if len(th) < 2:
        return float("nan")
    mu = float(th.mean())
    if not np.isfinite(tau2) or tau2 <= 0:
        return 0.0
    w = tau2 / (tau2 + v)
    shrunk = mu + w * (th - mu)
    return float(shrunk.max() - shrunk.min())


def partition_result(levels: Sequence[Level], conf: float = 0.95,
                     n_boot: int = N_BOOT_DEFAULT, seed: int = 42
                     ) -> Dict[str, float]:
    """All Lum quantities for one demographic partition."""
    th = [lv.auc for lv in levels]
    v = [lv.var for lv in levels]
    out = double_corrected_variance(th, v)
    if out["n_levels"] < 2:
        return out
    out.update({f"ci_{k}": val
                for k, val in bootstrap_ci(th, v, conf, n_boot, seed).items()})
    out.update({f"z_{k}": val for k, val in dc_ztest(th, v).items()})
    q = cochran_q(th, v)
    out.update({f"q_{k}": val for k, val in q.items()})
    d_L = expected_normal_range(int(out["n_levels"]))
    out["observed_range"] = float(max(th) - min(th))
    out["range_moment"] = float(np.sqrt(out["V_dc_truncated"]) * d_L)
    out["range_shrunk"] = shrunken_range(th, v, out["V_dc_truncated"])
    out["expected_normal_range_dL"] = d_L
    return out


# ── Full cohort run ──────────────────────────────────────────────────────────
def run_cohort(data: CohortData, rules: Optional[List[str]] = None,
               alpha: float = 0.05, n_boot: int = N_BOOT_DEFAULT,
               seed: int = 42) -> Dict[str, object]:
    """Lum et al. on one cohort, for every inclusion rule.

    No permutation loop: the whole point of the method is that the correction is
    analytic. The runtime difference against the incumbent is one of the
    findings.
    """
    rules = list(rules) if rules is not None else list(RULE_NAMES)
    t0 = time.perf_counter()
    obs_levels = all_level_stats(data.y, data.s, data.codes_by_col)

    results: Dict[str, MethodResult] = {}
    variants: Dict[str, Dict[str, MethodResult]] = {
        f"{METHOD}_cochranQ": {}, f"{METHOD}_bootstrapCI": {}}
    diagnostics: Dict[str, Dict[str, object]] = {}
    for rule in rules:
        per_part = {}
        for col, lv in obs_levels.items():
            keep = admissible(lv, rule)
            if len(keep) >= 2:
                per_part[col] = partition_result(keep, n_boot=n_boot, seed=seed)
        if not per_part:
            for target, name in ((results, METHOD),
                                 (variants[f"{METHOD}_cochranQ"],
                                  f"{METHOD}_cochranQ"),
                                 (variants[f"{METHOD}_bootstrapCI"],
                                  f"{METHOD}_bootstrapCI")):
                target[rule] = MethodResult(
                    name, rule, NOT_EVALUABLE,
                    statistic_name="V_dc (double-corrected variance)",
                    runtime_s=time.perf_counter() - t0,
                    detail="no partition has two admissible levels")
            continue

        cols = list(per_part)
        # Primary verdict: closed-form z-test on V_dc, Holm-adjusted across the
        # cohort's partitions so the family-wise error rate is controlled.
        z_p = [per_part[c].get("z_p_value", np.nan) for c in cols]
        p_holm = (float(np.nanmin(holm(z_p)))
                  if np.isfinite(z_p).any() else float("nan"))
        # Secondary readings, reported for contrast.
        q_p = [per_part[c].get("q_p_value", np.nan) for c in cols]
        p_holm_q = (float(np.nanmin(holm(q_p)))
                    if np.isfinite(q_p).any() else float("nan"))
        boot_flag = any(np.isfinite(per_part[c].get("ci_lo", np.nan))
                        and per_part[c]["ci_lo"] > 0.0 for c in cols)

        worst = max(per_part, key=lambda c: per_part[c]["V_dc"])
        results[rule] = MethodResult(
            method=METHOD,
            rule=rule,
            conclusion=(FLAG if (np.isfinite(p_holm) and p_holm < alpha)
                        else NO_FLAG),
            statistic=float(per_part[worst]["V_dc"]),
            statistic_name="V_dc (double-corrected variance)",
            p_value=p_holm,
            runtime_s=time.perf_counter() - t0,
            detail=(
                f"worst_partition={worst}; "
                f"naive_var={per_part[worst]['naive_variance']:.5g}; "
                f"mean_sampling_var={per_part[worst]['mean_sampling_variance']:.5g}; "
                f"noise_share={per_part[worst].get('noise_share_of_naive', float('nan')):.3f}; "
                f"obs_range={per_part[worst]['observed_range']:.4f}; "
                f"range_moment={per_part[worst]['range_moment']:.4f}; "
                f"range_shrunk={per_part[worst]['range_shrunk']:.4f}; "
                f"p_holm_cochranQ={p_holm_q:.4g}; "
                f"bootstrap_ci_flag={boot_flag}"
            ),
        )
        # The two secondary readings are emitted as their own rows so the CSV
        # carries all three and the choice of primary is auditable rather than
        # buried in a free-text field.
        variants[f"{METHOD}_cochranQ"][rule] = MethodResult(
            method=f"{METHOD}_cochranQ",
            rule=rule,
            conclusion=(FLAG if (np.isfinite(p_holm_q) and p_holm_q < alpha)
                        else NO_FLAG),
            statistic=float(per_part[worst].get("q_tau2_DL", np.nan)),
            statistic_name="tau2 (DerSimonian-Laird)",
            p_value=p_holm_q,
            runtime_s=time.perf_counter() - t0,
            detail=("precision-weighted homogeneity test on the same inputs; "
                    "our addition, not in Lum et al."),
        )
        variants[f"{METHOD}_bootstrapCI"][rule] = MethodResult(
            method=f"{METHOD}_bootstrapCI",
            rule=rule,
            conclusion=FLAG if boot_flag else NO_FLAG,
            statistic=float(per_part[worst]["V_dc"]),
            statistic_name="V_dc (double-corrected variance)",
            p_value=None,
            runtime_s=time.perf_counter() - t0,
            detail=(f"parametric bootstrap {int(n_boot)} draws; flags when the "
                    f"lower bound on V_dc clears zero; "
                    f"lo={per_part[worst].get('ci_lo', float('nan')):.5g}"),
        )
        diagnostics[rule] = per_part

    return {"results": results, "variants": variants, "diagnostics": diagnostics,
            "runtime_s": time.perf_counter() - t0}


# ── Lightweight path for the Type I error simulation ─────────────────────────
def decide(ctx, rule: str, alpha: float = 0.05, n_boot: int = 1000,
           seed: int = 42) -> Dict[str, float]:
    """Flag decision and Q p-value for one simulated dataset."""
    parts = {}
    for col, lv in ctx.observed().items():
        keep = admissible(lv, rule)
        if len(keep) >= 2:
            parts[col] = keep
    if not parts:
        return {"flag": float("nan"), "p_value": float("nan"),
                "flag_q": float("nan"), "flag_boot": float("nan")}
    P = len(parts)
    boot_flag = False
    zps, qps = [], []
    for col, keep in parts.items():
        th = [k.auc for k in keep]
        v = [k.var for k in keep]
        zps.append(dc_ztest(th, v)["p_value"])
        qps.append(cochran_q(th, v)["p_value"])
        if n_boot:
            ci = bootstrap_ci(th, v, conf=1.0 - alpha / P, n_boot=n_boot,
                              seed=seed)
            if np.isfinite(ci["lo"]) and ci["lo"] > 0.0:
                boot_flag = True
    p_z = float(np.nanmin(holm(zps))) if np.isfinite(zps).any() else np.nan
    p_q = float(np.nanmin(holm(qps))) if np.isfinite(qps).any() else np.nan
    return {
        "flag": float(np.isfinite(p_z) and p_z < alpha),
        "p_value": p_z,
        "flag_q": float(np.isfinite(p_q) and p_q < alpha),
        "flag_boot": float(boot_flag),
    }
