"""
The empirical anchor: what case-mix heterogeneity actually looks like in the ten
fitted models, and how much of their observed subgroup AUROC gaps it explains.

Two questions, two outputs
--------------------------
**How much case-mix heterogeneity is there really?**
(``recompute/results/cohort_sd_ratios.csv``.) The simulation's whole geometry is
one number: the ratio of the largest to the smallest per-level standard deviation
of the linear predictor. Round 2 set it to 1.22, 2.00 and 3.17 and called 2.00
"a moderate and clinically ordinary amount of case-mix heterogeneity", with no
citation and no measurement. The authors have ten fitted models and ten real
cohorts, so the quantity is directly measurable: for every demographic partition
of every cohort, take ``logit`` of the fitted model's predicted probability --
which *is* the linear predictor, exactly, for the logistic models, and is the
raw margin for the gradient-boosted ones -- and report its standard deviation in
each level. If real cohorts sit near 1.2 rather than 2.0, the headline geometry
is not the realistic case and the headline number is an extrapolation.

**How much of each observed gap is case mix?**
(``recompute/results/model_based_concordance.csv``.) The manuscript cites van
Klaveren et al. (2016) model-based concordance and explicitly does not evaluate
it. Model-based concordance is the c-statistic a model's *own predictions* imply
for the case mix it was applied to: treat each predicted risk as the truth and
average pairwise concordance. Computed per subgroup it answers exactly the
question the simulation is a proxy for -- of the observed max-min subgroup AUROC
gap, how much would be there anyway with a model that discriminates equally well
everywhere, purely because the subgroups' risk distributions differ?

What is reported, and why the primary summary is a difference
-------------------------------------------------------------
The headline quantity is the **residual gap**,

    residual_gap = observed_gap - mbc_gap_aligned,

the part of the observed max-min subgroup AUROC gap that case mix does *not*
explain, in AUROC units. It is a difference, so it is defined and stable for
every partition. The attributable *fraction*
``mbc_gap_aligned / observed_gap`` is also reported, but only where the
denominator can carry it: observed gaps in these ten cohorts run from 0.004 to
0.33, and a ratio whose denominator is indistinguishable from zero is arithmetic
rather than evidence. ``fraction_reportable`` is true only when the bootstrap
2.5th percentile of the gap is above zero.

The pair of levels that produces the observed maximum and minimum is selected
**once, on the observed data**, and held fixed inside every bootstrap replicate,
so the point estimate and the replicates target the same estimand. The interval
is therefore conditional on that selection; a sensitivity column re-selects the
extreme pair per replicate, and its denominator is winner's-curse inflated.

Caveats that must travel with the number
----------------------------------------
* Model-based concordance assumes the predictions are calibrated in the subgroup
  it is computed on. They frequently are not. A variant computed after a
  within-level Cox recalibration (``y ~ a + b logit(s)``) is also reported --
  **but it is close to a tautology and must not be used as the headline.** That
  recalibration is a monotone transform of the scores, so it leaves the level's
  observed AUROC exactly unchanged while forcing MBC onto it; the resulting
  "attributable fraction" is driven to 1 by the recalibration step rather than by
  case mix. ``max_abs_mbc_recalibrated_minus_obs_auc`` quantifies the collapse.
  See ``MBC_FIX.md``.
* Predictions are in-sample for the *case mix* (the same held-out rows the AUROC
  is computed on), so the two quantities are not independent.
* The bootstrap resamples rows within (level, outcome). For Diabetes 130 a
  patient can contribute several rows to the test set even under the group split,
  so its interval is optimistic.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import expit

from recompute.comparators.core import (
    COHORT_LABELS,
    COHORT_ORDER,
    REPO,
    auc_delong,
    load_cohort,
)
from recompute.comparators.round3_sim import (
    cox_calibration_test,
    model_based_concordance,
)
from recompute.null_reference import INCLUSION_RULES, _rule_admits

RESULTS = REPO / "recompute" / "results"
SD_CSV = RESULTS / "cohort_sd_ratios.csv"
MBC_CSV = RESULTS / "model_based_concordance.csv"

#: The published default inclusion rule; ``ev10`` is reported alongside because
#: the cohort analysis turns on it.
RULES = ("m30", "ev10")

_EPS = 1e-12


def linear_predictor(s: np.ndarray) -> np.ndarray:
    """``logit(s)``: the model's linear predictor, on the scale the DGP uses.

    For the logistic-regression cohorts this is exactly the fitted linear
    predictor. For the XGBoost cohorts it is the raw margin, which is the same
    object -- the quantity the model adds up and squashes -- and is the correct
    analogue of the simulation's ``lp``, whose per-level SD is the geometry.
    """
    s = np.clip(np.asarray(s, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(s) - np.log1p(-s)


def _cox_fit(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Newton-Raphson fit of ``y ~ a + b x``; identity ``(0, 1)`` on failure.

    Replaces a ``scipy.optimize.minimize`` call that had to be re-run inside every
    bootstrap replicate. Two parameters converge in a handful of Newton steps, and
    step-halving plus a coefficient bound keep quasi-separated levels -- which do
    occur in the small race strata -- from running off to infinity instead of
    silently returning ``success=False``.
    """
    n = x.size
    X = np.column_stack([np.ones(n), x])
    beta = np.array([0.0, 1.0])

    def nll(b_):
        eta = X @ b_
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    cur = nll(beta)
    for _ in range(60):
        eta = X @ beta
        mu = expit(eta)
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        grad = X.T @ (y - mu)
        hess = (X * w[:, None]).T @ X
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return np.array([0.0, 1.0])
        # Step-halving: never accept an uphill move.
        t = 1.0
        for _ in range(30):
            cand = beta + t * step
            val = nll(cand)
            if val <= cur:
                break
            t *= 0.5
        else:
            break
        beta, prev, cur = cand, cur, val
        if np.max(np.abs(t * step)) < 1e-10 or abs(prev - cur) < 1e-12:
            break
    if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e3:
        return np.array([0.0, 1.0])
    return beta


def _recalibrated(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Predictions after a within-level Cox recalibration ``a + b logit(s)``.

    Note that this is a *monotone* transform of ``s`` whenever ``b > 0``, so it
    leaves the level's observed AUROC exactly unchanged while forcing the
    predictions to be calibrated in that level. See ``MBC_FIX.md`` -- that is
    precisely why the recalibrated attributable fraction is near-tautological.
    """
    x = linear_predictor(s)
    y = np.asarray(y, dtype=float)
    a, b = _cox_fit(y, x)
    return expit(a + b * x)


def _levels(y: np.ndarray, s: np.ndarray, codes: np.ndarray, rule: str):
    """Estimable levels of one partition that the inclusion rule admits."""
    r = INCLUSION_RULES[rule]
    out = []
    for k in np.unique(codes):
        m = codes == k
        n = int(m.sum())
        n_pos = int(y[m].sum())
        n_neg = n - n_pos
        if n_pos < 2 or n_neg < 2:
            continue
        if not _rule_admits(r, n, n_pos, n_neg):
            continue
        out.append((int(k), m, n, n_pos, n_neg))
    return out


# ── (1) the empirical SD ratios ──────────────────────────────────────────────
def sd_ratio_rows(cohort) -> List[Dict]:
    lp = linear_predictor(cohort.s)
    rows: List[Dict] = []
    for rule in RULES:
        for col, codes in cohort.codes_by_col.items():
            lv = _levels(cohort.y, cohort.s, codes, rule)
            if len(lv) < 2:
                continue
            sds = {k: float(np.std(lp[m], ddof=1)) for k, m, *_ in lv}
            iqrs = {k: float(np.subtract(*np.percentile(lp[m], [75, 25])))
                    for k, m, *_ in lv}
            ratio = max(sds.values()) / min(sds.values()) if min(
                sds.values()) > 0 else float("inf")
            iqr_ratio = (max(iqrs.values()) / min(iqrs.values())
                         if min(iqrs.values()) > 0 else float("inf"))
            aucs = {}
            for k, m, n, n_pos, n_neg in lv:
                aucs[k] = auc_delong(cohort.y[m], cohort.s[m])[0]
            for k, m, n, n_pos, n_neg in lv:
                rows.append({
                    "cohort": cohort.name,
                    "cohort_label": COHORT_LABELS.get(cohort.name, cohort.name),
                    "is_clinical": cohort.is_clinical,
                    "n_test": cohort.n_test,
                    "rule": rule,
                    "partition": col,
                    "n_levels_admissible": len(lv),
                    "level": k,
                    "level_n": n,
                    "level_n_pos": n_pos,
                    "level_prevalence": n_pos / n,
                    "level_lp_sd": sds[k],
                    "level_lp_iqr": iqrs[k],
                    "level_lp_mean": float(np.mean(lp[m])),
                    "level_auc": aucs[k],
                    # The partition-level quantity the simulation's geometry IS.
                    "partition_sd_ratio": ratio,
                    "partition_iqr_ratio": iqr_ratio,
                    "partition_min_lp_sd": min(sds.values()),
                    "partition_max_lp_sd": max(sds.values()),
                    "partition_observed_auc_gap": (
                        max(aucs.values()) - min(aucs.values())),
                    "partition_level_prevalence_ratio": (
                        max(n_pos / n for _, _, n, n_pos, _ in lv)
                        / min(n_pos / n for _, _, n, n_pos, _ in lv)),
                })
    return rows


# ── (2) model-based concordance ──────────────────────────────────────────────
def _partition_seed(seed: int, cohort_name: str, rule: str, col: str) -> int:
    """A seed pinned to the (cohort, rule, partition) cell.

    The previous code threaded one generator through the whole cohort loop, so a
    partition's interval depended on how many partitions had been drawn before
    it and on ``--only``. Deriving the seed from the cell's identity makes every
    interval reproducible in isolation.
    """
    key = f"{cohort_name}|{rule}|{col}".encode()
    return int(seed) * 1_000_003 + int(zlib.crc32(key))


def _quantile_mc_se(a: np.ndarray, p: float) -> float:
    """Monte-Carlo standard error of a bootstrap percentile.

    ``se(q_p) = sqrt(p (1 - p) / B) / f(q_p)``, with the density at the quantile
    estimated by a finite difference of the empirical quantile function. This is
    the resampling noise in the *endpoint itself*, i.e. how much the reported
    interval would move under a different bootstrap seed.
    """
    a = np.asarray(a, dtype=float)
    B = a.size
    if B < 50:
        return float("nan")
    h = 0.01
    lo, hi = max(p - h, 0.0), min(p + h, 1.0)
    spread = float(np.percentile(a, hi * 100) - np.percentile(a, lo * 100))
    if not np.isfinite(spread) or spread <= 0:
        return float("nan")
    dens_inv = spread / (hi - lo)          # 1 / f(q_p)
    return float(dens_inv * np.sqrt(p * (1.0 - p) / B))


def _ci(a: np.ndarray) -> Tuple[float, float, float, float]:
    """``(lo95, hi95, mc_se_lo, mc_se_hi)`` from a replicate vector."""
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 50:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)),
            _quantile_mc_se(a, 0.025), _quantile_mc_se(a, 0.975))


def mbc_rows(cohort, n_boot: int = 2000, seed: int = 42) -> List[Dict]:
    rows: List[Dict] = []
    for rule in RULES:
        for col, codes in cohort.codes_by_col.items():
            lv = _levels(cohort.y, cohort.s, codes, rule)
            if len(lv) < 2:
                continue
            per: Dict[int, Dict[str, float]] = {}
            for k, m, n, n_pos, n_neg in lv:
                yk, sk = cohort.y[m], cohort.s[m]
                p_cal, a_cal, b_cal = cox_calibration_test(yk, sk)
                sk_rc = _recalibrated(yk, sk)
                auc, auc_var = auc_delong(yk, sk)
                mbc_raw = model_based_concordance(sk)
                mbc_rc = model_based_concordance(sk_rc)
                per[k] = {
                    "n": n, "n_pos": n_pos, "prevalence": n_pos / n,
                    "obs_auc": auc,
                    "obs_auc_var": auc_var,
                    "mbc": mbc_raw,
                    "mbc_recalibrated": mbc_rc,
                    # How far recalibration has pushed MBC onto the level's own
                    # observed AUROC. Near zero == the recalibrated attribution
                    # is tautological for this level.
                    "mbc_minus_obs_auc": mbc_raw - auc,
                    "mbc_recalibrated_minus_obs_auc": mbc_rc - auc,
                    "lp_sd": float(np.std(linear_predictor(sk), ddof=1)),
                    "calibration_intercept": a_cal,
                    "calibration_slope": b_cal,
                    "calibration_p": p_cal,
                }
            ks = list(per)
            # The extreme pair is selected ONCE, on the observed data, and is
            # then held fixed inside every bootstrap replicate. Point estimate
            # and replicates therefore target the same estimand: the gap between
            # these two named levels. See MBC_FIX.md 2.1 for why the interval is
            # conditional on this selection.
            k_hi = max(ks, key=lambda k: per[k]["obs_auc"])
            k_lo = min(ks, key=lambda k: per[k]["obs_auc"])
            obs_gap = per[k_hi]["obs_auc"] - per[k_lo]["obs_auc"]
            mbc_aligned = per[k_hi]["mbc"] - per[k_lo]["mbc"]
            mbc_aligned_rc = (per[k_hi]["mbc_recalibrated"]
                              - per[k_lo]["mbc_recalibrated"])
            mbc_maxmin = (max(per[k]["mbc"] for k in ks)
                          - min(per[k]["mbc"] for k in ks))

            rng = np.random.default_rng(
                _partition_seed(seed, cohort.name, rule, col))
            bt = _bootstrap_partition(cohort, lv, k_hi, k_lo, rng, n_boot)

            # ── the denominator, and whether it can carry a ratio ────────────
            gap_lo, gap_hi, gap_se_lo, gap_se_hi = _ci(bt["gap"])
            # A ratio is reported only where the observed gap is separated from
            # zero at 95%. Otherwise the fraction is arithmetic, not evidence.
            reportable = bool(np.isfinite(gap_lo) and gap_lo > 0)

            # ── primary summary: the DIFFERENCE, on the AUROC scale ─────────
            # residual = observed gap - model-based-concordance gap = the part
            # of the gap that case mix does NOT explain. No denominator, so it
            # is defined and stable for every partition.
            res_raw = obs_gap - mbc_aligned
            res_rc = obs_gap - mbc_aligned_rc
            r_lo, r_hi, r_se_lo, r_se_hi = _ci(bt["gap"] - bt["mbc_raw"])
            rc_lo, rc_hi, rc_se_lo, rc_se_hi = _ci(bt["gap"] - bt["mbc_rc"])

            # ── secondary summary: the fraction, gated on the denominator ────
            def _frac(num_pt, num_boot):
                pt = num_pt / obs_gap if obs_gap > 0 else float("nan")
                if not reportable:
                    return pt, float("nan"), float("nan"), float("nan"), \
                        float("nan"), float("nan")
                with np.errstate(divide="ignore", invalid="ignore"):
                    fr = num_boot / bt["gap"]
                lo, hi, se_lo, se_hi = _ci(fr)
                mean_se = (float(np.nanstd(fr, ddof=1) / np.sqrt(np.sum(
                    np.isfinite(fr)))) if np.sum(np.isfinite(fr)) > 1
                    else float("nan"))
                return pt, lo, hi, se_lo, se_hi, mean_se

            f_pt, f_lo, f_hi, f_se_lo, f_se_hi, f_mcse = _frac(
                mbc_aligned, bt["mbc_raw"])
            g_pt, g_lo, g_hi, g_se_lo, g_se_hi, g_mcse = _frac(
                mbc_aligned_rc, bt["mbc_rc"])

            # ── sensitivity: the shipped max-min functional, re-selecting the
            # extreme pair in every replicate. Reported for comparison only; its
            # denominator is winner's-curse inflated (MBC_FIX.md 1.2).
            with np.errstate(divide="ignore", invalid="ignore"):
                sel = np.where(bt["gap_sel"] > 0,
                               bt["mbc_raw_sel"] / bt["gap_sel"], np.nan)
            s_lo, s_hi, _, _ = _ci(sel)

            base = {
                "cohort": cohort.name,
                "cohort_label": COHORT_LABELS.get(cohort.name, cohort.name),
                "is_clinical": cohort.is_clinical,
                "n_test": cohort.n_test,
                "rule": rule,
                "partition": col,
                "n_levels_admissible": len(lv),
                "level_max_auc": k_hi,
                "level_min_auc": k_lo,
                "observed_auc_gap": obs_gap,
                "boot_gap_lo95": gap_lo,
                "boot_gap_hi95": gap_hi,
                "boot_gap_mean": float(np.mean(bt["gap"])),
                "boot_gap_mean_selection": float(np.mean(bt["gap_sel"])),
                "mbc_gap_aligned": mbc_aligned,
                "mbc_gap_aligned_recalibrated": mbc_aligned_rc,
                "mbc_gap_maxmin": mbc_maxmin,

                # PRIMARY: unexplained (residual) gap, AUROC units.
                "residual_gap": res_raw,
                "residual_gap_lo95": r_lo,
                "residual_gap_hi95": r_hi,
                "residual_gap_mc_se_lo95": r_se_lo,
                "residual_gap_mc_se_hi95": r_se_hi,
                "residual_gap_recalibrated": res_rc,
                "residual_gap_recalibrated_lo95": rc_lo,
                "residual_gap_recalibrated_hi95": rc_hi,

                # SECONDARY: attributable fraction, gated.
                "fraction_reportable": reportable,
                "casemix_attributable_fraction": f_pt,
                "casemix_attributable_fraction_lo95": f_lo,
                "casemix_attributable_fraction_hi95": f_hi,
                "casemix_attributable_fraction_mc_se_lo95": f_se_lo,
                "casemix_attributable_fraction_mc_se_hi95": f_se_hi,
                "casemix_attributable_fraction_mc_se_mean": f_mcse,
                "casemix_attributable_fraction_recalibrated": g_pt,
                "casemix_attributable_fraction_recalibrated_lo95": g_lo,
                "casemix_attributable_fraction_recalibrated_hi95": g_hi,
                "casemix_attributable_fraction_recalibrated_mc_se_mean": g_mcse,

                # SENSITIVITY: shipped max-min functional (re-selected).
                "fraction_selection_inclusive_lo95": s_lo,
                "fraction_selection_inclusive_hi95": s_hi,

                # Tautology diagnostic: recalibration leaves observed AUROC
                # unchanged (monotone), so MBC_recal collapsing onto obs_auc
                # means the recalibrated fraction is uninformative.
                "max_abs_mbc_recalibrated_minus_obs_auc": max(
                    abs(per[k]["mbc_recalibrated_minus_obs_auc"]) for k in ks),
                "max_abs_mbc_minus_obs_auc": max(
                    abs(per[k]["mbc_minus_obs_auc"]) for k in ks),

                "lp_sd_ratio": (max(per[k]["lp_sd"] for k in ks)
                                / min(per[k]["lp_sd"] for k in ks)),
                "max_abs_calibration_slope_minus_1": max(
                    abs(per[k]["calibration_slope"] - 1.0) for k in ks
                    if np.isfinite(per[k]["calibration_slope"])),
                "min_calibration_p": min(
                    per[k]["calibration_p"] for k in ks
                    if np.isfinite(per[k]["calibration_p"])),
                "boot_n": int(bt["n_ok"]),
                "n_boot": n_boot,
                "seed": seed,
            }
            for k in ks:
                rows.append({**base, "row_type": "level", "level": k,
                             **{f"level_{a}": b for a, b in per[k].items()}})
            rows.append({**base, "row_type": "partition", "level": ""})
    return rows


def _bootstrap_partition(cohort, lv, k_hi, k_lo, rng, n_boot: int) -> Dict:
    """Replicate draws for one partition, with the extreme pair held fixed.

    Resamples rows within (level, outcome), which holds every level's size and
    event count fixed -- the conditioning the DeLong variance also uses -- so the
    interval reflects sampling noise in the AUROC and in the predicted-risk
    distribution, not in the subgroup sizes.

    Every quantity the point estimate reports is recomputed here by the *same*
    code path, on the *same* pair of levels:

    * ``gap``      -- ``AUROC(k_hi) - AUROC(k_lo)`` on the replicate;
    * ``mbc_raw``  -- ``MBC(k_hi) - MBC(k_lo)`` on the replicate's raw scores;
    * ``mbc_rc``   -- the same after a Cox recalibration **refitted inside the
      replicate**, which is what the shipped code omitted;
    * ``gap_sel`` / ``mbc_raw_sel`` -- the max-min functional that re-selects the
      extreme pair per replicate, retained only as a sensitivity.
    """
    gap, mbc_raw, mbc_rc, gap_sel, mbc_raw_sel = [], [], [], [], []
    for _ in range(n_boot):
        stat = {}
        ok = True
        for k, m, n, n_pos, n_neg in lv:
            yk, sk = cohort.y[m], cohort.s[m]
            ip = np.flatnonzero(yk == 1)
            ineg = np.flatnonzero(yk == 0)
            idx = np.r_[rng.choice(ip, ip.size, replace=True),
                        rng.choice(ineg, ineg.size, replace=True)]
            yb, sb = yk[idx], sk[idx]
            a = auc_delong(yb, sb)[0]
            mb = model_based_concordance(sb)
            mr = model_based_concordance(_recalibrated(yb, sb))
            if not (np.isfinite(a) and np.isfinite(mb) and np.isfinite(mr)):
                ok = False
                break
            stat[k] = (a, mb, mr)
        if not ok or k_hi not in stat or k_lo not in stat:
            continue
        # Fixed pair -- the estimand the point estimate reports. The replicate
        # gap is NOT filtered on being positive: dropping the replicates where
        # the observed ordering reverses would condition the distribution and
        # bias the interval away from the point estimate.
        gap.append(stat[k_hi][0] - stat[k_lo][0])
        mbc_raw.append(stat[k_hi][1] - stat[k_lo][1])
        mbc_rc.append(stat[k_hi][2] - stat[k_lo][2])
        # Re-selected pair -- sensitivity only.
        kh = max(stat, key=lambda k: stat[k][0])
        kl = min(stat, key=lambda k: stat[k][0])
        gap_sel.append(stat[kh][0] - stat[kl][0])
        mbc_raw_sel.append(stat[kh][1] - stat[kl][1])
    return {
        "gap": np.asarray(gap, dtype=float),
        "mbc_raw": np.asarray(mbc_raw, dtype=float),
        "mbc_rc": np.asarray(mbc_rc, dtype=float),
        "gap_sel": np.asarray(gap_sel, dtype=float),
        "mbc_raw_sel": np.asarray(mbc_raw_sel, dtype=float),
        "n_ok": len(gap),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--only", type=str, default="")
    args = ap.parse_args(argv)

    names = ([c.strip() for c in args.only.split(",") if c.strip()]
             or list(COHORT_ORDER))
    sd_rows: List[Dict] = []
    mb_rows: List[Dict] = []
    for name in names:
        t0 = time.perf_counter()
        c = load_cohort(name)
        sd_rows += sd_ratio_rows(c)
        mb_rows += mbc_rows(c, n_boot=args.boot, seed=args.seed)
        print(f"  {name:14s} n={c.n_test:6d}  "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sd_rows).to_csv(SD_CSV, index=False)
    print(f"wrote {SD_CSV} ({len(sd_rows)} rows)")
    pd.DataFrame(mb_rows).to_csv(MBC_CSV, index=False)
    print(f"wrote {MBC_CSV} ({len(mb_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
