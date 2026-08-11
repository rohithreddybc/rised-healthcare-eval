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

The attributable fraction is reported as ``mbc_gap_aligned / observed_gap``,
where ``mbc_gap_aligned`` compares the *same two levels* that produce the
observed maximum and minimum. A fraction near 1 says the gap is case mix; near 0
says the model really is discriminating differently across subgroups, and the
paper's central claim does not transfer to that cohort.

Caveats that must travel with the number
----------------------------------------
* Model-based concordance assumes the predictions are calibrated in the subgroup
  it is computed on. They frequently are not. Both the raw figure and a variant
  computed after a within-level Cox recalibration (``y ~ a + b logit(s)``) are
  reported; where they disagree, miscalibration is doing the work and neither
  figure is a clean case-mix attribution.
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
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

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


def _recalibrated(y: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Predictions after a within-level Cox recalibration ``a + b logit(s)``."""
    from scipy.optimize import minimize

    x = linear_predictor(s)
    y = np.asarray(y, dtype=float)

    def nll(theta):
        eta = theta[0] + theta[1] * x
        return float(np.sum(np.logaddexp(0.0, eta) - y * eta))

    res = minimize(nll, np.array([0.0, 1.0]), method="BFGS")
    a, b = (res.x if res.success else np.array([0.0, 1.0]))
    return 1.0 / (1.0 + np.exp(-(a + b * x)))


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
def mbc_rows(cohort, n_boot: int = 400, seed: int = 42) -> List[Dict]:
    rng = np.random.default_rng(seed)
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
                per[k] = {
                    "n": n, "n_pos": n_pos, "prevalence": n_pos / n,
                    "obs_auc": auc_delong(yk, sk)[0],
                    "obs_auc_var": auc_delong(yk, sk)[1],
                    "mbc": model_based_concordance(sk),
                    "mbc_recalibrated": model_based_concordance(sk_rc),
                    "lp_sd": float(np.std(linear_predictor(sk), ddof=1)),
                    "calibration_intercept": a_cal,
                    "calibration_slope": b_cal,
                    "calibration_p": p_cal,
                }
            ks = list(per)
            k_hi = max(ks, key=lambda k: per[k]["obs_auc"])
            k_lo = min(ks, key=lambda k: per[k]["obs_auc"])
            obs_gap = per[k_hi]["obs_auc"] - per[k_lo]["obs_auc"]
            mbc_aligned = per[k_hi]["mbc"] - per[k_lo]["mbc"]
            mbc_aligned_rc = (per[k_hi]["mbc_recalibrated"]
                              - per[k_lo]["mbc_recalibrated"])
            mbc_maxmin = (max(per[k]["mbc"] for k in ks)
                          - min(per[k]["mbc"] for k in ks))
            frac = mbc_aligned / obs_gap if obs_gap > 0 else float("nan")

            boot = _bootstrap_fraction(cohort, lv, rng, n_boot)

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
                "mbc_gap_aligned": mbc_aligned,
                "mbc_gap_aligned_recalibrated": mbc_aligned_rc,
                "mbc_gap_maxmin": mbc_maxmin,
                "casemix_attributable_fraction": frac,
                "casemix_attributable_fraction_recalibrated": (
                    mbc_aligned_rc / obs_gap if obs_gap > 0 else float("nan")),
                "boot_fraction_lo95": boot[0],
                "boot_fraction_hi95": boot[1],
                "boot_n": boot[2],
                "lp_sd_ratio": (max(per[k]["lp_sd"] for k in ks)
                                / min(per[k]["lp_sd"] for k in ks)),
                "max_abs_calibration_slope_minus_1": max(
                    abs(per[k]["calibration_slope"] - 1.0) for k in ks
                    if np.isfinite(per[k]["calibration_slope"])),
                "min_calibration_p": min(
                    per[k]["calibration_p"] for k in ks
                    if np.isfinite(per[k]["calibration_p"])),
                "n_boot": n_boot,
                "seed": seed,
            }
            for k in ks:
                rows.append({**base, "row_type": "level", "level": k,
                             **{f"level_{a}": b for a, b in per[k].items()}})
            rows.append({**base, "row_type": "partition", "level": ""})
    return rows


def _bootstrap_fraction(cohort, lv, rng, n_boot: int):
    """Percentile interval for the case-mix-attributable fraction.

    Resamples rows within (level, outcome), which holds every level's size and
    event count fixed -- the conditioning the DeLong variance also uses -- so the
    interval reflects sampling noise in the AUROC and in the predicted-risk
    distribution, not in the subgroup sizes.
    """
    fr = []
    for _ in range(n_boot):
        per = {}
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
            if not (np.isfinite(a) and np.isfinite(mb)):
                ok = False
                break
            per[k] = (a, mb)
        if not ok or len(per) < 2:
            continue
        ks = list(per)
        k_hi = max(ks, key=lambda k: per[k][0])
        k_lo = min(ks, key=lambda k: per[k][0])
        gap = per[k_hi][0] - per[k_lo][0]
        if gap > 0:
            fr.append((per[k_hi][1] - per[k_lo][1]) / gap)
    if len(fr) < 20:
        return float("nan"), float("nan"), len(fr)
    return (float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5)),
            len(fr))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--boot", type=int, default=400)
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
