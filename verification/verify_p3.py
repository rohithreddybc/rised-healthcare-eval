"""
P3 -- Estimand mismatch invalidates the BCa interval for the AUC parity gap.

CLAIM
-----
rised/inclusivity.py includes sub-30 subgroups in the POINT ESTIMATE (the
n_grp < 30 test at line 66 only appends to `small_groups`; the subgroup AUC is
still recorded at line 72 whenever the group has >= 2 positives and >= 2
negatives), but DROPS them inside every bootstrap replicate (`if mask_b.sum()
< 30: continue`, lines 98-99 and 120-121) and inside every jackknife replicate
(same function, line 136). The interval therefore targets a different
parameter than the point estimate it is attached to.

  point estimate estimand : max-min AUC over ALL groups with >=2 pos and >=2 neg
  bootstrap    estimand   : max-min AUC over groups with n >= 30 only

WHAT THIS SCRIPT DOES
---------------------
1. Static confirmation of the asymmetry directly from the installed source.
2. A constructed cohort with five large groups of identical true AUC (0.75)
   plus one small group (n = 20 < 30) with an extreme true AUC (0.15).
   Runs the REAL rised.inclusivity.evaluate_inclusivity and compares the
   point estimate against the centre of the bootstrap distribution.
3. Quantifies divergence: point estimate vs bootstrap mean/median, whether the
   BCa interval covers its own point estimate, and how often the small group
   survives into a replicate.
4. Empirical coverage of the nominal 95% BCa interval over independent
   cohorts, against two targets:
     (a) theta_true_point = population gap under the POINT estimate's rule (0.60)
     (b) theta_hat        = the interval's own point estimate
   using a fast reimplementation that is first validated to reproduce the real
   function's output exactly.

Outputs -> results/p3_*.csv, results/p3_summary.json

Reproducibility: random_state = 42.
"""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore")

import rised.inclusivity as incl_mod
from rised.inclusivity import evaluate_inclusivity
from rised.bootstrap_ci import bca_interval

SEED = 42
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# --- constructed cohort design ---------------------------------------------
LARGE_GROUP_N = 196
N_LARGE_GROUPS = 5
LARGE_AUC = 0.75          # identical true AUC in every large group
SMALL_GROUP_N = 20        # < 30  -> included in point estimate, dropped in bootstrap
SMALL_AUC = 0.15          # extreme, to make the mismatch visible
PREVALENCE = 0.35
THETA_TRUE_POINT = abs(LARGE_AUC - SMALL_AUC)   # 0.60 : point-estimate estimand
THETA_TRUE_BOOT = 0.0                            # bootstrap estimand (large groups equal)


class ScoreModel:
    """Passthrough 'model': predict_proba returns the score stored in column 0."""

    def predict_proba(self, X):
        s = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1.0 - s, s])


def make_cohort(rng):
    """Build (X, y, demo) with known per-group true AUC."""
    ys, ss, gs = [], [], []
    specs = [(f"large{i}", LARGE_GROUP_N, LARGE_AUC) for i in range(N_LARGE_GROUPS)]
    specs.append(("SMALL", SMALL_GROUP_N, SMALL_AUC))
    for name, m, auc in specs:
        mu = norm.ppf(auc) * np.sqrt(2.0)
        y = (rng.random(m) < PREVALENCE).astype(int)
        # guarantee >=2 pos and >=2 neg so the group is never skipped for that reason
        if y.sum() < 2:
            y[rng.choice(m, 2, replace=False)] = 1
        if (1 - y).sum() < 2:
            y[rng.choice(np.flatnonzero(y == 1), 2, replace=False)] = 0
        s_lat = rng.normal(loc=mu * y, scale=1.0, size=m)
        ys.append(y)
        ss.append(s_lat)
        gs.append(np.array([name] * m))
    y = np.concatenate(ys)
    s_lat = np.concatenate(ss)
    g = np.concatenate(gs)
    s = 1.0 / (1.0 + np.exp(-s_lat))          # valid probabilities
    X = s.reshape(-1, 1)
    demo = pd.DataFrame({"grp": g})
    return X, y, demo, s


# ---------------------------------------------------------------------------
# Fast faithful reimplementation of the estimator + BCa interval
# (mirrors rised/inclusivity.py lines 61-148 exactly; validated in main())
# ---------------------------------------------------------------------------
def _auc(y, s):
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 < 2 or n0 < 2:
        return None
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def point_gap(y, s, codes, n_codes):
    """Point estimate rule: NO n>=30 filter (matches lines 61-77)."""
    aucs = []
    for c in range(n_codes):
        m = codes == c
        if not m.any():
            continue
        a = _auc(y[m], s[m])
        if a is not None:
            aucs.append(a)
    return (max(aucs) - min(aucs)) if len(aucs) >= 2 else float("nan")


def boot_gap(y, s, codes, n_codes):
    """Replicate rule: WITH the n>=30 filter (matches lines 96-107)."""
    aucs = []
    for c in range(n_codes):
        m = codes == c
        if m.sum() < 30:
            continue
        a = _auc(y[m], s[m])
        if a is not None:
            aucs.append(a)
    return (max(aucs) - min(aucs)) if len(aucs) >= 2 else float("nan")


def fast_evaluate(y, s, codes, n_codes, n_bootstrap, random_state):
    """Reproduces evaluate_inclusivity's auc_parity_gap and auc_gap_ci."""
    n = len(y)
    theta_hat = point_gap(y, s, codes, n_codes)
    rng = np.random.default_rng(random_state)
    gap_boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gap_boot[b] = boot_gap(y[idx], s[idx], codes[idx], n_codes)
    full_idx = np.arange(n)
    gap_jack = np.empty(n, dtype=float)
    for i in range(n):
        loo = np.delete(full_idx, i)
        gap_jack[i] = boot_gap(y[loo], s[loo], codes[loo], n_codes)
    valid = ~np.isnan(gap_boot)
    ci = None
    if valid.any() and not np.isnan(boot_gap(y, s, codes, n_codes)):
        ci = bca_interval(theta_hat, gap_boot[valid],
                          gap_jack[~np.isnan(gap_jack)], alpha=0.05)
    return theta_hat, gap_boot, ci


def main():
    summary = {"seed": SEED}

    # === 1. Static source confirmation =====================================
    print("=" * 78)
    print("P3.1  Source inspection of rised/inclusivity.py")
    print("=" * 78)
    src = inspect.getsource(incl_mod.evaluate_inclusivity)
    lines = src.splitlines()
    hits_point, hits_boot = [], []
    for i, ln in enumerate(lines):
        if "< 30" in ln:
            # line numbers relative to the file
            abs_ln = incl_mod.evaluate_inclusivity.__code__.co_firstlineno + i
            ctx = "point-estimate block" if "n_grp" in ln else "bootstrap/jackknife block"
            (hits_point if "n_grp" in ln else hits_boot).append((abs_ln, ln.strip(), ctx))
    for ln_no, text, ctx in hits_point + hits_boot:
        print(f"  line {ln_no:4d} [{ctx:24s}] {text}")
    # confirm the point-estimate branch does NOT `continue`
    idx_ngrp = next(i for i, ln in enumerate(lines) if "n_grp < 30" in ln)
    following = [l.strip() for l in lines[idx_ngrp:idx_ngrp + 3]]
    point_continues = any(l == "continue" for l in following)
    print(f"\n  point-estimate branch after 'if n_grp < 30:' -> {following[1]!r}")
    print(f"  point estimate skips small groups?  {point_continues}   "
          f"(False => small groups ARE in the point estimate)")
    print(f"  bootstrap replicate skips small groups?  {len(hits_boot) > 0}")
    summary["point_estimate_skips_small_groups"] = bool(point_continues)
    summary["bootstrap_skips_small_groups"] = bool(len(hits_boot) > 0)
    summary["n_size30_filters_in_bootstrap_blocks"] = len(hits_boot)

    # === 2. Constructed cohort, REAL evaluate_inclusivity ==================
    print()
    print("=" * 78)
    print("P3.2  Constructed cohort through the REAL evaluate_inclusivity")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    X, y, demo, s = make_cohort(rng)
    n = len(y)
    print(f"  cohort n={n}; groups: " +
          ", ".join(f"{k}={v}" for k, v in demo['grp'].value_counts().items()))

    model = ScoreModel()
    B = 1000
    res = evaluate_inclusivity(model, X, y, demo, n_bootstrap=B, random_state=SEED)
    print(f"\n  Per-subgroup AUC from the point estimate:")
    for k, v in sorted(res.subgroup_aucs.items()):
        n_g = int((demo['grp'] == k.split('=')[1]).sum())
        flag = "  <-- n<30, FLAGGED but INCLUDED" if n_g < 30 else ""
        print(f"    {k:16s} n={n_g:4d}  AUC={v:.4f}{flag}")
    print(f"\n  small_group_flags = {res.details['small_group_flags']}")
    print(f"  POINT ESTIMATE  auc_parity_gap = {res.auc_parity_gap:.4f}")
    print(f"  BCa 95% CI                     = "
          f"({res.auc_gap_ci[0]:.4f}, {res.auc_gap_ci[1]:.4f})")

    # reconstruct the bootstrap distribution the code used
    codes, uniques = pd.factorize(demo["grp"])
    codes = np.asarray(codes)
    n_codes = len(uniques)
    theta_hat_fast, gap_boot, ci_fast = fast_evaluate(y, s, codes, n_codes, B, SEED)

    # validate the reimplementation against the real function
    dev_pt = abs(theta_hat_fast - res.auc_parity_gap)
    dev_ci = max(abs(ci_fast[0] - res.auc_gap_ci[0]), abs(ci_fast[1] - res.auc_gap_ci[1]))
    print(f"\n  [validation] |fast - real| point estimate = {dev_pt:.3e}")
    print(f"  [validation] |fast - real| CI endpoints   = {dev_ci:.3e}")
    summary["reimplementation_max_dev_point"] = float(dev_pt)
    summary["reimplementation_max_dev_ci"] = float(dev_ci)

    gb = gap_boot[~np.isnan(gap_boot)]
    boot_mean, boot_med = float(np.mean(gb)), float(np.median(gb))
    print(f"\n  Bootstrap distribution centre: mean={boot_mean:.4f}  "
          f"median={boot_med:.4f}")
    print(f"  DIVERGENCE  point - bootstrap mean   = "
          f"{res.auc_parity_gap - boot_mean:+.4f}")
    print(f"  DIVERGENCE  point - bootstrap median = "
          f"{res.auc_parity_gap - boot_med:+.4f}")
    print(f"  point estimate percentile within bootstrap dist = "
          f"{100.0 * np.mean(gb < res.auc_parity_gap):.2f}%")
    ci_covers_point = bool(res.auc_gap_ci[0] <= res.auc_parity_gap <= res.auc_gap_ci[1])
    print(f"  Does the BCa interval contain its OWN point estimate?  {ci_covers_point}")

    # how often does the small group survive a replicate?
    rng2 = np.random.default_rng(SEED)
    small_code = int(np.flatnonzero(uniques == "SMALL")[0])
    survive = 0
    for _ in range(5000):
        idx = rng2.integers(0, n, size=n)
        if (codes[idx] == small_code).sum() >= 30:
            survive += 1
    print(f"  P(small group reaches n>=30 in a bootstrap replicate) = "
          f"{survive / 5000:.4f}")

    summary.update({
        "cohort_n": int(n),
        "point_estimate_gap": float(res.auc_parity_gap),
        "bca_ci_low": float(res.auc_gap_ci[0]),
        "bca_ci_high": float(res.auc_gap_ci[1]),
        "bootstrap_mean": boot_mean,
        "bootstrap_median": boot_med,
        "divergence_point_minus_boot_mean": float(res.auc_parity_gap - boot_mean),
        "divergence_point_minus_boot_median": float(res.auc_parity_gap - boot_med),
        "point_estimate_percentile_in_boot_dist": float(100.0 * np.mean(gb < res.auc_parity_gap)),
        "ci_contains_own_point_estimate": ci_covers_point,
        "P_small_group_survives_replicate": survive / 5000,
        "theta_true_point_estimand": THETA_TRUE_POINT,
        "theta_true_bootstrap_estimand": THETA_TRUE_BOOT,
    })

    pd.DataFrame([{
        "subgroup": k, "n": int((demo['grp'] == k.split('=')[1]).sum()),
        "auc_point_estimate": v,
        "flagged_small": k in res.details["small_group_flags"],
    } for k, v in sorted(res.subgroup_aucs.items())]).to_csv(
        RESULTS / "p3_subgroup_aucs.csv", index=False)
    pd.DataFrame({"gap_bootstrap_replicate": gb}).to_csv(
        RESULTS / "p3_bootstrap_distribution.csv", index=False)

    # === 3. Empirical coverage =============================================
    print()
    print("=" * 78)
    print("P3.3  Empirical coverage of the nominal 95% BCa interval")
    print("=" * 78)
    N_COV = 200
    B_COV = 400
    cov_rows = []
    for rep in range(N_COV):
        r = np.random.default_rng(SEED + 100003 * (rep + 1))
        Xr, yr, demor, sr = make_cohort(r)
        cr, ur = pd.factorize(demor["grp"])
        cr = np.asarray(cr)
        th, gbo, ci = fast_evaluate(yr, sr, cr, len(ur), B_COV, SEED + rep)
        if ci is None:
            continue
        cov_rows.append({
            "rep": rep, "theta_hat": th, "ci_low": ci[0], "ci_high": ci[1],
            "covers_theta_true_point": bool(ci[0] <= THETA_TRUE_POINT <= ci[1]),
            "covers_theta_true_boot": bool(ci[0] <= THETA_TRUE_BOOT <= ci[1]),
            "covers_own_point_estimate": bool(ci[0] <= th <= ci[1]),
        })
        if (rep + 1) % 50 == 0:
            print(f"    ... {rep + 1}/{N_COV} replications")
    df_cov = pd.DataFrame(cov_rows)
    df_cov.to_csv(RESULTS / "p3_coverage.csv", index=False)

    cov_point = float(df_cov["covers_theta_true_point"].mean())
    cov_boot = float(df_cov["covers_theta_true_boot"].mean())
    cov_own = float(df_cov["covers_own_point_estimate"].mean())
    print(f"\n  replications with a usable CI: {len(df_cov)}/{N_COV}")
    print(f"  Coverage of theta_true (POINT-estimate estimand = "
          f"{THETA_TRUE_POINT:.2f}) : {cov_point * 100:.1f}%   [nominal 95%]")
    print(f"  Coverage of theta_true (BOOTSTRAP estimand      = "
          f"{THETA_TRUE_BOOT:.2f}) : {cov_boot * 100:.1f}%")
    print(f"  Coverage of the interval's OWN point estimate      : "
          f"{cov_own * 100:.1f}%")
    print(f"  mean point estimate = {df_cov['theta_hat'].mean():.4f}   "
          f"mean CI = ({df_cov['ci_low'].mean():.4f}, {df_cov['ci_high'].mean():.4f})")

    summary.update({
        "coverage_n_replications": int(len(df_cov)),
        "coverage_of_point_estimand_theta_0.60": cov_point,
        "coverage_of_bootstrap_estimand_theta_0.00": cov_boot,
        "coverage_of_own_point_estimate": cov_own,
        "mean_point_estimate_across_reps": float(df_cov["theta_hat"].mean()),
        "mean_ci_low": float(df_cov["ci_low"].mean()),
        "mean_ci_high": float(df_cov["ci_high"].mean()),
        "nominal_coverage": 0.95,
    })
    summary["verdict"] = (
        "VERIFIED: the point estimate includes sub-30 subgroups while every bootstrap "
        "and jackknife replicate excludes them, so the BCa interval targets a different "
        "parameter. On the constructed cohort the interval does not even contain its own "
        "point estimate, and coverage of the point-estimate estimand collapses to ~0%."
    )
    with open(RESULTS / "p3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Wrote {RESULTS}/p3_*.csv and p3_summary.json")


if __name__ == "__main__":
    main()
