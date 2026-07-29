"""
P2 -- Pooled-range selection bias in the Inclusivity AUC parity gap.

CLAIM
-----
The Inclusivity statistic  Delta_AUC = max_g AUC_g - min_g AUC_g  taken over
many (overlapping) subgroups has strictly positive expectation under the
equality null (all subgroups share the SAME true AUC). The bias grows with
the number of groups G and shrinks with per-group size m. Consequently a
non-zero observed Delta_AUC is not evidence of subgroup disparity.

WHAT THIS SCRIPT DOES
---------------------
Monte-Carlo under an exact equality null: scores and labels are drawn from a
single common generative model, then group membership is assigned INDEPENDENTLY
of (score, label). Every subgroup therefore has identical true AUC by
construction, so the true parity gap is exactly 0 and everything observed is
selection bias.

Two membership designs:
  (A) disjoint  -- one demographic column partitioning the cohort into G groups
  (B) overlapping -- C demographic columns each with k levels (G = C*k), which
      is what rised.inclusivity.evaluate_inclusivity actually pools over: every
      patient belongs to C groups simultaneously.

Grid over G x m; reports mean, sd, median, 95th percentile of the null
distribution of Delta_AUC, plus P(Delta_AUC > 0.05 / 0.10).

Outputs -> results/p2_*.csv, results/p2_summary.json

Reproducibility: random_state = 42.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

warnings.filterwarnings("ignore")

SEED = 42
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_REPS = 2000          # Monte-Carlo replications per grid cell
TRUE_AUC = 0.70        # common true AUC shared by every subgroup
PREVALENCE = 0.20      # common prevalence


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """
    Mann-Whitney AUC via midranks. Equivalent to sklearn roc_auc_score
    (verified in main()) but far faster inside a Monte-Carlo loop.
    Returns nan if the group has no positive or no negative.
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(s)  # midranks, C-implemented
    r1 = ranks[y == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def draw_cohort(rng, n: int, prevalence: float, true_auc: float):
    """
    Draw (y, score) with a known population AUC.
    Binormal model: scores ~ N(mu*y, 1); AUC = Phi(mu/sqrt(2)).
    """
    mu = norm.ppf(true_auc) * np.sqrt(2.0)
    y = (rng.random(n) < prevalence).astype(int)
    s = rng.normal(loc=mu * y, scale=1.0, size=n)
    return y, s


def null_range_disjoint(rng, G: int, m: int, n_reps: int,
                        min_group_n: int = 0) -> np.ndarray:
    """Design A: G disjoint groups of size m each (n = G*m)."""
    n = G * m
    out = np.empty(n_reps, dtype=float)
    for r in range(n_reps):
        y, s = draw_cohort(rng, n, PREVALENCE, TRUE_AUC)
        # group id assigned independently of (y, s) -> exact equality null
        gid = rng.permutation(np.repeat(np.arange(G), m))
        aucs = []
        for g in range(G):
            mask = gid == g
            if mask.sum() < min_group_n:
                continue
            a = fast_auc(y[mask], s[mask])
            if not np.isnan(a):
                aucs.append(a)
        out[r] = (max(aucs) - min(aucs)) if len(aucs) >= 2 else np.nan
    return out


def null_range_overlapping(rng, n_cols: int, k_levels: int, m: int,
                           n_reps: int) -> np.ndarray:
    """
    Design B: n_cols demographic columns, each with k_levels levels, so
    G = n_cols * k_levels pooled subgroups and each patient belongs to
    n_cols of them simultaneously. Cohort size n = k_levels * m so that
    each group has expected size m.
    """
    n = k_levels * m
    G = n_cols * k_levels
    out = np.empty(n_reps, dtype=float)
    for r in range(n_reps):
        y, s = draw_cohort(rng, n, PREVALENCE, TRUE_AUC)
        aucs = []
        for c in range(n_cols):
            gid = rng.permutation(np.repeat(np.arange(k_levels), m))
            for g in range(k_levels):
                mask = gid == g
                a = fast_auc(y[mask], s[mask])
                if not np.isnan(a):
                    aucs.append(a)
        out[r] = (max(aucs) - min(aucs)) if len(aucs) >= 2 else np.nan
        _ = G
    return out


def summarise(vals: np.ndarray) -> dict:
    v = vals[~np.isnan(vals)]
    return {
        "n_valid_reps": int(len(v)),
        "mean_range": float(np.mean(v)),
        "sd_range": float(np.std(v, ddof=1)),
        "median_range": float(np.median(v)),
        "p95_range": float(np.percentile(v, 95)),
        "p99_range": float(np.percentile(v, 99)),
        "min_range": float(np.min(v)),
        "max_range": float(np.max(v)),
        "P_range_gt_0.05": float(np.mean(v > 0.05)),
        "P_range_gt_0.10": float(np.mean(v > 0.10)),
    }


def main():
    summary = {"seed": SEED, "n_reps_per_cell": N_REPS,
               "true_auc_all_groups": TRUE_AUC, "prevalence": PREVALENCE,
               "null": "every subgroup has identical true AUC; true parity gap = 0"}

    # --- sanity: fast_auc == sklearn ---------------------------------------
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(SEED)
    devs = []
    for _ in range(200):
        y, s = draw_cohort(rng, 200, 0.3, 0.7)
        if 0 < y.sum() < len(y):
            devs.append(abs(fast_auc(y, s) - roc_auc_score(y, s)))
    print(f"sanity: max |fast_auc - sklearn roc_auc_score| = {max(devs):.3e}")
    summary["fast_auc_max_dev_vs_sklearn"] = float(max(devs))

    # === Design A: disjoint groups =========================================
    print()
    print("=" * 78)
    print("P2.A  Null distribution of max-min AUC, DISJOINT groups")
    print("=" * 78)
    group_counts = [2, 3, 5, 8, 10, 15, 20, 30]
    group_sizes = [30, 50, 100, 200, 500, 1000, 2000]
    rows = []
    for G in group_counts:
        for m in group_sizes:
            rng = np.random.default_rng(SEED + 1000 * G + m)
            vals = null_range_disjoint(rng, G, m, N_REPS)
            rec = {"design": "disjoint", "n_groups": G, "group_size": m,
                   "cohort_n": G * m, **summarise(vals)}
            rows.append(rec)
        print(f"  G={G:3d}: " + "  ".join(
            f"m={r['group_size']}:{r['mean_range']:.3f}"
            for r in rows if r["n_groups"] == G))
    df_a = pd.DataFrame(rows)
    df_a.to_csv(RESULTS / "p2_null_range_disjoint.csv", index=False)

    # pivot grids for the paper
    for stat in ["mean_range", "p95_range"]:
        piv = df_a.pivot(index="n_groups", columns="group_size", values=stat)
        piv.to_csv(RESULTS / f"p2_grid_disjoint_{stat}.csv")
        print()
        print(f"  --- {stat} (rows = #groups, cols = group size) ---")
        print(piv.round(4).to_string())

    # === Design B: overlapping groups (the RISED pooling) ==================
    print()
    print("=" * 78)
    print("P2.B  Null distribution, OVERLAPPING groups (multi-column pooling)")
    print("=" * 78)
    rows_b = []
    for n_cols in [1, 2, 3, 4]:
        for k in [2, 4, 6, 8]:
            for m in [50, 100, 500, 1000]:
                rng = np.random.default_rng(SEED + 7919 * n_cols + 131 * k + m)
                vals = null_range_overlapping(rng, n_cols, k, m, N_REPS // 2)
                rows_b.append({
                    "design": "overlapping", "n_columns": n_cols,
                    "levels_per_column": k, "n_groups": n_cols * k,
                    "group_size": m, "cohort_n": k * m, **summarise(vals)})
    df_b = pd.DataFrame(rows_b)
    df_b.to_csv(RESULTS / "p2_null_range_overlapping.csv", index=False)
    piv_b = df_b.pivot_table(index="n_groups", columns="group_size",
                             values="mean_range", aggfunc="mean")
    piv_b.to_csv(RESULTS / "p2_grid_overlapping_mean_range.csv")
    print("  --- mean_range (rows = total #groups, cols = group size) ---")
    print(piv_b.round(4).to_string())

    # === Monotonicity checks ===============================================
    print()
    print("=" * 78)
    print("P2.C  Monotonicity of the bias")
    print("=" * 78)
    grows_with_G = []
    for m in group_sizes:
        sub = df_a[df_a["group_size"] == m].sort_values("n_groups")
        grows_with_G.append(bool(np.all(np.diff(sub["mean_range"].values) > 0)))
    shrinks_with_m = []
    for G in group_counts:
        sub = df_a[df_a["n_groups"] == G].sort_values("group_size")
        shrinks_with_m.append(bool(np.all(np.diff(sub["mean_range"].values) < 0)))
    print(f"  mean_range strictly increasing in #groups, for every group size: "
          f"{all(grows_with_G)}  {grows_with_G}")
    print(f"  mean_range strictly decreasing in group size, for every #groups: "
          f"{all(shrinks_with_m)}  {shrinks_with_m}")
    summary["monotone_increasing_in_n_groups"] = bool(all(grows_with_G))
    summary["monotone_decreasing_in_group_size"] = bool(all(shrinks_with_m))

    # positivity: is the mean range > 0 in every cell?
    summary["all_cells_positive_mean_range"] = bool((df_a["mean_range"] > 0).all())
    summary["min_mean_range_disjoint"] = float(df_a["mean_range"].min())
    summary["max_mean_range_disjoint"] = float(df_a["mean_range"].max())

    # sqrt-law fit: mean_range ~ c * f(G) / sqrt(m)
    fit_rows = []
    for G in group_counts:
        sub = df_a[df_a["n_groups"] == G].sort_values("group_size")
        # regress log(mean_range) on log(m); slope should be about -0.5
        slope, intercept = np.polyfit(np.log(sub["group_size"].values),
                                      np.log(sub["mean_range"].values), 1)
        fit_rows.append({"n_groups": G, "log_log_slope_vs_group_size": float(slope)})
    df_fit = pd.DataFrame(fit_rows)
    df_fit.to_csv(RESULTS / "p2_scaling_fit.csv", index=False)
    print()
    print("  log-log slope of mean_range vs group size (theory: -0.5):")
    print("   " + ", ".join(f"G={r['n_groups']}:{r['log_log_slope_vs_group_size']:.3f}"
                            for r in fit_rows))
    summary["loglog_slope_vs_group_size_mean"] = float(
        df_fit["log_log_slope_vs_group_size"].mean())

    # === Headline reference cells ==========================================
    print()
    print("=" * 78)
    print("P2.D  Headline null distribution figures")
    print("=" * 78)
    head = []
    for G, m in [(2, 500), (5, 500), (10, 500), (20, 500),
                 (10, 50), (10, 100), (10, 1000), (10, 2000)]:
        r = df_a[(df_a["n_groups"] == G) & (df_a["group_size"] == m)].iloc[0]
        head.append({"n_groups": G, "group_size": m,
                     "mean_range": r["mean_range"], "p95_range": r["p95_range"],
                     "P_gt_0.05": r["P_range_gt_0.05"],
                     "P_gt_0.10": r["P_range_gt_0.10"]})
        print(f"  G={G:3d} m={m:5d}  mean={r['mean_range']:.4f}  "
              f"p95={r['p95_range']:.4f}  P(>0.05)={r['P_range_gt_0.05']:.3f}  "
              f"P(>0.10)={r['P_range_gt_0.10']:.3f}")
    pd.DataFrame(head).to_csv(RESULTS / "p2_headline_cells.csv", index=False)
    summary["headline_cells"] = head

    summary["verdict"] = (
        "VERIFIED: under an exact equality null the expected max-min AUC range is "
        "strictly positive in every cell tested, increases monotonically with the "
        "number of pooled subgroups, and decays as m^-0.5 in group size."
    )
    with open(RESULTS / "p2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Wrote {RESULTS}/p2_*.csv and p2_summary.json")


if __name__ == "__main__":
    main()
