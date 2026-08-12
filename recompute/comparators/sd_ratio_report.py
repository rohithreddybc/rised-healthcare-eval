"""
What ``sd_ratio_robustness.csv`` says: spread, rank stability, decomposition.

Four questions, in the order the editor asked them.

**1. Spread.** For each partition, the median and range of rho-hat across the 24
refit specifications, against the single published value. The headline summary
is not the pooled median of all 504 numbers but the *per-specification* median
over the 21 clinical partitions -- 24 values, one per fit -- because that is the
quantity the manuscript reports as "median 1.145" and it is the one that has to
be stable for the headline to survive.

**2. Rank stability.** The manuscript's most useful claim is ordinal: age
partitions sit high and sex partitions low. Three tests, in increasing
strictness:

* Kendall's W over the 24 specifications' rankings of the 21 partitions, with a
  permutation reference. W = 1 is identical orderings, W = 0 is no agreement
  beyond chance.
* Pairwise Spearman correlation between every pair of specifications' partition
  orderings: median, minimum, and the fraction of pairs below 0.5.
* The age-versus-sex claim on its own terms: within each cohort that has both an
  age and a sex partition, how often does rho-hat(age) exceed rho-hat(sex)? The
  claim is ordinal and paired, so a paired sign test across the five clinical
  cohorts, per specification, is the direct test. A claim that holds in one fit
  and fails in a third of the others is not a finding.

**3. Induced false-flag rate.** Every rho-hat maps to a case-mix false-alarm
rate on the ``casemix_sweep.csv`` curve. The manuscript's claim is that the
median is roughly double nominal (0.05). Reported per specification, so the
claim's stability is visible rather than asserted.

**4. Cohort or model?** A balanced three-way crossed random-effects
decomposition of ``log rho-hat`` over (partition, model class, seed). Logs
because rho-hat is a ratio: an additive model on the log scale is the one whose
components are interpretable as proportional moves. The design is fully crossed
and balanced with one observation per cell, so the ANOVA (Henderson I) moment
estimators are exact and the three-way interaction is the residual. Negative
variance-component estimates are reported as estimated, not truncated at zero,
because truncation biases the shares upward and hides the fact that a component
is indistinguishable from absent.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, spearmanr

from recompute.comparators.core import REPO
from recompute.comparators.sd_ratio_robustness import (
    OUT_CSV,
    SWEEP_METHODS,
)
from recompute.refit import PUBLISHED

RESULTS = REPO / "recompute" / "results"
REPORT_MD = REPO / "SD_RATIO_ROBUSTNESS.md"

#: The published headline, for reference. Recomputed and asserted in the tests.
PUBLISHED_MEDIAN = 1.1449810617386968
PUBLISHED_MIN = 1.0215101992692015
PUBLISHED_MAX = 3.3035828296356278

#: The manuscript's rule and the clinical restriction the 21 partitions come from.
RULE = "m30"

#: Partition names that are age partitions and sex partitions, across cohorts.
AGE_PARTS = ("age_group",)
SEX_PARTS = ("sex", "gender")

#: The incumbent -- the manuscript's own procedure -- on the sweep.
HEADLINE_METHOD = "permutation_null"
NOMINAL = 0.05


# ── loading ──────────────────────────────────────────────────────────────────
def load(path: Path = OUT_CSV, rule: str = RULE, clinical_only: bool = True
         ) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["rule"] == rule].copy()
    if clinical_only:
        df = df[df["is_clinical"]].copy()
    df["spec_id"] = df["spec_id"].astype(str)
    df["partition_key"] = df["partition_key"].astype(str)
    return df


def split_published(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """``(refits, published)``. The published fit is never inside the refit grid."""
    pub = df[df["model_class"] == PUBLISHED].copy()
    ref = df[df["model_class"] != PUBLISHED].copy()
    return ref, pub


def wide(ref: pd.DataFrame, value: str = "partition_sd_ratio") -> pd.DataFrame:
    """``partition_key`` x ``spec_id`` matrix, dropping incomplete partitions.

    A partition that some specification does not admit (the inclusion rule can
    drop a level when a refit's split moves a small stratum below the
    threshold) cannot enter a balanced decomposition or a rank correlation. Its
    identity is reported rather than silently dropped.
    """
    m = ref.pivot_table(index="partition_key", columns="spec_id", values=value,
                        aggfunc="first")
    return m


def balanced_subset(m: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Rows with no missing specification, plus the names of the rows dropped."""
    ok = m.notna().all(axis=1)
    return m.loc[ok], sorted(m.index[~ok].tolist())


# ── (1) spread ───────────────────────────────────────────────────────────────
def per_partition_spread(ref: pd.DataFrame, pub: pd.DataFrame) -> pd.DataFrame:
    g = ref.groupby("partition_key")["partition_sd_ratio"]
    out = pd.DataFrame({
        "n_specs": g.size(),
        "median": g.median(),
        "min": g.min(),
        "max": g.max(),
        "q25": g.quantile(0.25),
        "q75": g.quantile(0.75),
    })
    out["max_over_min"] = out["max"] / out["min"]
    p = pub.set_index("partition_key")["partition_sd_ratio"]
    out["published"] = p
    out["published_pctile"] = [
        float(np.mean(ref.loc[ref["partition_key"] == k,
                              "partition_sd_ratio"].to_numpy()
                      <= out.loc[k, "published"]) * 100.0)
        if k in p.index else np.nan
        for k in out.index
    ]
    return out.sort_values("median", ascending=False)


def per_spec_summary(ref: pd.DataFrame) -> pd.DataFrame:
    """Median / min / max of rho-hat over the partitions, one row per spec."""
    g = ref.groupby(["spec_id", "model_class", "seed"])["partition_sd_ratio"]
    out = g.agg(["size", "median", "min", "max"]).reset_index()
    return out.rename(columns={"size": "n_partitions"})


# ── (2) rank stability ───────────────────────────────────────────────────────
def kendall_w(m: pd.DataFrame) -> Dict[str, float]:
    """Kendall's coefficient of concordance over the specifications' rankings.

    ``m`` is items x raters (partitions x specifications). Ties within a rater
    are handled by midranks and by the tie-corrected denominator, so a rater
    that cannot separate two partitions does not inflate agreement.
    """
    a = m.to_numpy(dtype=float)
    n_items, n_raters = a.shape
    if n_items < 2 or n_raters < 2:
        return {"W": float("nan"), "n_items": n_items, "n_raters": n_raters}
    R = np.apply_along_axis(rankdata, 0, a)          # rank within each rater
    Rsum = R.sum(axis=1)
    S = float(np.sum((Rsum - Rsum.mean()) ** 2))
    # Tie correction: sum over raters of (t^3 - t) for each tie group.
    T = 0.0
    for j in range(n_raters):
        _, counts = np.unique(a[:, j], return_counts=True)
        T += float(np.sum(counts ** 3 - counts))
    denom = (n_raters ** 2 * (n_items ** 3 - n_items) - n_raters * T) / 12.0
    return {"W": float(S / denom) if denom > 0 else float("nan"),
            "n_items": int(n_items), "n_raters": int(n_raters)}


def kendall_w_null(n_items: int, n_raters: int, n_perm: int = 2000,
                   seed: int = 42) -> Dict[str, float]:
    """Reference distribution of W under independent random rankings."""
    rng = np.random.default_rng(seed)
    vals = np.empty(n_perm)
    for b in range(n_perm):
        R = np.column_stack([rng.permutation(n_items) + 1.0
                             for _ in range(n_raters)])
        Rsum = R.sum(axis=1)
        S = float(np.sum((Rsum - Rsum.mean()) ** 2))
        denom = (n_raters ** 2 * (n_items ** 3 - n_items)) / 12.0
        vals[b] = S / denom
    return {"mean": float(vals.mean()), "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)), "n_perm": int(n_perm)}


def pairwise_spearman(m: pd.DataFrame) -> Dict[str, float]:
    cols = list(m.columns)
    rs: List[float] = []
    worst: Tuple[float, str, str] = (np.inf, "", "")
    for i, j in itertools.combinations(range(len(cols)), 2):
        r = float(spearmanr(m[cols[i]].to_numpy(),
                            m[cols[j]].to_numpy()).statistic)
        rs.append(r)
        if r < worst[0]:
            worst = (r, cols[i], cols[j])
    a = np.asarray(rs, dtype=float)
    return {
        "n_pairs": int(a.size),
        "median": float(np.median(a)),
        "q05": float(np.percentile(a, 5)),
        "min": float(a.min()),
        "frac_below_0.5": float(np.mean(a < 0.5)),
        "frac_below_0.7": float(np.mean(a < 0.7)),
        "worst_pair": f"{worst[1]} vs {worst[2]}",
    }


def spearman_vs_published(m: pd.DataFrame, pub: pd.DataFrame
                          ) -> Dict[str, float]:
    """How well each refit's ordering agrees with the published ordering."""
    p = pub.set_index("partition_key")["partition_sd_ratio"]
    p = p.reindex(m.index)
    ok = p.notna().to_numpy()
    rs = [float(spearmanr(m[c].to_numpy()[ok], p.to_numpy()[ok]).statistic)
          for c in m.columns]
    a = np.asarray(rs, dtype=float)
    return {"n_specs": int(a.size), "median": float(np.median(a)),
            "min": float(a.min()), "max": float(a.max()),
            "frac_below_0.5": float(np.mean(a < 0.5))}


def age_vs_sex(ref: pd.DataFrame, pub: pd.DataFrame) -> Dict[str, object]:
    """The ordinal claim, tested where it is actually stated: within a cohort.

    For every (specification, cohort) with both an age and a sex partition, does
    rho-hat(age) exceed rho-hat(sex)? Under the null that the two are
    exchangeable the indicator is a fair coin, so the count over cohorts within
    one specification is Binomial(k, 1/2) -- a paired sign test. The
    specification-level results are then summarised.
    """
    d = ref[ref["partition"].isin(AGE_PARTS + SEX_PARTS)].copy()
    d["kind"] = np.where(d["partition"].isin(AGE_PARTS), "age", "sex")
    piv = d.pivot_table(index=["spec_id", "model_class", "seed", "cohort"],
                        columns="kind", values="partition_sd_ratio",
                        aggfunc="first").dropna()
    piv["age_gt_sex"] = piv["age"] > piv["sex"]
    piv["log_ratio"] = np.log(piv["age"] / piv["sex"])

    by_spec = piv.groupby(level=["spec_id", "model_class", "seed"]).agg(
        n_cohorts=("age_gt_sex", "size"),
        n_age_gt_sex=("age_gt_sex", "sum"),
        median_log_ratio=("log_ratio", "median"),
    ).reset_index()
    by_spec["frac"] = by_spec["n_age_gt_sex"] / by_spec["n_cohorts"]
    by_spec["sign_test_p"] = [
        float(binomtest(int(k), int(n), 0.5, alternative="greater").pvalue)
        for k, n in zip(by_spec["n_age_gt_sex"], by_spec["n_cohorts"])]

    # Published fit, same construction.
    dp = pub[pub["partition"].isin(AGE_PARTS + SEX_PARTS)].copy()
    dp["kind"] = np.where(dp["partition"].isin(AGE_PARTS), "age", "sex")
    pp = dp.pivot_table(index="cohort", columns="kind",
                        values="partition_sd_ratio", aggfunc="first").dropna()
    pub_k = int((pp["age"] > pp["sex"]).sum())
    pub_n = int(len(pp))

    # Per-cohort stability across specs: how often does the cohort's own
    # age > sex ordering hold?
    by_cohort = piv.groupby(level="cohort").agg(
        n_specs=("age_gt_sex", "size"),
        n_age_gt_sex=("age_gt_sex", "sum"),
        median_log_ratio=("log_ratio", "median"),
        min_log_ratio=("log_ratio", "min"),
        max_log_ratio=("log_ratio", "max"),
    )
    by_cohort["frac"] = by_cohort["n_age_gt_sex"] / by_cohort["n_specs"]

    all_pairs = piv["age_gt_sex"].to_numpy()
    return {
        "by_spec": by_spec,
        "by_cohort": by_cohort,
        "n_pairs_total": int(all_pairs.size),
        "n_age_gt_sex_total": int(all_pairs.sum()),
        "frac_total": float(all_pairs.mean()),
        "published_k": pub_k,
        "published_n": pub_n,
        "n_specs_unanimous": int((by_spec["frac"] == 1.0).sum()),
        "n_specs": int(len(by_spec)),
        "median_frac": float(by_spec["frac"].median()),
        "min_frac": float(by_spec["frac"].min()),
        "n_specs_sig05": int((by_spec["sign_test_p"] <= 0.05).sum()),
    }


# ── (3) induced false-flag rate ──────────────────────────────────────────────
def flag_rate_summary(ref: pd.DataFrame, pub: pd.DataFrame,
                      methods: Sequence[str] = SWEEP_METHODS) -> pd.DataFrame:
    rows = []
    for m in methods:
        col = f"induced_flag_rate_{m}"
        if col not in ref.columns:
            continue
        per_spec = ref.groupby("spec_id")[col].median()
        se_col = f"induced_flag_rate_mc_se_{m}"
        rows.append({
            "method": m,
            "published_median": (float(pub[col].median())
                                 if col in pub.columns else np.nan),
            "refit_median_of_spec_medians": float(per_spec.median()),
            "refit_min_spec_median": float(per_spec.min()),
            "refit_max_spec_median": float(per_spec.max()),
            "pooled_median": float(ref[col].median()),
            "pooled_min": float(ref[col].min()),
            "pooled_max": float(ref[col].max()),
            "n_specs_median_above_2x_nominal": int(
                (per_spec > 2 * NOMINAL).sum()),
            "n_specs": int(per_spec.size),
            "sweep_mc_se_at_median": (float(ref[se_col].median())
                                      if se_col in ref.columns else np.nan),
        })
    return pd.DataFrame(rows)


# ── (4) variance decomposition ───────────────────────────────────────────────
def anova3_random(y: np.ndarray) -> Dict[str, float]:
    """Balanced three-way crossed random-effects ANOVA, one obs per cell.

    ``y`` has shape ``(a, b, c)`` -- (partition, model class, seed). The model is

        y_ijk = mu + A_i + B_j + C_k + AB_ij + AC_ik + BC_jk + e_ijk

    with every term random and independent. With one observation per cell the
    residual is the three-way interaction and is not separately identified from
    it; that is stated rather than papered over. Expected mean squares for this
    design give a triangular system that solves exactly:

        E[MS_A]   = s2e + c*s2AB + b*s2AC + b*c*s2A
        E[MS_B]   = s2e + c*s2AB + a*s2BC + a*c*s2B
        E[MS_C]   = s2e + b*s2AC + a*s2BC + a*b*s2C
        E[MS_AB]  = s2e + c*s2AB
        E[MS_AC]  = s2e + b*s2AC
        E[MS_BC]  = s2e + a*s2BC
        E[MS_ABC] = s2e

    Components are reported as estimated, including negative values. Truncating
    a negative estimate at zero biases every share upward and would hide a
    component that the data say is absent.
    """
    y = np.asarray(y, dtype=float)
    a, b, c = y.shape
    gm = y.mean()
    mA = y.mean(axis=(1, 2))
    mB = y.mean(axis=(0, 2))
    mC = y.mean(axis=(0, 1))
    mAB = y.mean(axis=2)
    mAC = y.mean(axis=1)
    mBC = y.mean(axis=0)

    ssA = b * c * np.sum((mA - gm) ** 2)
    ssB = a * c * np.sum((mB - gm) ** 2)
    ssC = a * b * np.sum((mC - gm) ** 2)
    ssAB = c * np.sum((mAB - mA[:, None] - mB[None, :] + gm) ** 2)
    ssAC = b * np.sum((mAC - mA[:, None] - mC[None, :] + gm) ** 2)
    ssBC = a * np.sum((mBC - mB[:, None] - mC[None, :] + gm) ** 2)
    resid = (y - mAB[:, :, None] - mAC[:, None, :] - mBC[None, :, :]
             + mA[:, None, None] + mB[None, :, None] + mC[None, None, :] - gm)
    ssE = float(np.sum(resid ** 2))
    ssT = float(np.sum((y - gm) ** 2))

    dfA, dfB, dfC = a - 1, b - 1, c - 1
    dfAB, dfAC, dfBC = dfA * dfB, dfA * dfC, dfB * dfC
    dfE = dfA * dfB * dfC

    msA, msB, msC = ssA / dfA, ssB / dfB, ssC / dfC
    msAB, msAC, msBC = ssAB / dfAB, ssAC / dfAC, ssBC / dfBC
    msE = ssE / dfE

    s2e = msE
    s2AB = (msAB - s2e) / c
    s2AC = (msAC - s2e) / b
    s2BC = (msBC - s2e) / a
    s2A = (msA - s2e - c * s2AB - b * s2AC) / (b * c)
    s2B = (msB - s2e - c * s2AB - a * s2BC) / (a * c)
    s2C = (msC - s2e - b * s2AC - a * s2BC) / (a * b)

    comp = {
        "partition": s2A, "model_class": s2B, "seed": s2C,
        "partition_x_class": s2AB, "partition_x_seed": s2AC,
        "class_x_seed": s2BC, "residual_3way": s2e,
    }
    # Shares use the sum of the NON-NEGATIVE parts, and the truncation is
    # reported so a reader can see how much of the total it moved.
    pos = {k: max(v, 0.0) for k, v in comp.items()}
    tot = sum(pos.values())
    out: Dict[str, float] = {}
    for k, v in comp.items():
        out[f"var_{k}"] = float(v)
        out[f"share_{k}"] = float(pos[k] / tot) if tot > 0 else float("nan")
    out["total_positive_variance"] = float(tot)
    out["sum_negative_components"] = float(
        sum(v for v in comp.values() if v < 0))
    out["ss_total"] = ssT
    out["dims"] = f"{a}x{b}x{c}"
    # Directly interpretable complement: the fraction of total observed spread
    # in log rho-hat that a one-way "which partition is it?" model explains.
    out["oneway_partition_r2"] = float(ssA / ssT) if ssT > 0 else float("nan")
    out["model_side_share"] = float(
        sum(pos[k] for k in ("model_class", "seed", "partition_x_class",
                             "partition_x_seed", "class_x_seed",
                             "residual_3way")) / tot) if tot > 0 else float("nan")
    return out


def decomposition(ref: pd.DataFrame, classes: Sequence[str],
                  seeds: Sequence[int]) -> Tuple[Dict[str, float], List[str]]:
    """Build the balanced (partition x class x seed) cube of ``log rho-hat``."""
    m = wide(ref)
    bal, dropped = balanced_subset(m)
    parts = list(bal.index)
    cube = np.empty((len(parts), len(classes), len(seeds)))
    for i, p in enumerate(parts):
        for j, c in enumerate(classes):
            for k, s in enumerate(seeds):
                cube[i, j, k] = bal.loc[p, f"{c}|s{s}"]
    res = anova3_random(np.log(cube))
    res["n_partitions"] = len(parts)
    return res, dropped


# ── report ───────────────────────────────────────────────────────────────────
def _fmt(x: float, d: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def build_report(path: Path = OUT_CSV) -> str:
    from recompute.refit import MODEL_CLASSES, SEEDS

    df = load(path)
    ref, pub = split_published(df)
    classes = [c for c in MODEL_CLASSES if c in set(ref["model_class"])]
    seeds = [s for s in SEEDS
             if f"{classes[0]}|s{s}" in set(ref["spec_id"])]

    spread = per_partition_spread(ref, pub)
    specs = per_spec_summary(ref)
    m = wide(ref)
    bal, dropped = balanced_subset(m)
    W = kendall_w(bal)
    Wnull = kendall_w_null(W["n_items"], W["n_raters"])
    ps = pairwise_spearman(bal)
    pvp = spearman_vs_published(bal, pub)
    avs = age_vs_sex(ref, pub)
    flags = flag_rate_summary(ref, pub)
    dec, dec_dropped = decomposition(ref, classes, seeds)

    spec_medians = specs["median"].to_numpy(dtype=float)
    runtime = float(df["fit_runtime_s"].sum())

    L: List[str] = []
    A = L.append

    A("# Is rho-hat robust to model specification?")
    A("")
    A("`rho-hat` is the ratio of the largest to the smallest per-level standard")
    A("deviation of the linear predictor, measured across the 21 clinical")
    A("demographic partitions under the published `m30` inclusion rule. The")
    A("manuscript reports median **1.145**, min **1.022**, max **3.304**, all")
    A("from a single fitted model per cohort at seed 42.")
    A("")
    A("This document refits every cohort under **4 model classes x "
      f"{len(seeds)} seeds = {len(classes) * len(seeds)} specifications**,")
    A("varying the train/test split and the estimator initialisation together,")
    A("and recomputes rho-hat for every partition under every one.")
    A("")

    # ── verdict ──────────────────────────────────────────────────────────────
    A("## Verdict")
    A("")
    med_lo, med_hi = float(spec_medians.min()), float(spec_medians.max())
    A(f"**(a) The median.** Across the {len(specs)} refit specifications the")
    A(f"median rho-hat over the 21 partitions ranges from **{_fmt(med_lo)}** to")
    A(f"**{_fmt(med_hi)}**, with a median of **{_fmt(float(np.median(spec_medians)))}**.")
    A(f"The published value is {_fmt(PUBLISHED_MEDIAN)}.")
    A("")
    A(f"**(b) The age-versus-sex ordering.** Across all "
      f"{avs['n_pairs_total']} (specification, cohort) pairs the age partition's")
    A(f"rho-hat exceeds the sex partition's in **{avs['n_age_gt_sex_total']}** "
      f"({avs['frac_total'] * 100:.0f}%).")
    A(f"{avs['n_specs_unanimous']} of {avs['n_specs']} specifications reproduce "
      "the ordering in every cohort.")
    A("")

    # ── methods ──────────────────────────────────────────────────────────────
    A("## Methods")
    A("")
    A("### Model classes")
    A("")
    A("| class | configuration | which cohorts published it |")
    A("|---|---|---|")
    A("| `xgb_published` | XGBoost, `n_estimators=200, max_depth=4, "
      "learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` -- the "
      "manuscript's own configuration, seed varied | the 6 clinical cohorts "
      "and `synthetic` |")
    A("| `logreg_l2` | `StandardScaler` + L2-penalised logistic regression, "
      "`C=1.0, max_iter=2000` | `adult_income`, `acs_income`, "
      "`german_credit` |")
    A("| `random_forest` | `n_estimators=300, min_samples_leaf=5, "
      "max_features='sqrt'` | none -- new |")
    A("| `hgb_deep` | `HistGradientBoostingClassifier, max_iter=300, "
      "learning_rate=0.10, max_leaf_nodes=63, l2_regularization=1.0` -- a "
      "deliberately different point in the boosting hyperparameter space | "
      "none -- new |")
    A("")
    A("Every cohort gets all four, so every cohort has at least two classes")
    A("beyond its own. No configuration was tuned toward the published numbers;")
    A("the four are declared once in `recompute/refit.py` and applied unchanged.")
    A("")
    A(f"### Seeds")
    A("")
    A(f"`{', '.join(str(s) for s in seeds)}`. Each seed sets **both** the")
    A("train/test split's `random_state` **and** the estimator's own, so a seed")
    A("moves the resampling and the fit together. Diabetes 130 keeps its")
    A("`GroupShuffleSplit` on `patient_nbr` under every seed, so no refit")
    A("reintroduces the row-level leakage the group split exists to remove.")
    A("")
    A("### Traceability")
    A("")
    A("The published fit is carried as its own row (`model_class=published`) and")
    A("reproduces `recompute/results/cohort_sd_ratios.csv` to the last bit; the")
    A("test suite asserts it. The published split is **not** recoverable from")
    A("the loaders' outputs (`train_test_split` depends on row order and the")
    A("reconstruction reorders rows), so the seed-42 refit is a genuinely new")
    A("split rather than a re-run of the published one.")
    A("")
    A(f"Total estimator fit time: **{runtime / 60:.1f} minutes** over "
      f"{len(df['spec_id'].unique())} specifications x 10 cohorts.")
    A("")

    # ── (1) spread ───────────────────────────────────────────────────────────
    A("## 1. Per-partition spread of rho-hat")
    A("")
    A("Across the 24 refits, per partition. `published %ile` is where the")
    A("published value sits inside the refit distribution -- 50 means the")
    A("published fit was typical, 0 or 100 means it was an extreme.")
    A("")
    A("| partition | specs | published | refit median | min | max | max/min | "
      "published %ile |")
    A("|---|---|---|---|---|---|---|---|")
    for k, r in spread.iterrows():
        A(f"| `{k}` | {int(r['n_specs'])} | {_fmt(r['published'])} | "
          f"{_fmt(r['median'])} | "
          f"{_fmt(r['min'])} | {_fmt(r['max'])} | {_fmt(r['max_over_min'], 2)} | "
          f"{_fmt(r['published_pctile'], 0)} |")
    A("")
    n_incomplete = int((spread["n_specs"] < len(specs)).sum())
    if n_incomplete:
        A(f"{n_incomplete} partition(s) have fewer than {len(specs)} "
          "specifications: the `m30` inclusion rule drops a level when a "
          "refit's split pushes a small stratum below the threshold, so the "
          "partition ceases to be evaluable. That is itself instability in the "
          "anchor -- the *set* of 21 partitions is not fixed across "
          "specifications either.")
        A("")
    A("### The headline summary, one row per specification")
    A("")
    A("| model class | seed | n partitions | median | min | max |")
    A("|---|---|---|---|---|---|")
    for _, r in specs.sort_values(["model_class", "seed"]).iterrows():
        A(f"| `{r['model_class']}` | {int(r['seed'])} | "
          f"{int(r['n_partitions'])} | {_fmt(r['median'])} | "
          f"{_fmt(r['min'])} | {_fmt(r['max'])} |")
    A(f"| **published** | 42 | {int(len(pub))} | "
      f"{_fmt(float(pub['partition_sd_ratio'].median()))} | "
      f"{_fmt(float(pub['partition_sd_ratio'].min()))} | "
      f"{_fmt(float(pub['partition_sd_ratio'].max()))} |")
    A("")

    # ── (2) rank stability ───────────────────────────────────────────────────
    A("## 2. Does the ordinal age-versus-sex pattern survive?")
    A("")
    if dropped:
        A(f"{len(dropped)} partition(s) are not admissible under every "
          "specification and are excluded from the rank analysis: "
          + ", ".join(f"`{d}`" for d in dropped) + ".")
        A("")
    A("### Overall concordance of the partition ordering")
    A("")
    A(f"* Kendall's W over {W['n_raters']} specifications ranking "
      f"{W['n_items']} partitions: **{_fmt(W['W'])}**")
    A(f"  (independent random rankings give mean {_fmt(Wnull['mean'])}, "
      f"95th percentile {_fmt(Wnull['p95'])} over {Wnull['n_perm']} draws).")
    A(f"* Pairwise Spearman between specifications ({ps['n_pairs']} pairs): "
      f"median **{_fmt(ps['median'])}**, 5th percentile {_fmt(ps['q05'])}, "
      f"min {_fmt(ps['min'])};")
    A(f"  {ps['frac_below_0.5'] * 100:.0f}% of pairs below 0.5, "
      f"{ps['frac_below_0.7'] * 100:.0f}% below 0.7. Worst pair: "
      f"`{ps['worst_pair']}`.")
    A(f"* Spearman of each refit against the **published** ordering: median "
      f"**{_fmt(pvp['median'])}**, min {_fmt(pvp['min'])}, "
      f"max {_fmt(pvp['max'])}.")
    A("")
    A("### The claim itself: age above sex, within cohort")
    A("")
    A("The claim is ordinal and paired, so the direct test is: in each cohort")
    A("that has both partitions, is rho-hat(age) > rho-hat(sex)?")
    A("")
    A("| cohort | specs | age > sex | frac | median log(age/sex) | min | max |")
    A("|---|---|---|---|---|---|---|")
    for k, r in avs["by_cohort"].iterrows():
        A(f"| `{k}` | {int(r['n_specs'])} | {int(r['n_age_gt_sex'])} | "
          f"{_fmt(r['frac'], 2)} | {_fmt(r['median_log_ratio'])} | "
          f"{_fmt(r['min_log_ratio'])} | {_fmt(r['max_log_ratio'])} |")
    A("")
    A(f"Published fit: {avs['published_k']} of {avs['published_n']} cohorts.")
    A(f"Across refits: **{avs['n_age_gt_sex_total']} of "
      f"{avs['n_pairs_total']}** (specification, cohort) pairs, "
      f"{avs['frac_total'] * 100:.0f}%.")
    A(f"Per-specification paired sign test over cohorts: "
      f"{avs['n_specs_sig05']} of {avs['n_specs']} specifications reach "
      "p <= 0.05.")
    A("")

    # ── (3) induced flag rate ────────────────────────────────────────────────
    A("## 3. The induced false-flag-rate distribution")
    A("")
    A("Each rho-hat is mapped onto the existing `casemix_sweep.csv` curve by")
    A("linear interpolation in the SD ratio. `permutation_null` is the")
    A("incumbent -- the manuscript's own procedure. Nominal level is 0.05, so")
    A("the manuscript's \"median roughly double nominal\" claim is a median")
    A("near 0.10.")
    A("")
    A("| method | published median | refit: median of spec medians | min | max "
      "| specs with median > 0.10 | sweep MC SE |")
    A("|---|---|---|---|---|---|---|")
    for _, r in flags.iterrows():
        A(f"| `{r['method']}` | {_fmt(r['published_median'])} | "
          f"{_fmt(r['refit_median_of_spec_medians'])} | "
          f"{_fmt(r['refit_min_spec_median'])} | "
          f"{_fmt(r['refit_max_spec_median'])} | "
          f"{int(r['n_specs_median_above_2x_nominal'])} / {int(r['n_specs'])} | "
          f"{_fmt(r['sweep_mc_se_at_median'], 4)} |")
    A("")
    A("The sweep itself has Monte-Carlo error: 1,000 simulations per grid node,")
    A("so a flag rate near 0.10 carries an SE near 0.009. Differences between")
    A("specifications smaller than about 0.02 are not resolvable by this curve.")
    A("")

    # ── (4) decomposition ────────────────────────────────────────────────────
    A("## 4. Cohort or model? A variance decomposition")
    A("")
    A(f"Balanced three-way crossed random-effects ANOVA of `log rho-hat` over")
    A(f"({dec['n_partitions']} partitions) x ({len(classes)} model classes) x "
      f"({len(seeds)} seeds), one observation per cell. The residual is the")
    A("three-way interaction and is not separately identified from it.")
    A("")
    A("| component | variance | share of positive total |")
    A("|---|---|---|")
    for key, label in (("partition", "partition (the cohort side)"),
                       ("model_class", "model class"),
                       ("seed", "seed / split"),
                       ("partition_x_class", "partition x class"),
                       ("partition_x_seed", "partition x seed"),
                       ("class_x_seed", "class x seed"),
                       ("residual_3way", "residual (3-way interaction)")):
        A(f"| {label} | {dec['var_' + key]:.5f} | "
          f"{dec['share_' + key] * 100:.1f}% |")
    A("")
    A(f"* Cohort side (partition main effect): **"
      f"{dec['share_partition'] * 100:.1f}%** of the positive variance.")
    A(f"* Model side (everything else): **"
      f"{dec['model_side_share'] * 100:.1f}%**.")
    A(f"* One-way check: a model that knows only *which partition it is* "
      f"explains **{dec['oneway_partition_r2'] * 100:.1f}%** of the total sum "
      "of squares in `log rho-hat`.")
    if dec["sum_negative_components"] < 0:
        A(f"* Negative component estimates summing to "
          f"{dec['sum_negative_components']:.5f} were set to zero for the share "
          "column only; they are reported as estimated in the variance column.")
    A("")
    A("---")
    A("")
    A("Generated by `python -m recompute.comparators.sd_ratio_report`. Inputs:")
    A("`recompute/results/sd_ratio_robustness.csv` (produced by")
    A("`python -m recompute.comparators.sd_ratio_robustness`) and")
    A("`recompute/results/casemix_sweep.csv`.")
    return "\n".join(L) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=str, default=str(OUT_CSV))
    ap.add_argument("--out", type=str, default=str(REPORT_MD))
    args = ap.parse_args(argv)
    txt = build_report(Path(args.csv))
    Path(args.out).write_text(txt, encoding="utf-8")
    print(f"wrote {args.out} ({len(txt)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
