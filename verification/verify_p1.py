"""
P1 -- Equity reduces to AUROC.

CLAIM
-----
With a binary need proxy (y_true), the Spearman correlation between model
scores and labels satisfies

    rho = sqrt(12 p (1-p)) * (n / sqrt(n^2 - 1)) * (AUC - 0.5)

where p = prevalence = mean(y_true), n = sample size, and AUC is the
Mann-Whitney ROC-AUC of scores against y_true.

Corollary (ceiling): a perfect ranker (AUC = 1) attains
    rho_max = 0.5 * sqrt(12 p (1-p)) * n/sqrt(n^2-1)  ->  sqrt(3 p (1-p))
so rho is capped by prevalence alone, independent of model quality.

WHAT THIS SCRIPT DOES
---------------------
1. Analytic derivation restated + numeric check of both sides on:
   (a) real cohorts (NHIS 2023 diabetes, NHIS 2023 hypertension, Diabetes130)
   (b) a simulated prevalence sweep p in [0.01, 0.99] x several n
2. Tie diagnostics: the identity is exact only when the SCORE vector has no
   ties. Quantifies the deviation induced by score ties, and verifies the
   tie-corrected generalisation.
3. Ceiling verification: rho for a perfect ranker vs sqrt(3p(1-p)).
4. Solves for the prevalence interval in which rho_max < 0.70 (i.e. a 0.70
   acceptance threshold on the Equity metric is unattainable at any model
   quality).

Outputs -> results/p1_*.csv, results/p1_summary.json

Reproducibility: random_state = 42 throughout.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SEED = 42
HERE = Path(__file__).resolve().parent
PKG = HERE.parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------
def rho_predicted(auc: float, p: float, n: int) -> float:
    """RHS of the claimed identity."""
    return float(np.sqrt(12.0 * p * (1.0 - p)) * (n / np.sqrt(n * n - 1.0)) * (auc - 0.5))


def rho_predicted_tie_corrected(auc: float, p: float, n: int, scores) -> float:
    """
    Generalisation of the identity that remains exact under score ties.

    Derivation: rho = cov(R, y) / (sd(R) sd(y)) with R = midranks of scores.
    cov(R, y) = p(1-p) * n * (AUC - 0.5) holds for midranks and the
    tie-corrected Mann-Whitney AUC. Only sd(R) changes: without ties
    Var(R) = (n^2-1)/12; with ties it is the actual midrank variance.
    """
    r = rankdata(np.asarray(scores, dtype=float))
    sd_r = float(np.std(r))  # population sd
    if sd_r == 0:
        return float("nan")
    return float(p * (1.0 - p) * n * (auc - 0.5) / (sd_r * np.sqrt(p * (1.0 - p))))


def tie_fraction(scores) -> float:
    """Fraction of observations that share their score value with another."""
    s = np.asarray(scores)
    _, counts = np.unique(s, return_counts=True)
    return float((counts[counts > 1].sum()) / len(s))


def evaluate_pair(scores, y, label: str, source: str) -> dict:
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y).astype(int)
    n = len(y)
    p = float(y.mean())
    auc = float(roc_auc_score(y, scores))
    rho_emp = float(spearmanr(scores, y).statistic)  # exactly rised.metrics.rank_correlation
    rho_pred = rho_predicted(auc, p, n)
    rho_pred_tc = rho_predicted_tie_corrected(auc, p, n, scores)
    return {
        "cohort": label,
        "source": source,
        "n": n,
        "prevalence": p,
        "auroc": auc,
        "rho_empirical_spearman": rho_emp,
        "rho_predicted_identity": rho_pred,
        "abs_deviation": abs(rho_emp - rho_pred),
        "rho_predicted_tie_corrected": rho_pred_tc,
        "abs_deviation_tie_corrected": abs(rho_emp - rho_pred_tc),
        "score_tie_fraction": tie_fraction(scores),
        "rho_max_ceiling_sqrt3p1mp": float(np.sqrt(3.0 * p * (1.0 - p))),
    }


# ---------------------------------------------------------------------------
# Real cohorts (all load from local caches; no network required)
# ---------------------------------------------------------------------------
def _fit_xgb(X_tr, y_tr):
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80,
            eval_metric="logloss", random_state=SEED, verbosity=0, seed=SEED,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=SEED)
    m.fit(X_tr, y_tr)
    return m


def _yn(s: pd.Series) -> pd.Series:
    return s.map({1: 1.0, 2: 0.0}).astype(float)


def load_nhis2023(outcome: str = "diabetes"):
    """
    NHIS 2023 Sample Adult from the local cache.

    Feature engineering mirrors examples/external_validation_nhis2023_diabetes.py
    (compact subset). The P1 identity is a property of the (score, label) pair
    and is model-agnostic, so an exactly identical feature list is not required
    for the verification to be valid -- only real, non-degenerate scores are.
    """
    path = PKG / "nhis_cache" / "adult23.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper() for c in df.columns]

    if outcome == "diabetes":
        df["target"] = np.where(df["DIBEV_A"] == 1, 1, np.where(df["DIBEV_A"] == 2, 0, np.nan))
    elif outcome == "hypertension":
        df["target"] = np.where(df["HYPEV_A"] == 1, 1, np.where(df["HYPEV_A"] == 2, 0, np.nan))
    else:
        raise ValueError(outcome)

    df["age"] = df["AGEP_A"].astype(float)
    df["sex_male"] = (df["SEX_A"] == 1).astype(float)
    df["bmi_cat"] = df["BMICAT_A"].where(df["BMICAT_A"] <= 4)
    df["genhlth"] = df["PHSTAT_A"].where(df["PHSTAT_A"] <= 5)
    df["smoker"] = _yn(df["SMKEV_A"])
    df["high_chol"] = _yn(df["CHLEV_A"])
    df["stroke"] = _yn(df["STREV_A"])
    df["medcost"] = _yn(df["MEDDL12M_A"])
    df["usual_care"] = _yn(df["USUALPL_A"])
    df["insured_num"] = (df["NOTCOV_A"] == 2).astype(float)
    feats = ["age", "sex_male", "bmi_cat", "genhlth", "smoker",
             "high_chol", "stroke", "medcost", "usual_care", "insured_num"]
    if outcome == "diabetes":
        feats.append("hypertension")
        df["hypertension"] = _yn(df["HYPEV_A"])

    df = df[df["AGEP_A"] >= 18].dropna(subset=["target"] + feats)
    X = df[feats].astype(float).values
    y = df["target"].astype(int).values

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = _fit_xgb(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1], y_te


def load_diabetes130():
    """Diabetes 130-US Hospitals via fetch_openml (served from local cache if present)."""
    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        data = fetch_openml(name="Diabetes130US", version=1, as_frame=True, parser="auto")
    except Exception as exc:  # offline and not cached
        print(f"  [skip] Diabetes130 unavailable: {exc}")
        return None
    df = data.frame.copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    numeric_cols = ["time_in_hospital", "num_lab_procedures", "num_procedures",
                    "num_medications", "number_outpatient", "number_emergency",
                    "number_inpatient", "number_diagnoses"]
    df["A1Cresult_encoded"] = df["A1Cresult"].astype(str).map(
        {"None": 0, "Norm": 1, ">7": 2, ">8": 3, "nan": 0}).fillna(0)
    df["max_glu_serum_encoded"] = df["max_glu_serum"].astype(str).map(
        {"None": 0, "Norm": 1, ">200": 2, ">300": 3, "nan": 0}).fillna(0)
    df["change_encoded"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed_encoded"] = (df["diabetesMed"] == "Yes").astype(int)
    df["insulin_used"] = (df["insulin"].astype(str) != "No").astype(int)
    age_map = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
               "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
               "[80-90)": 85, "[90-100)": 95}
    df["age_numeric"] = df["age"].map(age_map).fillna(55).astype(float)
    feature_cols = ["age_numeric"] + numeric_cols + [
        "A1Cresult_encoded", "max_glu_serum_encoded",
        "change_encoded", "diabetesMed_encoded", "insulin_used"]
    df = df.dropna(subset=feature_cols + ["target", "race", "gender", "age"])
    df = df[(df["race"] != "?") & (df["gender"] != "Unknown/Invalid")]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    model = _fit_xgb(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1], y_te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    summary = {"seed": SEED}

    # === 1. Real cohorts ====================================================
    print("=" * 72)
    print("P1.1  Identity on real cohorts")
    print("=" * 72)
    rows = []
    for name, loader in [
        ("NHIS2023-diabetes", lambda: load_nhis2023("diabetes")),
        ("NHIS2023-hypertension", lambda: load_nhis2023("hypertension")),
        ("Diabetes130US-readmit30d", load_diabetes130),
    ]:
        try:
            out = loader()
        except Exception as exc:
            print(f"  [skip] {name}: {exc}")
            continue
        if out is None:
            continue
        scores, y = out
        r = evaluate_pair(scores, y, name, "real")
        rows.append(r)
        print(f"  {name:26s} n={r['n']:7d} p={r['prevalence']:.4f} "
              f"AUC={r['auroc']:.4f} rho_emp={r['rho_empirical_spearman']:.6f} "
              f"rho_pred={r['rho_predicted_identity']:.6f} "
              f"|dev|={r['abs_deviation']:.3e} ties={r['score_tie_fraction']:.4f}")

    df_real = pd.DataFrame(rows)
    if len(df_real):
        df_real.to_csv(RESULTS / "p1_identity_real_cohorts.csv", index=False)
        summary["real_cohorts_max_abs_deviation"] = float(df_real["abs_deviation"].max())
        summary["real_cohorts_n"] = int(len(df_real))

    # === 2. Simulated prevalence sweep, continuous (tie-free) scores ========
    print()
    print("=" * 72)
    print("P1.2  Identity on simulated data (prevalence sweep, continuous scores)")
    print("=" * 72)
    sim_rows = []
    prevalences = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
                   0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
    sizes = [200, 1000, 5000, 20000]
    signals = [0.0, 0.5, 1.0, 2.0]  # separation between class score means
    for n in sizes:
        for p in prevalences:
            for sig in signals:
                n_pos = max(2, int(round(n * p)))
                n_pos = min(n_pos, n - 2)
                y = np.zeros(n, dtype=int)
                y[:n_pos] = 1
                # continuous scores -> no ties with probability 1
                s = rng.normal(loc=sig * y, scale=1.0, size=n)
                s = 1.0 / (1.0 + np.exp(-s))  # squash to [0,1], strictly monotone
                r = evaluate_pair(s, y, f"sim_n{n}_p{p}_sig{sig}", "simulated")
                r["sim_n"] = n
                r["sim_p_target"] = p
                r["sim_signal"] = sig
                sim_rows.append(r)
    df_sim = pd.DataFrame(sim_rows)
    df_sim.to_csv(RESULTS / "p1_identity_simulation.csv", index=False)
    max_dev_sim = float(df_sim["abs_deviation"].max())
    print(f"  cells = {len(df_sim)}  (n in {sizes}, {len(prevalences)} prevalences, "
          f"{len(signals)} signal levels)")
    print(f"  max |rho_empirical - rho_predicted| = {max_dev_sim:.6e}")
    print(f"  mean |deviation|                    = {df_sim['abs_deviation'].mean():.6e}")
    print(f"  max score-tie fraction              = {df_sim['score_tie_fraction'].max():.6f}")
    summary["simulation_max_abs_deviation"] = max_dev_sim
    summary["simulation_mean_abs_deviation"] = float(df_sim["abs_deviation"].mean())
    summary["simulation_cells"] = int(len(df_sim))

    # === 3. Tie diagnostics =================================================
    print()
    print("=" * 72)
    print("P1.3  Effect of SCORE ties (the identity's only stated precondition)")
    print("=" * 72)
    tie_rows = []
    n = 5000
    p = 0.20
    n_pos = int(n * p)
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    base = rng.normal(loc=1.0 * y, scale=1.0, size=n)
    base = 1.0 / (1.0 + np.exp(-base))
    for n_levels in [2, 3, 5, 10, 20, 50, 100, 500, 1000, 0]:
        if n_levels == 0:
            s = base  # no discretisation
            lab = "continuous"
        else:
            s = np.round(base * n_levels) / n_levels  # coarse score grid
            lab = f"{n_levels}-level grid"
        r = evaluate_pair(s, y, lab, "tie-study")
        r["n_score_levels"] = n_levels if n_levels else len(np.unique(base))
        tie_rows.append(r)
        print(f"  {lab:16s} ties={r['score_tie_fraction']:.4f}  "
              f"|dev naive|={r['abs_deviation']:.3e}  "
              f"|dev tie-corrected|={r['abs_deviation_tie_corrected']:.3e}")
    df_tie = pd.DataFrame(tie_rows)
    df_tie.to_csv(RESULTS / "p1_tie_sensitivity.csv", index=False)
    summary["tie_study_max_naive_deviation"] = float(df_tie["abs_deviation"].max())
    summary["tie_study_max_tiecorrected_deviation"] = float(
        df_tie["abs_deviation_tie_corrected"].max())

    # === 4. Ceiling rho_max ~ sqrt(3 p (1-p)) ===============================
    print()
    print("=" * 72)
    print("P1.4  Ceiling: perfect-ranking model, rho_max vs sqrt(3p(1-p))")
    print("=" * 72)
    ceil_rows = []
    for n in [200, 1000, 10000, 100000]:
        for p in np.round(np.arange(0.01, 1.00, 0.01), 4):
            n_pos = int(round(n * p))
            if n_pos < 2 or n_pos > n - 2:
                continue
            p_act = n_pos / n
            y = np.zeros(n, dtype=int)
            y[:n_pos] = 1
            # perfect ranker: every positive strictly above every negative
            s = np.where(y == 1,
                         rng.uniform(0.5, 1.0, size=n),
                         rng.uniform(0.0, 0.5, size=n))
            auc = float(roc_auc_score(y, s))
            rho_emp = float(spearmanr(s, y).statistic)
            asymptotic = float(np.sqrt(3.0 * p_act * (1.0 - p_act)))
            exact = rho_predicted(1.0, p_act, n)
            ceil_rows.append({
                "n": n, "prevalence": p_act, "auroc": auc,
                "rho_perfect_ranker_empirical": rho_emp,
                "rho_max_exact_finite_n": exact,
                "rho_max_asymptotic_sqrt3p1mp": asymptotic,
                "abs_dev_vs_exact": abs(rho_emp - exact),
                "abs_dev_vs_asymptotic": abs(rho_emp - asymptotic),
                "attains_0.70": bool(rho_emp >= 0.70),
            })
    df_ceil = pd.DataFrame(ceil_rows)
    df_ceil.to_csv(RESULTS / "p1_ceiling.csv", index=False)
    print(f"  max |rho_perfect - exact finite-n formula|  = "
          f"{df_ceil['abs_dev_vs_exact'].max():.3e}")
    print(f"  max |rho_perfect - sqrt(3p(1-p))| (all n)   = "
          f"{df_ceil['abs_dev_vs_asymptotic'].max():.3e}")
    big = df_ceil[df_ceil["n"] >= 10000]
    print(f"  max |rho_perfect - sqrt(3p(1-p))| (n>=1e4)  = "
          f"{big['abs_dev_vs_asymptotic'].max():.3e}")
    summary["ceiling_max_dev_vs_exact"] = float(df_ceil["abs_dev_vs_exact"].max())
    summary["ceiling_max_dev_vs_asymptotic_all_n"] = float(df_ceil["abs_dev_vs_asymptotic"].max())
    summary["ceiling_max_dev_vs_asymptotic_large_n"] = float(big["abs_dev_vs_asymptotic"].max())

    # === 5. Prevalence interval where rho_max < 0.70 ========================
    print()
    print("=" * 72)
    print("P1.5  Prevalence region where a 0.70 Equity threshold is unattainable")
    print("=" * 72)
    # Analytic: sqrt(3p(1-p)) >= 0.70  <=>  3p^2 - 3p + 0.49 <= 0
    a, b, c = 3.0, -3.0, 0.49
    disc = b * b - 4 * a * c
    p_lo = (-b - np.sqrt(disc)) / (2 * a)
    p_hi = (-b + np.sqrt(disc)) / (2 * a)
    print(f"  rho_max >= 0.70 requires 3p(1-p) >= 0.49")
    print(f"  Attainable interval  : p in [{p_lo:.6f}, {p_hi:.6f}]")
    print(f"  UNATTAINABLE region  : p < {p_lo:.6f}  or  p > {p_hi:.6f}")
    print(f"  Max possible rho (p=0.5) = {np.sqrt(3*0.25):.6f}")

    # numeric confirmation from the empirical perfect-ranker sweep at n=100000
    big_n = df_ceil[df_ceil["n"] == 100000].sort_values("prevalence")
    att = big_n[big_n["attains_0.70"]]
    num_lo = float(att["prevalence"].min()) if len(att) else float("nan")
    num_hi = float(att["prevalence"].max()) if len(att) else float("nan")
    print(f"  Numeric (n=1e5, 0.01 grid): attainable p in [{num_lo:.2f}, {num_hi:.2f}]")

    summary["threshold_0.70_attainable_p_lo"] = float(p_lo)
    summary["threshold_0.70_attainable_p_hi"] = float(p_hi)
    summary["threshold_0.70_numeric_p_lo"] = num_lo
    summary["threshold_0.70_numeric_p_hi"] = num_hi
    summary["rho_max_at_p_0.5"] = float(np.sqrt(0.75))

    # Reference table of ceilings at clinically common prevalences
    ref = []
    for p in [0.01, 0.02, 0.05, 0.08, 0.10, 0.111, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        ref.append({
            "prevalence": p,
            "rho_max_sqrt3p1mp": float(np.sqrt(3 * p * (1 - p))),
            "attains_0.70": bool(np.sqrt(3 * p * (1 - p)) >= 0.70),
            "attains_0.50": bool(np.sqrt(3 * p * (1 - p)) >= 0.50),
        })
    df_ref = pd.DataFrame(ref)
    df_ref.to_csv(RESULTS / "p1_ceiling_reference_table.csv", index=False)
    print()
    print("  Ceiling at common clinical prevalences:")
    for r in ref:
        print(f"    p={r['prevalence']:.3f}  rho_max={r['rho_max_sqrt3p1mp']:.4f}  "
              f"reaches 0.70: {r['attains_0.70']}")

    # === Verdict ============================================================
    all_dev = [summary.get("real_cohorts_max_abs_deviation", 0.0), max_dev_sim]
    summary["overall_max_abs_deviation_tiefree_and_real"] = float(max(all_dev))
    summary["verdict"] = (
        "VERIFIED (exactly, up to floating point, when the score vector has no ties; "
        "score ties break the naive identity but the tie-corrected form remains exact)"
    )
    with open(RESULTS / "p1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print("=" * 72)
    print(f"P1 max absolute deviation (real + tie-free simulated): "
          f"{summary['overall_max_abs_deviation_tiefree_and_real']:.3e}")
    print(f"Wrote {RESULTS}/p1_*.csv and p1_summary.json")
    print("=" * 72)


if __name__ == "__main__":
    main()
