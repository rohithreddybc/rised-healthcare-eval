"""
External validation of the RISED Framework on NHANES 2021-2023.

Reference: National Center for Health Statistics. National Health and Nutrition
Examination Survey, 2021-2023. U.S. Department of Health and Human Services,
Centers for Disease Control and Prevention. Public-use microdata released 2024.
https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?Cycle=2021-2023

Dataset: ~9,000 adult participants with complete lab work from the
NHANES 2021-2023 (cycle L) public-use XPT files. NHANES is the gold-standard
U.S. nationally representative survey combining interview and physical
examination data, conducted continuously by NCHS since 1960. The 2021-2023
cycle is the most recent completed NHANES cycle with full laboratory results
available (as of May 2026).

Outcome: physician-diagnosed diabetes (DIQ010 == 1; excludes borderline).

Features: age, sex, race/ethnicity, BMI, HbA1c, total cholesterol,
  systolic BP, diastolic BP, hypertension Dx, smoking history, heavy drinking,
  coronary heart disease Dx, stroke Dx, kidney disease Dx, physical activity
  (aerobic guideline met), insurance coverage.

Need proxy (Equity dimension): HbA1c (LBXGH) -- a continuous marker of
  glycaemic burden. Higher values indicate greater clinical need for diabetes
  management intervention and are independent of the binary diagnosis label.

Data access: all five XPT files used here are downloadable without
  registration or DUA from the CDC public-use server at
  https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/

Run:
    python external_validation_nhanes2122.py

The script downloads ~15 MB of XPT data on first run.  Set env var
NHANES_CACHE_DIR to a directory path to cache files locally and avoid
re-downloading on subsequent runs.
"""

from __future__ import annotations

import io
import os
import warnings
warnings.filterwarnings("ignore")

import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import rised
from rised.equity import evaluate_equity


# ---------------------------------------------------------------------------
# Paths and URLs
# ---------------------------------------------------------------------------
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/"

FILES = {
    "DEMO":  "DEMO_L.xpt",    # Demographics
    "DIQ":   "DIQ_L.xpt",     # Diabetes questionnaire (outcome)
    "GHB":   "GHB_L.xpt",     # HbA1c lab
    "TCHOL": "TCHOL_L.xpt",   # Total cholesterol lab
    "BMX":   "BMX_L.xpt",     # Body measures (BMI)
    "BPQ":   "BPQ_L.xpt",     # Blood pressure questionnaire
    "BPXO":  "BPXO_L.xpt",    # Blood pressure exam (oscillometric)
    "SMQ":   "SMQ_L.xpt",     # Smoking
    "ALQ":   "ALQ_L.xpt",     # Alcohol
    "PAQ":   "PAQ_L.xpt",     # Physical activity
    "MCQ":   "MCQ_L.xpt",     # Medical conditions
    "HIQ":   "HIQ_L.xpt",     # Health insurance
}

CACHE_DIR = Path(os.environ.get("NHANES_CACHE_DIR", "nhanes_cache"))


def _load_xpt(key: str) -> pd.DataFrame:
    """Download (or load from cache) one NHANES XPT file."""
    fname = FILES[key]
    cache_path = CACHE_DIR / fname
    if cache_path.exists():
        raw = cache_path.read_bytes()
    else:
        url = BASE_URL + fname
        print(f"  Downloading {fname} ...", flush=True)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                raw = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"Could not download {url}.\n"
                "Check your internet connection or set NHANES_CACHE_DIR to a "
                "directory containing pre-downloaded XPT files."
            ) from exc
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw)
    return pd.read_sas(io.BytesIO(raw), format="xport")


def _yn(s: pd.Series) -> pd.Series:
    """NHANES yes/no recode: 1=Yes->1.0, 2=No->0.0, 7/9=missing->NaN."""
    return s.map({1: 1.0, 2: 0.0}).astype(float)


def main():
    print("Loading NHANES 2021-2023 (cycle L) XPT files ...")

    # 1. Load and merge files on sequence number (SEQN)
    demo  = _load_xpt("DEMO")
    diq   = _load_xpt("DIQ")
    ghb   = _load_xpt("GHB")
    tchol = _load_xpt("TCHOL")
    bmx   = _load_xpt("BMX")
    bpq   = _load_xpt("BPQ")
    bpxo  = _load_xpt("BPXO")
    smq   = _load_xpt("SMQ")
    alq   = _load_xpt("ALQ")
    paq   = _load_xpt("PAQ")
    mcq   = _load_xpt("MCQ")
    hiq   = _load_xpt("HIQ")

    df = (demo
          .merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
          .merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
          .merge(tchol[["SEQN", "LBXTC"]], on="SEQN", how="left")
          .merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
          .merge(bpq[["SEQN", "BPQ020"]], on="SEQN", how="left")
          .merge(bpxo[["SEQN", "BPXOSY1", "BPXODI1"]], on="SEQN", how="left")
          .merge(smq[["SEQN", "SMQ020"]], on="SEQN", how="left")
          .merge(alq[["SEQN", "ALQ151"]], on="SEQN", how="left")
          .merge(paq[["SEQN", "PAD680"]], on="SEQN", how="left")
          .merge(mcq[["SEQN", "MCQ160C", "MCQ160F"]], on="SEQN", how="left")  # MCQ160K removed in L-cycle
          .merge(hiq[["SEQN", "HIQ011"]], on="SEQN", how="left")
    )

    print(f"  Merged: {len(df):,} participants")

    # 2. Restrict to adults (>=18 yrs)
    df = df[df["RIDAGEYR"] >= 18].copy()
    print(f"  Adults (>=18): {len(df):,}")

    # 3. Outcome: physician-diagnosed diabetes (DIQ010==1)
    #    Exclude borderline (3), refused (7), don't know (9)
    df["target"] = np.where(df["DIQ010"] == 1, 1,
                   np.where(df["DIQ010"] == 2, 0, np.nan))

    # 4. Demographic axes
    RACE_MAP = {
        1: "Mexican-American",
        2: "Other-Hispanic",
        3: "NH-White",
        4: "NH-Black",
        5: "NH-Asian",
        6: "NH-Other",
        7: "NH-Other",   # lump rare cells
    }
    df["race"]      = df["RIDRETH3"].map(RACE_MAP)
    df["sex"]       = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["age_group"] = pd.cut(
        df["RIDAGEYR"],
        bins=[18, 35, 50, 65, 200],
        labels=["18-34", "35-49", "50-64", "65+"],
        include_lowest=True,
    ).astype(str)
    df["insured"] = df["HIQ011"].map({1: "Insured", 2: "Uninsured"})

    # 5. Feature engineering
    df["age"]         = df["RIDAGEYR"].astype(float)
    df["sex_male"]    = (df["RIAGENDR"] == 1).astype(float)
    df["bmi"]         = df["BMXBMI"].astype(float)
    df["hba1c"]       = df["LBXGH"].astype(float)
    df["chol"]        = df["LBXTC"].astype(float)
    df["sbp"]         = df["BPXOSY1"].astype(float)
    df["dbp"]         = df["BPXODI1"].astype(float)
    df["htn_dx"]      = _yn(df["BPQ020"])
    df["ever_smoked"] = _yn(df["SMQ020"])
    # ALQ151: 1=Yes (>4/5 drinks/day on 12+ days), 2=No
    df["heavy_drink"] = _yn(df["ALQ151"])
    # PAD680: minutes sedentary/day; invert for "physically active" flag
    df["inactive"]    = (df["PAD680"] >= 480).astype(float)  # >=8h sedentary
    df["chd_dx"]      = _yn(df["MCQ160C"])   # coronary heart disease ever
    df["stroke_dx"]   = _yn(df["MCQ160F"])   # stroke ever
    # MCQ160K (kidney failure) removed from NHANES 2021-2023 MCQ; dropped from feature set
    df["insured_num"] = (df["HIQ011"] == 1).astype(float)

    feature_cols = [
        "age", "sex_male", "bmi", "hba1c", "chol", "sbp", "dbp",
        "htn_dx", "ever_smoked", "heavy_drink", "inactive",
        "chd_dx", "stroke_dx", "insured_num",
    ]
    demo_cols = ["age_group", "sex", "race", "insured"]

    # 6. Drop rows with any missing target, feature, or demographic
    needed = ["target"] + feature_cols + demo_cols
    df = df.dropna(subset=needed).copy()

    print(f"\nCohort: n={len(df):,}, "
          f"prevalence (diabetes Dx)={df['target'].mean():.3f}")
    print(f"Subgroups: sex  {dict(df['sex'].value_counts())}")
    print(f"           race {dict(df['race'].value_counts())}")
    print(f"           age  {dict(df['age_group'].value_counts())}")
    print(f"           ins  {dict(df['insured'].value_counts())}")

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    # 7. Train / test split (80/20, stratified)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)

    # 8. Train XGBoost (same hyperparameters as the rest of the paper)
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80,
            eval_metric="logloss", random_state=42, verbosity=0, seed=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)

    scores_te = model.predict_proba(X_te)[:, 1]
    print(f"\nTest AUROC: {roc_auc_score(y_te, scores_te):.3f}")
    print(f"Test Brier:  {float(np.mean((scores_te - y_te) ** 2)):.3f}")

    # 9. RISED evaluation
    #    feature_index=3 -> hba1c (strong perturbation target: unit-system
    #    drift is a real EHR integration hazard for lab values)
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Noise +10%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "Age +5%"},
        {"type": "unit_rescaling", "feature_index": 3, "factor": 1.08,
         "label": "HbA1c +8%"},   # POCT vs lab-calibrated offset (feature_index 3 = hba1c)
    ]

    print("\nRunning RISED evaluation (B=1000 bootstrap, ~1-3 min) ...")
    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 10. Equity: HbA1c as need proxy.
    #     HbA1c is a continuous measure of glycaemic burden -- patients with
    #     higher HbA1c have greater clinical need for intervention regardless
    #     of whether they carry a formal diagnosis. This is the cleanest
    #     "need" proxy in the RISED suite: it is directly measured, clinically
    #     validated, and not colinear with the binary diagnostic label.
    demo_with_need = d_te.reset_index(drop=True).copy()
    hba1c_te = pd.DataFrame(X_te, columns=feature_cols
                            ).reset_index(drop=True)["hba1c"]
    demo_with_need["hba1c_proxy"] = hba1c_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="hba1c_proxy")

    # Bootstrap CI for rho_need
    B = 1000
    rs = np.random.RandomState(42)
    n = len(scores_te)
    hba1c_arr = hba1c_te.values
    boot_y, boot_hba = [], []
    for _ in range(B):
        idx = rs.choice(n, size=n, replace=True)
        r1, _ = spearmanr(scores_te[idx], y_te[idx])
        r2, _ = spearmanr(scores_te[idx], hba1c_arr[idx])
        boot_y.append(r1)
        boot_hba.append(r2)
    boot_y   = np.array(boot_y)
    boot_hba = np.array(boot_hba)
    rho_y_ci   = (np.percentile(boot_y,   2.5), np.percentile(boot_y,   97.5))
    rho_hba_ci = (np.percentile(boot_hba, 2.5), np.percentile(boot_hba, 97.5))

    # 11. Print scorecard
    print(f"\n=== RISED scorecard on NHANES 2021-2023 (NCHS nationally representative) ===")
    print(f"  Cohort: n={len(df):,}, test n={len(y_te):,}")
    print(f"  Outcome: physician-diagnosed diabetes; "
          f"prevalence={df['target'].mean():.3f}")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    tfr_ci = (tuple(round(x * 100, 1) for x in report.sensitivity.max_tfr_ci)
              if report.sensitivity.max_tfr_ci else None)
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  95% CI {tfr_ci}")
    # Equity against y_true is withdrawn: with a binary outcome proxy the
    # statistic is an affine reparameterisation of AUROC (see rised.equity).
    print(f"  Equity rho_need (HbA1c)    = "
          f"{eq_independent.need_prediction_correlation:.4f}  "
          f"95% CI [{rho_hba_ci[0]:.4f}, {rho_hba_ci[1]:.4f}]")
    print(f"  Deployability batch scoring time (whole cohort) = "
          f"{report.deployability.batch_scoring_time_ms:.3f} ms")
    f_top3 = report.deployability.local_global_topk_agreement
    print(f"  Deployability local-top1-in-global-topk = "
          f"{f_top3:.4f}" if f_top3 is not None else "  Deployability F_top3 = N/A (SHAP error)")
    print(f"  SHAP error: {report.deployability.details.get('shap_error', 'none')}")

    return report, eq_independent


if __name__ == "__main__":
    main()
