"""
External validation of the RISED Framework on the CDC/NCHS NHIS 2023 dataset
with diabetes as the primary outcome.

Reference: National Center for Health Statistics. National Health Interview
Survey, 2023 Public-Use Data File. U.S. Department of Health and Human
Services, Centers for Disease Control and Prevention, released 2024.
https://www.cdc.gov/nchs/nhis/documentation/2023-nhis.html

Dataset: ~27,000 Sample Adult interviews collected during calendar year 2023,
released by NCHS as a public-use CSV file in 2024.  NHIS is the principal
source of information on the health of the U.S. civilian non-institutionalized
population, conducted continuously by NCHS since 1957 and redesigned in 2019.

Why this variant complements external_validation_nhis2024.py:
* Different calendar year (2023 vs. 2024) -> independent sample
* Different clinical outcome: physician-diagnosed diabetes (DIBEV_A)
  vs. composite cardiovascular disease in the 2024 cohort
* Diabetes prediction engages a distinct feature space (BMI, physical
  activity, dietary proxies, kidney disease, HbA1c-adjacent comorbidities)
  and distinct disparities (race/ethnicity gradients for type-2 diabetes)
* Together the two NHIS cohorts demonstrate cross-outcome generalizability

Outcome: physician-diagnosed diabetes (DIBEV_A == 1; borderline excluded).

Features: age, sex, race/ethnicity, BMI category, general health, smoking,
  heavy drinking, physical activity, hypertension Dx, high cholesterol Dx,
  stroke Dx, kidney disease Dx, arthritis Dx, depression Dx, usual-care
  access, delayed care due to cost, insurance coverage.

Need proxy (Equity dimension): self-reported general health (PHSTAT_A,
  1=Excellent..5=Poor). Higher scores represent greater health burden and
  therefore a stronger need for diabetes management services. This proxy
  is available in NHIS without laboratory data and is routinely used in
  population-health equity analyses.

Data: downloaded programmatically from CDC FTP (no registration required).
  Approximately 4.8 MB compressed.

Run:
    python external_validation_nhis2023_diabetes.py
"""

from __future__ import annotations

import io
import os
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import rised
from rised.equity import evaluate_equity


# ---------------------------------------------------------------------------
# Data path
# ---------------------------------------------------------------------------
FTP_URL  = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/"
            "Datasets/NHIS/2023/adult23csv.zip")
CSV_NAME = "adult23.csv"
CACHE_DIR = Path(os.environ.get("NHIS_CACHE_DIR", "nhis_cache"))


def _load_nhis2023() -> pd.DataFrame:
    """Download NHIS 2023 Sample Adult CSV from CDC FTP if not cached."""
    cache_path = CACHE_DIR / CSV_NAME
    if cache_path.exists():
        print(f"  Using cached {cache_path}")
        return pd.read_csv(cache_path, low_memory=False)

    print(f"  Downloading NHIS 2023 Sample Adult (~4.8 MB) ...", flush=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(FTP_URL, timeout=120) as resp:
            raw = resp.read()
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {FTP_URL}.\n"
            "Check your internet connection.  Alternatively, download "
            "adult23csv.zip from https://ftp.cdc.gov/pub/Health_Statistics/"
            "NCHS/Datasets/NHIS/2023/ and unzip to a directory set via the "
            "NHIS_CACHE_DIR environment variable."
        ) from exc
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open(CSV_NAME) as csv_file:
            df = pd.read_csv(csv_file, low_memory=False)
    df.to_csv(cache_path, index=False)   # cache unpacked CSV
    return df


def _yn(s: pd.Series) -> pd.Series:
    """NHIS yes/no recode: 1=Yes->1.0, 2=No->0.0; 7/8/9 -> NaN."""
    return s.map({1: 1.0, 2: 0.0}).astype(float)


def main():
    print("Loading NHIS 2023 Sample Adult ...")
    raw = _load_nhis2023()
    df = raw.copy()
    df.columns = [c.upper() for c in df.columns]
    print(f"  Raw rows: {len(df):,}, columns: {df.shape[1]}")

    # 1. Outcome: physician-diagnosed diabetes
    #    DIBEV_A: 1=Yes, 2=No, 3=Borderline, 7=Refused, 9=Don't know
    df["target"] = np.where(df["DIBEV_A"] == 1, 1,
                   np.where(df["DIBEV_A"] == 2, 0, np.nan))

    # 2. Demographic axes
    RACE_MAP = {
        1: "Hispanic",
        2: "NH-White",
        3: "NH-Black",
        4: "NH-Asian",
        5: "NH-AIAN",
        6: "NH-AIAN+other",
        7: "NH-Other/Multi",
    }
    AGE_BUCKETS = [
        (18, 35, "18-34"),
        (35, 50, "35-49"),
        (50, 65, "50-64"),
        (65, 200, "65+"),
    ]

    def _age_group(age):
        if pd.isna(age):
            return np.nan
        for lo, hi, label in AGE_BUCKETS:
            if lo <= age < hi:
                return label
        return np.nan

    df["age_group"] = df["AGEP_A"].apply(_age_group)
    df["sex"]       = df["SEX_A"].map({1: "Male", 2: "Female"})
    df["race"]      = df["HISPALLP_A"].map(RACE_MAP)
    df["insured"]   = df["NOTCOV_A"].map({1: "Uninsured", 2: "Insured"})

    # 3. Feature engineering
    df["age"]         = df["AGEP_A"].astype(float)
    df["sex_male"]    = (df["SEX_A"] == 1).astype(float)
    df["bmi_cat"]     = df["BMICAT_A"].where(df["BMICAT_A"] <= 4)    # 1-4
    df["genhlth"]     = df["PHSTAT_A"].where(df["PHSTAT_A"] <= 5)    # 1-5

    df["smoker"]       = _yn(df["SMKEV_A"])
    df["current_smk"]  = (df["SMKCIGST_A"].isin([1, 2])).astype(float)
    # Alcohol and PA: use .get() to handle column absence gracefully
    _na = pd.Series(np.nan, index=df.index)
    df["heavy_drink"]  = _yn(df.get("DRKHVY12M_A", _na))
    df["phys_active"]  = _yn(df.get("PA18_02R_A",  _na))
    df["hypertension"] = _yn(df["HYPEV_A"])
    df["high_chol"]    = _yn(df["CHLEV_A"])
    df["stroke"]       = _yn(df["STREV_A"])
    df["arthritis"]    = _yn(df.get("ARTHEV_A", _na))
    df["depression"]   = _yn(df.get("DEPEV_A",  _na))
    df["kidney"]       = _yn(df.get("KIDWEAKEV_A", _na))
    df["medcost"]      = _yn(df["MEDDL12M_A"])
    df["usual_care"]   = _yn(df["USUALPL_A"])
    df["insured_num"]  = (df["NOTCOV_A"] == 2).astype(float)  # insured=2

    # Build feature list from only columns with sufficient non-missing data
    candidate_cols = [
        "age", "sex_male", "bmi_cat", "genhlth",
        "smoker", "current_smk", "heavy_drink", "phys_active",
        "hypertension", "high_chol",
        "stroke", "arthritis", "depression", "kidney",
        "medcost", "usual_care", "insured_num",
    ]
    # Drop columns with >80% missing (gracefully handles year-specific modules)
    feature_cols = [
        c for c in candidate_cols
        if df[c].notna().mean() >= 0.20
    ]
    print(f"  Using {len(feature_cols)} features: {feature_cols}")
    demo_cols = ["age_group", "sex", "race", "insured"]

    # 4. Restrict to adults and drop rows missing outcome / features / demos
    df = df[df["AGEP_A"] >= 18].copy()
    needed = ["target"] + feature_cols + demo_cols
    df = df.dropna(subset=needed).copy()

    print(f"\nCohort: n={len(df):,}, "
          f"prevalence (diabetes Dx)={df['target'].mean():.3f}")
    print(f"Subgroups: sex   {dict(df['sex'].value_counts())}")
    print(f"           race  {dict(df['race'].value_counts())}")
    print(f"           age   {dict(df['age_group'].value_counts())}")
    print(f"           ins   {dict(df['insured'].value_counts())}")

    X    = df[feature_cols].astype(float).values
    y    = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    # 5. Train / test split (80/20, stratified)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)

    # 6. Train XGBoost (same hyperparameters as the rest of the paper)
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

    # 7. RISED evaluation
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Noise +10%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "Age +5%"},
        {"type": "unit_rescaling", "feature_index": 2, "factor": 1.10,
         "label": "BMI-cat +10%"},
    ]

    print("\nRunning RISED evaluation (B=1000 bootstrap, ~1-3 min) ...")
    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 8. Equity: self-reported general health as need proxy.
    #    PHSTAT_A (1=Excellent..5=Poor) is a validated population-health
    #    need measure widely used in NHIS-based equity analyses.
    demo_with_need = d_te.reset_index(drop=True).copy()
    genhlth_te = pd.DataFrame(X_te, columns=feature_cols
                              ).reset_index(drop=True)["genhlth"]
    demo_with_need["genhlth_proxy"] = genhlth_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="genhlth_proxy")

    # Bootstrap CI for rho_need
    B  = 1000
    rs = np.random.RandomState(42)
    n  = len(scores_te)
    gh = genhlth_te.values
    boot_y, boot_gh = [], []
    for _ in range(B):
        idx = rs.choice(n, size=n, replace=True)
        r1, _ = spearmanr(scores_te[idx], y_te[idx])
        r2, _ = spearmanr(scores_te[idx], gh[idx])
        boot_y.append(r1)
        boot_gh.append(r2)
    boot_y  = np.array(boot_y)
    boot_gh = np.array(boot_gh)
    rho_y_ci  = (np.percentile(boot_y,  2.5), np.percentile(boot_y,  97.5))
    rho_gh_ci = (np.percentile(boot_gh, 2.5), np.percentile(boot_gh, 97.5))

    # 9. Print scorecard
    print(f"\n=== RISED scorecard on NHIS 2023 — diabetes (NCHS national survey) ===")
    print(f"  Cohort: n={len(df):,}, test n={len(y_te):,}")
    print(f"  Outcome: physician-diagnosed diabetes; "
          f"prevalence={df['target'].mean():.3f}")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    tfr_ci  = (tuple(round(x * 100, 1) for x in report.sensitivity.max_tfr_ci)
               if report.sensitivity.max_tfr_ci else None)
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  95% CI {tfr_ci}")
    print(f"  Equity rho_need (y_true)      = "
          f"{report.equity.need_prediction_correlation:.4f}  "
          f"95% CI [{rho_y_ci[0]:.4f}, {rho_y_ci[1]:.4f}]")
    print(f"  Equity rho_need (gen-health)  = "
          f"{eq_independent.need_prediction_correlation:.4f}  "
          f"95% CI [{rho_gh_ci[0]:.4f}, {rho_gh_ci[1]:.4f}]")
    print(f"  Deployability latency = "
          f"{report.deployability.mean_inference_latency_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
