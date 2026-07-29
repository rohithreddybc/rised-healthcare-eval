"""
External validation of the RISED Framework on the CDC/NCHS NHIS 2024 dataset.

Reference: National Center for Health Statistics. National Health Interview
Survey, 2024 Public-Use Data File. U.S. Department of Health and Human
Services, Centers for Disease Control and Prevention, released 2025.
https://www.cdc.gov/nchs/nhis/documentation/2024-nhis.html

Dataset: ~29,000 Sample Adult interviews collected during calendar year 2024,
released by NCHS as a public-use CSV file in 2025. NHIS is the principal
source of information on the health of the U.S. civilian non-institutionalized
population, conducted continuously by NCHS since 1957 and redesigned in 2019.
The NHIS 2025 cycle is still in collection and data collection (microdata)
will not be released until late 2026 / 2027; NHIS 2024 is the freshest
publicly downloadable NHIS Sample Adult file as of May 2026.

Why this dataset for RISED:
* NCHS-released (NIH-affiliated through HHS) -- contemporary national-survey
  microdata for "AI in Medicine"-tier external validation
* Genuinely contemporary: 2024-collected, 2025-released
* Has all four demographic axes RISED's Inclusivity dimension uses:
    age, sex, race/ethnicity, family-income tier, plus insurance coverage
* No credentialing required; CSV file format
* Tabular structure drops into rised.evaluate_all() unchanged

Outcome: composite cardiovascular disease (ever told CHD or ever told MI),
mirroring the BRFSS _MICHD construct so cross-cohort comparison is clean.

Data prep:
    1. Download adult24csv.zip from
       https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/adult24csv.zip
    2. Unzip and place adult24.csv next to this script, OR set NHIS_PATH below.

Run:
    python external_validation_nhis2024.py
"""

from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import rised
from rised.equity import evaluate_equity


# Path to the unzipped NHIS 2024 Sample Adult CSV. Override via env var.
NHIS_PATH = os.environ.get("NHIS_PATH", "adult24.csv")


# NHIS 2024 calculated-variable codings. NHIS uses 7=refused, 8=not
# ascertained, 9=don't know across most binary recodes; these become NaN.
RACE_LABELS = {
    1: "Hispanic", 2: "NH-White", 3: "NH-Black", 4: "NH-Asian",
    5: "NH-AIAN", 6: "NH-AIAN+other", 7: "NH-Other/Multi",
}
# AGEP_A is top-coded at 85; we bucket into NHIS standard age groups
AGE_BUCKETS = [(18, 25, "18-24"), (25, 35, "25-34"), (35, 45, "35-44"),
               (45, 55, "45-54"), (55, 65, "55-64"), (65, 200, "65+")]
INCOME_LABELS = {
    1: "<$35K", 2: "$35-50K", 3: "$50-75K",
    4: "$75-100K", 5: ">=$100K",
}
INSURANCE_LABELS = {1: "Uninsured", 2: "Insured"}


def _yn(s):
    """NHIS recode: 1=Yes, 2=No, 7/8/9=missing -> NaN."""
    return s.map({1: 1.0, 2: 0.0}).astype(float)


def _age_bucket(age):
    if pd.isna(age):
        return np.nan
    for lo, hi, label in AGE_BUCKETS:
        if lo <= age < hi:
            return label
    return np.nan


def main():
    # 1. Load NHIS 2024 Sample Adult file
    if not os.path.exists(NHIS_PATH):
        raise FileNotFoundError(
            f"Could not find {NHIS_PATH}. Download adult24csv.zip from "
            "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHIS/2024/"
            "adult24csv.zip, unzip, and set NHIS_PATH or place adult24.csv "
            "next to this script."
        )
    print(f"Loading NHIS 2024 Sample Adult from {NHIS_PATH} ...")
    df = pd.read_csv(NHIS_PATH, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    print(f"  raw rows: {len(df):,}, raw columns: {df.shape[1]}")

    # 2. Composite cardiovascular outcome: ever told CHD or ever told MI
    chdev = _yn(df.get("CHDEV_A", pd.Series(np.nan, index=df.index)))
    miev = _yn(df.get("MIEV_A", pd.Series(np.nan, index=df.index)))
    df["target"] = ((chdev == 1) | (miev == 1)).astype(int)
    # If both vars are missing for a respondent, drop them
    df.loc[chdev.isna() & miev.isna(), "target"] = np.nan

    # 3. Demographic axes (NHIS standard recodes)
    df["age_group"] = df["AGEP_A"].apply(_age_bucket)
    df["sex"] = df["SEX_A"].map({1: "M", 2: "F"})
    df["race"] = df["HISPALLP_A"].map(RACE_LABELS)
    # RATCAT_A is a 14-level poverty-ratio variable; collapse to 5 tiers
    rat = df.get("RATCAT_A")
    if rat is not None:
        # 1-3 = <1.0 PIR (~<$35K), 4-6 = 1.0-1.5 (~$35-50K),
        # 7-9 = 1.5-3.0 (~$50-75K), 10-12 = 3.0-5.0 (~$75-100K),
        # 13-14 = >=5.0 (>=$100K). 98/99 = unknown -> NaN.
        df["income"] = pd.cut(
            rat.where(rat <= 14),
            bins=[0, 3.5, 6.5, 9.5, 12.5, 14.5],
            labels=["<$35K", "$35-50K", "$50-75K", "$75-100K", ">=$100K"],
        ).astype(str).replace({"nan": np.nan})
    else:
        df["income"] = np.nan
    df["insurance"] = df["NOTCOV_A"].map(INSURANCE_LABELS)

    # 4. Numeric/binary feature matrix
    df["age_numeric"] = df["AGEP_A"].where(df["AGEP_A"] <= 85)
    df["sex_male"] = (df["SEX_A"] == 1).astype(float)
    # BMICAT_A: 1=under, 2=normal, 3=over, 4=obese
    df["bmi_cat"] = df["BMICAT_A"].where(df["BMICAT_A"] <= 4)

    # Self-rated general health (PHSTAT_A: 1=excellent..5=poor)
    df["genhlth"] = df["PHSTAT_A"].where(df["PHSTAT_A"] <= 5)

    df["smoker"] = _yn(df["SMKEV_A"])             # ever smoked >=100 cigs
    df["current_smoker"] = (df["SMKCIGST_A"].isin([1, 2])).astype(float)
    df["heavy_drink"] = _yn(df["DRKHVY12M_A"])    # heavy drinking past 12mo
    df["phys_active"] = _yn(df["PA18_02R_A"])     # meets aerobic guidelines

    # Risk factors / comorbid conditions (used as features, not outcomes)
    df["hypertension"] = _yn(df["HYPEV_A"])
    df["high_chol"] = _yn(df["CHLEV_A"])
    # DIBEV_A: 1=yes, 2=no, 3=borderline. Treat 3 as 0 (no clinical Dx yet).
    dibev_recoded = df["DIBEV_A"].map(
        {1: 1, 2: 2, 3: 2, 7: np.nan, 8: np.nan, 9: np.nan})
    df["diabetes"] = _yn(dibev_recoded)
    df["asthma"] = _yn(df["ASEV_A"])
    df["stroke"] = _yn(df["STREV_A"])
    df["copd"] = _yn(df["COPDEV_A"])
    df["arthritis"] = _yn(df["ARTHEV_A"])
    df["depression"] = _yn(df["DEPEV_A"])
    df["kidney"] = _yn(df["KIDWEAKEV_A"])         # weak/failing kidneys ever
    df["medcost"] = _yn(df["MEDDL12M_A"])         # delayed care due to cost
    df["usual_care"] = _yn(df["USUALPL_A"])

    feature_cols = [
        "age_numeric", "sex_male", "bmi_cat", "genhlth",
        "smoker", "current_smoker", "heavy_drink", "phys_active",
        "hypertension", "high_chol", "diabetes",
        "asthma", "stroke", "copd", "arthritis", "depression", "kidney",
        "medcost", "usual_care",
    ]
    demo_cols = ["age_group", "sex", "race", "income", "insurance"]

    # 5. Drop rows missing the outcome, demographics, or features
    needed = ["target"] + demo_cols + feature_cols
    df = df.dropna(subset=needed).copy()
    print(f"  after cleaning: n={len(df):,}, "
          f"prevalence (CHD/MI)={df['target'].mean():.3f}")
    print(f"  subgroups: age   {dict(df['age_group'].value_counts())}")
    print(f"             sex   {dict(df['sex'].value_counts())}")
    print(f"             race  {dict(df['race'].value_counts())}")
    print(f"             insurance {dict(df['insurance'].value_counts())}")

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    # 6. Train / test split (80/20)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)

    # 7. Train XGBoost (same hyperparameters as the rest of the paper)
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

    # 8. RISED evaluation
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Noise +10%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "Age +5%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.06,
         "label": "Age +6%"},
    ]

    print("\nRunning RISED evaluation (B=1000 bootstrap, may take 1-3 min) ...")
    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 9. Equity with independent need proxy.
    # Use self-rated general health (genhlth, 1=excellent..5=poor) as a
    # less-circular need proxy. genhlth IS in the feature set, so this is
    # "less circular" in the same sense as CCI on the synthetic cohort,
    # not unconfounded. A linked outcome-independent measure (e.g.,
    # subsequent hospitalization in linked NCHS-NDI data) would be cleaner
    # and is left to future work.
    demo_with_need = d_te.reset_index(drop=True).copy()
    genhlth_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["genhlth"]
    demo_with_need["genhlth_proxy"] = genhlth_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="genhlth_proxy")

    # Bootstrap CI for rho_need under both proxies
    B = 1000
    rs = np.random.RandomState(42)
    n = len(scores_te)
    gh = genhlth_te.values
    boot_y, boot_gh = [], []
    for _ in range(B):
        idx = rs.choice(n, size=n, replace=True)
        r1, _ = spearmanr(scores_te[idx], y_te[idx])
        r2, _ = spearmanr(scores_te[idx], gh[idx])
        boot_y.append(r1)
        boot_gh.append(r2)
    boot_y = np.array(boot_y)
    boot_gh = np.array(boot_gh)
    rho_y_ci = (np.percentile(boot_y, 2.5), np.percentile(boot_y, 97.5))
    rho_gh_ci = (np.percentile(boot_gh, 2.5),
                 np.percentile(boot_gh, 97.5))

    # 10. Print scorecard
    print("\n=== RISED scorecard on NHIS 2024 (NCHS contemporary survey) ===")
    print(f"  Cohort: n={len(df):,}, test n={len(y_te):,}")
    print(f"  Outcome: ever told CHD or MI; prevalence={df['target'].mean():.3f}")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    tfr_ci = (tuple(round(x*100, 1) for x in report.sensitivity.max_tfr_ci)
              if report.sensitivity.max_tfr_ci else None)
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  95% CI {tfr_ci}")
    # Equity against y_true is withdrawn: with a binary outcome proxy the
    # statistic is an affine reparameterisation of AUROC (see rised.equity).
    print(f"  Equity rho_need (gen-health)    = "
          f"{eq_independent.need_prediction_correlation:.4f}  "
          f"95% CI [{rho_gh_ci[0]:.4f}, {rho_gh_ci[1]:.4f}]")
    print(f"  Deployability batch scoring time (whole cohort) = "
          f"{report.deployability.batch_scoring_time_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
