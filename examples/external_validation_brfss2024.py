"""
External validation of the RISED Framework on the CDC BRFSS 2024 dataset.

Reference: Centers for Disease Control and Prevention. Behavioral Risk Factor
Surveillance System: 2024 Annual Survey Data. U.S. Department of Health and
Human Services, August 2025.
https://www.cdc.gov/brfss/annual_data/annual_2024.html

Dataset: 457,670 respondents from 49 states + DC + Guam + Puerto Rico + USVI,
collected during calendar year 2024, released August 2025. 345 variables in
the SAS Transport file. Outcome: _MICHD (binary, ever told had coronary heart
disease or myocardial infarction).

Why this dataset for RISED:
* Genuinely contemporary (2024-collected, 2025-released) clinical-survey data
* Largest single-year BRFSS to date (n approx. 457,670)
* Has all four demographic axes RISED's Inclusivity dimension uses:
    age category, sex, race/ethnicity, income tier, plus health-plan coverage
    as an insurance proxy
* No credentialing required (direct CDC download)
* Tabular structure drops into rised.evaluate_all() unchanged

Data prep:
    1. Download the SAS Transport file LLCP2024.XPT.zip from
       https://www.cdc.gov/brfss/annual_data/annual_2024.html
    2. Unzip and either (a) save LLCP2024.XPT alongside this script, or
       (b) edit BRFSS_PATH below to point to its location.

Run:
    python external_validation_brfss2024.py
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


# Path to the unzipped LLCP2024.XPT file. Override here or via env var.
BRFSS_PATH = os.environ.get("BRFSS_PATH", "LLCP2024.XPT")


# Calculated-variable codings used by BRFSS for _MICHD, _AGE_G, _RACE,
# _INCOMG1, and _HLTHPLN. We map them to readable labels and drop the
# don't-know / refused / missing codes.
RACE_LABELS = {
    1: "White", 2: "Black", 3: "AIAN", 4: "Asian",
    5: "NHPI", 6: "Other", 7: "Multiracial", 8: "Hispanic", 9: None,
}
AGE_LABELS = {
    1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54", 5: "55-64", 6: "65+",
}
INCOME_LABELS = {
    1: "<$15K", 2: "$15-25K", 3: "$25-35K", 4: "$35-50K",
    5: "$50-100K", 6: "$100-200K", 7: ">=$200K", 9: None,
}
HEALTHPLAN_LABELS = {1: "Insured", 2: "Uninsured", 9: None}  # _HLTHPL2 coding (2024)


def _yes_no_to_int(s):
    """1 = Yes, 2 = No, 7/9 = don't know/refused -> NaN."""
    return s.map({1: 1, 2: 0}).astype(float)


def main():
    # 1. Load BRFSS 2024 SAS Transport file
    if not os.path.exists(BRFSS_PATH):
        raise FileNotFoundError(
            f"Could not find {BRFSS_PATH}. Download LLCP2024.XPT.zip from "
            "https://www.cdc.gov/brfss/annual_data/annual_2024.html, unzip, "
            "and set BRFSS_PATH or place LLCP2024.XPT next to this script."
        )
    print(f"Loading BRFSS 2024 from {BRFSS_PATH} ...")
    df = pd.read_sas(BRFSS_PATH, format="xport", encoding="latin-1")
    print(f"  raw rows: {len(df):,}, raw columns: {df.shape[1]}")

    # 2. Outcome: _MICHD -- ever told had CHD or MI (1 = yes, 2 = no)
    df["target"] = df["_MICHD"].map({1.0: 1, 2.0: 0})

    # 3. Demographic axes (calculated variables, drop don't-know/refused)
    df["age_group"] = df["_AGE_G"].map(AGE_LABELS)
    df["sex"] = df["SEXVAR"].map({1: "M", 2: "F"})
    df["race"] = df["_RACE"].map(RACE_LABELS)
    df["income"] = df["_INCOMG1"].map(INCOME_LABELS)
    df["health_plan"] = df["_HLTHPL2"].map(HEALTHPLAN_LABELS)  # renamed in 2024

    # 4. Numeric/binary feature matrix (~22 standardized predictors)
    df["bmi"] = df["_BMI5"] / 100.0  # _BMI5 is BMI * 100
    df["age_numeric"] = df["_AGE80"]  # imputed age, top-coded at 80
    df["genhlth"] = df["GENHLTH"].where(df["GENHLTH"] <= 5)  # 1=excellent..5=poor
    df["physhlth"] = df["PHYSHLTH"].replace({77: np.nan, 88: 0, 99: np.nan})
    df["menthlth"] = df["MENTHLTH"].replace({77: np.nan, 88: 0, 99: np.nan})
    # Note: SLEPTIM1, _RFHYPE6 (hypertension), _RFCHOL3 (cholesterol) are absent
    # from the 2024 BRFSS core questionnaire (dropped/rotated); excluded from features.
    df["sex_male"] = (df["SEXVAR"] == 1).astype(float)

    df["smoker"] = _yes_no_to_int(df["_RFSMOK3"])  # current smoker
    df["heavy_drink"] = _yes_no_to_int(df["_RFDRHV9"])  # _RFDRHV8 renamed to _RFDRHV9
    df["phys_active"] = _yes_no_to_int(df["_TOTINDA"])  # any leisure activity
    df["diabetes"] = _yes_no_to_int(df["DIABETE4"].map(
        {1: 1, 2: 1, 3: 0, 4: 0, 7: np.nan, 9: np.nan}))
    df["asthma"] = _yes_no_to_int(df["_LTASTH1"])
    df["stroke"] = _yes_no_to_int(df["CVDSTRK3"])
    df["kidney"] = _yes_no_to_int(df["CHCKDNY2"])
    df["copd"] = _yes_no_to_int(df["CHCCOPD3"])
    df["arthritis"] = _yes_no_to_int(df["_DRDXAR2"])
    df["depression"] = _yes_no_to_int(df["ADDEPEV3"])
    df["medcost"] = _yes_no_to_int(df["MEDCOST1"])  # cost barrier to care
    df["any_insurance"] = _yes_no_to_int(df["_HLTHPL2"])  # _HLTHPLN renamed to _HLTHPL2
    df["checkup_recent"] = (df["CHECKUP1"].isin([1, 2])).astype(float)

    feature_cols = [
        "age_numeric", "sex_male", "bmi", "genhlth", "physhlth", "menthlth",
        "smoker", "heavy_drink", "phys_active",
        "diabetes", "asthma", "stroke",
        "kidney", "copd", "arthritis", "depression",
        "medcost", "any_insurance", "checkup_recent",
    ]
    demo_cols = ["age_group", "sex", "race", "income", "health_plan"]

    # 5. Drop rows missing the outcome, any demographic axis, or any feature
    needed = ["target"] + demo_cols + feature_cols
    df = df.dropna(subset=needed).copy()
    print(f"  after cleaning: n={len(df):,}, "
          f"prevalence (CHD/MI)={df['target'].mean():.3f}")
    print(f"  subgroups: age   {dict(df['age_group'].value_counts())}")
    print(f"             sex   {dict(df['sex'].value_counts())}")
    print(f"             race  {dict(df['race'].value_counts())}")
    print(f"             insurance {dict(df['health_plan'].value_counts())}")

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
    # Use the number of self-reported physical-health bad days (physhlth) as a
    # less-circular need proxy. physhlth IS in the feature set, so this is
    # "less circular" in the same sense as CCI on the synthetic cohort, not
    # unconfounded. An outcome-independent proxy (post-survey hospitalization)
    # would be cleaner and is left to future work.
    demo_with_need = d_te.reset_index(drop=True).copy()
    physhlth_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["physhlth"]
    demo_with_need["physhlth_proxy"] = physhlth_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="physhlth_proxy")

    # Bootstrap CI for rho_need under both proxies
    B = 1000
    rs = np.random.RandomState(42)
    n = len(scores_te)
    phys = physhlth_te.values
    boot_y, boot_phys = [], []
    for _ in range(B):
        idx = rs.choice(n, size=n, replace=True)
        r1, _ = spearmanr(scores_te[idx], y_te[idx])
        r2, _ = spearmanr(scores_te[idx], phys[idx])
        boot_y.append(r1)
        boot_phys.append(r2)
    boot_y = np.array(boot_y)
    boot_phys = np.array(boot_phys)
    rho_y_ci = (np.percentile(boot_y, 2.5), np.percentile(boot_y, 97.5))
    rho_phys_ci = (np.percentile(boot_phys, 2.5),
                   np.percentile(boot_phys, 97.5))

    # 10. Print scorecard
    print("\n=== RISED scorecard on CDC BRFSS 2024 (real, contemporary) ===")
    print(f"  Cohort: n={len(df):,}, test n={len(y_te):,}")
    print(f"  Outcome: CHD or MI; prevalence={df['target'].mean():.3f}")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    tfr_ci = (tuple(round(x*100, 1) for x in report.sensitivity.max_tfr_ci)
              if report.sensitivity.max_tfr_ci else None)
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  95% CI {tfr_ci}")
    print(f"  Equity rho_need (y_true)        = "
          f"{report.equity.need_prediction_correlation:.4f}  "
          f"95% CI [{rho_y_ci[0]:.4f}, {rho_y_ci[1]:.4f}]")
    print(f"  Equity rho_need (physhlth)      = "
          f"{eq_independent.need_prediction_correlation:.4f}  "
          f"95% CI [{rho_phys_ci[0]:.4f}, {rho_phys_ci[1]:.4f}]")
    print(f"  Deployability latency = "
          f"{report.deployability.mean_inference_latency_ms:.3f} ms")
    f_top3 = report.deployability.explanation_faithfulness
    print(f"  Deployability F_top3 = "
          f"{f_top3:.4f}" if f_top3 is not None else "  Deployability F_top3 = N/A (SHAP error)")

    return report, eq_independent


if __name__ == "__main__":
    main()
