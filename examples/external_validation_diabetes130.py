"""
External validation of the RISED Framework on the UCI Diabetes 130-US Hospitals dataset.

Reference: Strack et al. (2014). "Impact of HbA1c measurement on hospital
readmission rates: analysis of 70,000 clinical database patient records."
BioMed Research International, 2014:781670.

Dataset: 101,766 hospital encounters of patients with diabetes from 130 US
hospitals between 1999 and 2008. Target: 30-day hospital readmission.
Demographics: race, gender, age (decade bins).

Why this dataset for RISED:
* Real, modern (post-2000) EHR-derived clinical cohort
* Order of magnitude larger than UCI Heart Disease (n=101,766 vs n=303)
* Has real demographic axes (race, gender, age) for Inclusivity evaluation
* Clinically meaningful binary outcome (early readmission)

Run:
    python external_validation_diabetes130.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import rised
from rised.equity import evaluate_equity


def main():
    # 1. Load UCI Diabetes 130-US Hospitals dataset
    print("Loading Diabetes 130-US Hospitals from OpenML ...")
    data = fetch_openml(name="Diabetes130US", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # 2. Binary outcome: early (<30 day) readmission vs not
    df["target"] = (df["readmitted"] == "<30").astype(int)
    print(f"Cohort: n={len(df)}, prevalence (early readmission)={df['target'].mean():.3f}")

    # 3. Build feature matrix from numeric + key categorical features
    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ]
    # One-hot key categorical
    df["A1Cresult_encoded"] = df["A1Cresult"].astype(str).map(
        {"None": 0, "Norm": 1, ">7": 2, ">8": 3, "nan": 0}).fillna(0)
    df["max_glu_serum_encoded"] = df["max_glu_serum"].astype(str).map(
        {"None": 0, "Norm": 1, ">200": 2, ">300": 3, "nan": 0}).fillna(0)
    df["change_encoded"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed_encoded"] = (df["diabetesMed"] == "Yes").astype(int)
    df["insulin_used"] = (df["insulin"].astype(str) != "No").astype(int)

    feature_cols = numeric_cols + [
        "A1Cresult_encoded", "max_glu_serum_encoded",
        "change_encoded", "diabetesMed_encoded", "insulin_used",
    ]

    # Map age decade buckets to numeric midpoints (kept as feature)
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_numeric"] = df["age"].map(age_map).fillna(55).astype(float)
    feature_cols = ["age_numeric"] + feature_cols

    # Drop missing rows
    df = df.dropna(subset=feature_cols + ["target", "race", "gender", "age"])
    df = df[df["race"] != "?"]
    df = df[df["gender"] != "Unknown/Invalid"]
    print(f"After cleaning: n={len(df)}, prevalence={df['target'].mean():.3f}")

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values

    # Demographic frame (kept categorical for subgroup eval)
    demo = df[["race", "gender", "age"]].copy().rename(
        columns={"age": "age_group"})
    demo["race"] = demo["race"].astype(str)
    demo["gender"] = demo["gender"].astype(str)
    demo["age_group"] = demo["age_group"].astype(str)

    print(f"Subgroups: race {dict(demo['race'].value_counts())}")
    print(f"          gender {dict(demo['gender'].value_counts())}")
    print(f"          age    {dict(demo['age_group'].value_counts())}")

    # 4. Train / test split (80/20)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)

    # 5. Train XGBoost (same hyperparameters as the synthetic-cohort run)
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

    # 6. RISED evaluation
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

    print("\nRunning RISED evaluation ...")
    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 7. Equity with independent need proxy.
    # Use number_inpatient (prior inpatient admissions in past year) as a
    # less-circular need proxy. number_inpatient IS in the feature set,
    # so this is "less circular" in the same sense as CCI on the synthetic
    # cohort -- not unconfounded.
    demo_with_need = d_te.reset_index(drop=True).copy()
    n_inpatient_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["number_inpatient"]
    demo_with_need["n_inpatient_proxy"] = n_inpatient_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="n_inpatient_proxy")

    # Bootstrap CI for rho_need under both proxies
    B = 1000
    rs = np.random.RandomState(42)
    n = len(scores_te)
    boot_y, boot_inp = [], []
    n_inp = n_inpatient_te.values
    for _ in range(B):
        idx = rs.choice(n, size=n, replace=True)
        r1, _ = spearmanr(scores_te[idx], y_te[idx])
        r2, _ = spearmanr(scores_te[idx], n_inp[idx])
        boot_y.append(r1)
        boot_inp.append(r2)
    boot_y = np.array(boot_y)
    boot_inp = np.array(boot_inp)
    rho_y_ci = (np.percentile(boot_y, 2.5), np.percentile(boot_y, 97.5))
    rho_inp_ci = (np.percentile(boot_inp, 2.5), np.percentile(boot_inp, 97.5))

    # 8. Print scorecard
    print("\n=== RISED scorecard on UCI Diabetes 130-US Hospitals (real data) ===")
    print(f"  Cohort: n={len(df):,}, test n={len(y_te):,}")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  "
          f"95% CI {tuple(round(x*100,1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None}")
    # Equity against y_true is withdrawn: with a binary outcome proxy the
    # statistic is an affine reparameterisation of AUROC (see rised.equity).
    print(f"  Equity rho_need (n_inpatient)    = "
          f"{eq_independent.need_prediction_correlation:.4f} "
          f"95% CI [{rho_inp_ci[0]:.4f}, {rho_inp_ci[1]:.4f}]")
    print(f"  Deployability batch scoring time (whole cohort) = "
          f"{report.deployability.batch_scoring_time_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
