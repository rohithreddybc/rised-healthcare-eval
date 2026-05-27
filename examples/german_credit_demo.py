"""
Cross-domain demo: RISED evaluation on the Statlog German Credit dataset.

Reference: Hofmann (1994), UCI Machine Learning Repository.
Accessed via sklearn.datasets.fetch_openml('credit-g', version=1).

This is the second cross-domain reference cohort for the RISED Framework
(Section 5 of the main paper). German Credit is the canonical European
benchmark for credit-risk fairness work and complements the US-centred
Adult Income demo in `adult_income_demo.py`.

Domain-calibrated thresholds applied here:
    Reliability  PSS < 0.05                (RISED default)
    Inclusivity  selection-rate ratio >= 0.80    (EEOC-style adverse impact)
    Sensitivity  max TFR < 10%             (RISED default, +/- 5pp sweep)
    Equity       proxy = savings-status    (independent of credit-risk label)
    Deployability latency < 100 ms         (RISED default)

Run:
    python german_credit_demo.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import rised
from rised.equity import evaluate_equity


def selection_rate_ratio(y_pred: np.ndarray, group: pd.Series) -> float:
    rates = {}
    for g in group.unique():
        mask = group == g
        rates[g] = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0
    if max(rates.values()) == 0:
        return 1.0
    return min(rates.values()) / max(rates.values())


def main():
    # 1. Load Statlog German Credit (Hofmann 1994)
    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: 'good' credit -> 1 (approved), 'bad' credit -> 0
    df["target"] = (df["class"].astype(str).str.strip() == "good").astype(int)
    df = df.drop(columns=["class"]).dropna()

    # Derive protected attributes.
    # German Credit's `personal_status` column conflates marital status and sex;
    # the codes starting "male" denote male applicants and the rest are female.
    df["sex_str"] = np.where(
        df["personal_status"].astype(str).str.strip().str.startswith("male"),
        "Male", "Female")
    df["age_group"] = np.where(df["age"] >= 40, "40+", "<40")

    # Light feature engineering: keep numerical features + a handful of binarised
    # categoricals so a logistic-regression baseline can run without one-hot blow-up.
    df["sex_bin"] = (df["sex_str"] == "Male").astype(int)
    df["owns_property_bin"] = df["property_magnitude"].astype(str).str.contains(
        "real estate|building society|life insurance", case=False, regex=True
    ).astype(int)
    df["foreign_bin"] = (df["foreign_worker"].astype(str).str.strip() == "yes").astype(int)
    df["job_skilled_bin"] = df["job"].astype(str).str.contains(
        "skilled", case=False, regex=False
    ).astype(int)

    # Map the savings-status ordinal categories to a numeric scale so we can
    # use it as an independent need proxy in the Equity diagnostic.
    savings_order = {
        "no known savings": 0,
        "<100": 1,
        "100<=X<500": 2,
        "500<=X<1000": 3,
        ">=1000": 4,
    }
    df["savings_num"] = df["savings_status"].astype(str).str.strip().map(savings_order)
    df["savings_num"] = df["savings_num"].fillna(df["savings_num"].median())

    feature_cols = [
        "duration", "credit_amount", "installment_commitment",
        "residence_since", "age", "existing_credits", "num_dependents",
        "sex_bin", "owns_property_bin", "foreign_bin", "job_skilled_bin",
    ]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "age_group"]].rename(columns={"sex_str": "sex"})

    print(f"Cohort: n={len(df)}, approval prevalence={y.mean():.3f}")
    print(f"Subgroups: sex {dict(demo['sex'].value_counts())}, "
          f"age {dict(demo['age_group'].value_counts())}")

    # 2. Train / test split (80/20), stratified
    X_tr, X_te, y_tr, y_te, d_tr, d_te, sav_tr, sav_te = train_test_split(
        X, y, demo, df["savings_num"].values,
        test_size=0.2, random_state=42, stratify=y
    )

    # 3. Logistic-regression baseline
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_tr, y_tr)

    scores_te = model.predict_proba(X_te)[:, 1]
    print(f"\nTest AUROC: {roc_auc_score(y_te, scores_te):.3f}")
    print(f"Test Brier:  {float(np.mean((scores_te - y_te) ** 2)):.3f}")

    # 4. RISED evaluation
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Noise +10%"},
        {"type": "unit_rescaling", "feature_index": 4, "factor": 1.05,
         "label": "Age +5%"},
        {"type": "unit_rescaling", "feature_index": 1, "factor": 1.05,
         "label": "Credit-amount +5%"},
    ]

    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 5. Equity with independent need proxy: savings-status is in the feature
    #    list as a categorical original column but we use the numeric encoding
    #    here as a domain-meaningful proxy for ability-to-repay independent
    #    of the credit-risk label.
    demo_with_need = d_te.reset_index(drop=True).copy()
    demo_with_need["savings_proxy"] = sav_te
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="savings_proxy"
    )

    # 6. EEOC-style adverse-impact ratios
    y_pred_te = (scores_te >= 0.5).astype(int)
    sr_ratio_sex = selection_rate_ratio(y_pred_te, d_te["sex"].reset_index(drop=True))
    sr_ratio_age = selection_rate_ratio(y_pred_te, d_te["age_group"].reset_index(drop=True))

    # 7. Scorecard
    print("\n=== RISED scorecard on Statlog German Credit (credit-risk framing) ===")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    print(f"  Inclusivity (adverse-impact ratio, four-fifths rule):")
    print(f"    sex  ratio = {sr_ratio_sex:.3f}  "
          f"{'PASS' if sr_ratio_sex >= 0.80 else 'FAIL'} (threshold 0.80)")
    print(f"    age  ratio = {sr_ratio_age:.3f}  "
          f"{'PASS' if sr_ratio_age >= 0.80 else 'FAIL'} (threshold 0.80)")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  "
          f"95% CI {tuple(round(x*100,1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None}")
    print(f"  Equity rho_need (y_true)        = "
          f"{report.equity.need_prediction_correlation:.4f}")
    print(f"  Equity rho_need (savings_proxy) = "
          f"{eq_independent.need_prediction_correlation:.4f}")
    print(f"  Deployability latency = "
          f"{report.deployability.mean_inference_latency_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
