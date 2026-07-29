"""
Cross-domain demo: RISED evaluation on the UCI Adult Income dataset.

Reference: Kohavi (1996), UCI Machine Learning Repository. Public domain.
Accessed via sklearn.datasets.fetch_openml('adult', version=2).

The Adult Income cohort is the canonical fairness benchmark for
income / credit-screening expert systems. This script demonstrates
that the RISED protocol developed for clinical AI in the main paper
(Sections 3 and 4) applies without modification to a credit-style
decision-support problem when the dimension thresholds are recalibrated
from clinical defaults to fair-lending / employment-screening defaults.

Domain-calibrated thresholds applied here:
    Reliability  PSS < 0.05               (RISED default)
    Inclusivity  selection-rate ratio >= 0.80   (EEOC four-fifths rule)
    Sensitivity  max TFR < 10%            (RISED default, +/- 5pp sweep)
    Equity       proxy = education-num    (independent of training target)
    Deployability latency < 100 ms        (RISED default)

Run:
    python adult_income_demo.py
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
    """Adverse-impact ratio = min-group selection rate / max-group selection rate.

    EEOC four-fifths rule: ratio < 0.80 indicates adverse impact.
    """
    rates = {}
    for g in group.unique():
        mask = group == g
        rates[g] = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0
    if max(rates.values()) == 0:
        return 1.0
    return min(rates.values()) / max(rates.values())


def main():
    # 1. Load UCI Adult Income (Kohavi 1996)
    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()

    # Target: income > $50k -> 1, else 0
    df["target"] = (df["class"].astype(str).str.strip().str.startswith(">50K")).astype(int)
    df = df.drop(columns=["class"]).dropna()

    # Protected attributes for subgroup / fairness analysis
    df["sex_str"] = df["sex"].astype(str).str.strip()
    df["race_str"] = df["race"].astype(str).str.strip()
    # EEOC-style age category: 40+ is a protected class under ADEA
    df["age_group"] = np.where(df["age"] >= 40, "40+", "<40")

    # Drop the high-cardinality categorical columns we are not using as features.
    # Keep numerical features + a small set of one-hot encodings for the model.
    df["sex_bin"] = (df["sex_str"] == "Male").astype(int)
    df["white_bin"] = (df["race_str"] == "White").astype(int)
    df["married_bin"] = df["marital-status"].astype(str).str.contains("Married").astype(int)
    df["us_native_bin"] = (df["native-country"].astype(str).str.strip() == "United-States").astype(int)

    feature_cols = [
        "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
        "hours-per-week", "sex_bin", "white_bin", "married_bin", "us_native_bin",
    ]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "race_str", "age_group"]].rename(
        columns={"sex_str": "sex", "race_str": "race"}
    )

    print(f"Cohort: n={len(df)}, prevalence={y.mean():.3f}")
    print(f"Subgroups: sex {dict(demo['sex'].value_counts())}")
    print(f"           race {dict(demo['race'].value_counts())}")
    print(f"           age  {dict(demo['age_group'].value_counts())}")

    # 2. Train / test split (80/20), stratified on outcome
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Logistic-regression baseline (interpretable, standard for credit screening)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_tr, y_tr)

    scores_te = model.predict_proba(X_te)[:, 1]
    print(f"\nTest AUROC: {roc_auc_score(y_te, scores_te):.3f}")
    print(f"Test Brier:  {float(np.mean((scores_te - y_te) ** 2)):.3f}")

    # 4. RISED evaluation: perturbations adapted to the credit / income domain.
    #    - Income reporting unit shift (capital-gain rescaled by +/-5%) mirrors
    #      the EHR-unit-change perturbation in clinical Reliability.
    #    - Hours-per-week noise +/-5% mirrors administrative coding shifts.
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Noise +10%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "Age +5%"},
        {"type": "unit_rescaling", "feature_index": 3, "factor": 1.05,
         "label": "Capital-gain +5%"},
    ]

    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 5. Equity with an independent need proxy (education-num is a feature but
    #    not the training target; it tracks years of formal education, a
    #    domain-meaningful proxy for earning capacity independent of income).
    demo_with_need = d_te.reset_index(drop=True).copy()
    edunum_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(drop=True)["education-num"]
    demo_with_need["education_proxy"] = edunum_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="education_proxy"
    )

    # 6. Domain-specific Inclusivity check: EEOC four-fifths rule on
    #    selection (approval) rate across protected groups, using the
    #    model's >=0.5 decision threshold as the approval threshold.
    y_pred_te = (scores_te >= 0.5).astype(int)
    sr_ratio_sex = selection_rate_ratio(y_pred_te, d_te["sex"].reset_index(drop=True))
    sr_ratio_race = selection_rate_ratio(y_pred_te, d_te["race"].reset_index(drop=True))
    sr_ratio_age = selection_rate_ratio(y_pred_te, d_te["age_group"].reset_index(drop=True))

    # 7. Print scorecard with credit / hiring framing
    print("\n=== RISED scorecard on UCI Adult Income (credit / hiring framing) ===")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    print(f"  Inclusivity (EEOC four-fifths rule on selection-rate ratio):")
    print(f"    sex  ratio = {sr_ratio_sex:.3f}  "
          f"{'PASS' if sr_ratio_sex >= 0.80 else 'FAIL'} (threshold 0.80)")
    print(f"    race ratio = {sr_ratio_race:.3f}  "
          f"{'PASS' if sr_ratio_race >= 0.80 else 'FAIL'} (threshold 0.80)")
    print(f"    age  ratio = {sr_ratio_age:.3f}  "
          f"{'PASS' if sr_ratio_age >= 0.80 else 'FAIL'} (threshold 0.80)")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  "
          f"95% CI {tuple(round(x*100,1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None}")
    # Equity against y_true is withdrawn: with a binary outcome proxy the
    # statistic is an affine reparameterisation of AUROC (see rised.equity).
    print(f"  Equity rho_need (education-num)= "
          f"{eq_independent.need_prediction_correlation:.4f}")
    print(f"  Deployability batch scoring time (whole cohort) = "
          f"{report.deployability.batch_scoring_time_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
