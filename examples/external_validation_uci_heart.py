"""
External validation of the RISED Framework on the UCI Heart Disease dataset.

Reference: Detrano et al. (1989), Cleveland Clinic Foundation. Public domain.
Accessed via sklearn.datasets.fetch_openml('heart-disease', version=1).

This is a real (non-synthetic) tabular clinical cohort used here to
demonstrate that RISED's pass/fail signals are not artifacts of the
synthetic-cohort outcome derivation.

Run:
    python external_validation_uci_heart.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import rised
from rised.equity import evaluate_equity


def main():
    # 1. Load UCI Heart Disease (Cleveland)
    data = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto")
    df = data.frame.copy().dropna()
    df["sex_str"] = df["sex"].map({0: "F", 1: "M"})
    # Age tertiles for subgroup evaluation
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 50, 60, 200],
        labels=["<=50", "51-60", ">60"], include_lowest=True,
    ).astype(str)

    feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "age_group"]].rename(columns={"sex_str": "sex"})

    print(f"Cohort: n={len(df)}, prevalence={y.mean():.3f}")
    print(f"Subgroups: sex {dict(demo['sex'].value_counts())}, "
          f"age {dict(demo['age_group'].value_counts())}")

    # 2. Train / test split (80/20)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)

    # 3. Train XGBoost
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

    from sklearn.metrics import roc_auc_score
    scores_te = model.predict_proba(X_te)[:, 1]
    print(f"Test AUROC: {roc_auc_score(y_te, scores_te):.3f}")
    print(f"Test Brier:  {float(np.mean((scores_te - y_te) ** 2)):.3f}")

    # 4. RISED evaluation
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

    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 5. Equity with independent need proxy (use chol = serum cholesterol,
    #    which is a feature but not the training target — analogue of CCI).
    demo_with_need = d_te.reset_index(drop=True).copy()
    chol_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(drop=True)["chol"]
    demo_with_need["chol_proxy"] = chol_te.values
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_with_need, need_column="chol_proxy")

    # 6. Print scorecard
    print("\n=== RISED scorecard on UCI Heart Disease (real data) ===")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  "
          f"95% CI {tuple(round(x*100,1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None}")
    print(f"  Equity rho_need (y_true) = "
          f"{report.equity.need_prediction_correlation:.4f}")
    print(f"  Equity rho_need (chol)   = "
          f"{eq_independent.need_prediction_correlation:.4f}")
    print(f"  Deployability latency = "
          f"{report.deployability.mean_inference_latency_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
