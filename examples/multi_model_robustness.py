"""
Multi-model robustness check for RISED.

Trains three model classes on the same synthetic cohort and runs the full
RISED framework on each, to test whether the framework's pass/fail signals
are properties of the model or properties of XGBoost specifically.

Models: XGBoost, Logistic Regression, Random Forest.

Run:
    python multi_model_robustness.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import rised
from rised.datasets import load_synthea_cohort, FEATURE_COLS


def fit_models(X_tr, y_tr):
    models = {}

    # XGBoost (fallback to HistGBM)
    try:
        from xgboost import XGBClassifier
        m = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80,
            eval_metric="logloss", random_state=42, verbosity=0, seed=42,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
    m.fit(X_tr, y_tr)
    models["XGBoost"] = m

    # Logistic Regression with scaling
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=42)),
    ])
    lr.fit(X_tr, y_tr)
    models["LogReg"] = lr

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    models["RandomForest"] = rf

    return models


def main():
    X, y, demo = load_synthea_cohort()
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.20, random_state=42, stratify=y)

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

    models = fit_models(X_tr, y_tr)
    rows = []

    for name, model in models.items():
        scores_te = model.predict_proba(X_te)[:, 1]
        auroc = roc_auc_score(y_te, scores_te)
        brier = float(np.mean((scores_te - y_te) ** 2))

        t0 = time.perf_counter()
        report = rised.evaluate_all(
            model, X_te, y_te, d_te,
            perturbation_specs=perturbation_specs,
            random_state=42, n_bootstrap=1000,
        )
        elapsed = time.perf_counter() - t0

        max_tfr = max(report.sensitivity.threshold_flip_rates.values())
        rows.append({
            "model": name,
            "AUROC": round(auroc, 3),
            "Brier": round(brier, 3),
            "JSS": round(report.reliability.judge_sensitivity_score, 4),
            "JSS_CI": tuple(round(x, 4) for x in report.reliability.jss_ci) if report.reliability.jss_ci else None,
            "DeltaAUC": round(report.inclusivity.auc_parity_gap, 4),
            "DeltaAUC_CI": tuple(round(x, 4) for x in report.inclusivity.auc_gap_ci) if report.inclusivity.auc_gap_ci else None,
            "MaxTFR_pct": round(max_tfr * 100, 1),
            "MaxTFR_CI_pct": tuple(round(x*100, 1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None,
            "batch_scoring_ms": round(report.deployability.batch_scoring_time_ms, 2),
            "rised_eval_time_s": round(elapsed, 1),
        })

    df = pd.DataFrame(rows)
    print("\n=== Multi-model RISED comparison (synthetic cohort, n_test=2000) ===")
    print(df.to_string(index=False))
    print()

    # Summary: which dimensions fail across all models?
    print("=== Pass/fail pattern by dimension ===")
    for r in rows:
        print(f"\n{r['model']}:")
        print(f"  Reliability   {'FAIL' if r['JSS'] >= 0.05 else 'PASS'} (JSS={r['JSS']})")
        print(f"  Inclusivity   {'FAIL' if r['DeltaAUC'] >= 0.05 else 'PASS'} (DeltaAUC={r['DeltaAUC']})")
        print(f"  Sensitivity   {'FAIL' if r['MaxTFR_pct'] >= 10 else 'PASS'} (max TFR={r['MaxTFR_pct']}%)")
        print(f"  Equity(y)     {'PASS' if r['rho_need(y_true)'] >= 0.70 else 'FAIL'} (rho={r['rho_need(y_true)']})")
        print(f"  Deployability {'PASS' if r['latency_ms'] <= 500 else 'FAIL'} (latency={r['latency_ms']}ms)")

    return df


if __name__ == "__main__":
    main()
