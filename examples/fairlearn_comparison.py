"""
Head-to-head comparison: RISED vs Fairlearn on the same model and cohort.

Runs the standard Fairlearn fairness metrics on the synthetic cohort and
contrasts what each tool detects.

Run:
    python fairlearn_comparison.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from rised.datasets import load_synthea_cohort, FEATURE_COLS


def main():
    X, y, demo = load_synthea_cohort()
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.20, random_state=42, stratify=y)

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
    preds_te = (scores_te >= 0.5).astype(int)

    # --- Fairlearn metrics ---
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        false_positive_rate,
        false_negative_rate,
        selection_rate,
    )
    from sklearn.metrics import accuracy_score, roc_auc_score

    sensitive = d_te["race"].values

    mf = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "FPR": false_positive_rate,
            "FNR": false_negative_rate,
        },
        y_true=y_te, y_pred=preds_te, sensitive_features=sensitive,
    )
    print("=== Fairlearn metrics by race ===")
    print(mf.by_group)
    print()

    dpd = demographic_parity_difference(
        y_true=y_te, y_pred=preds_te, sensitive_features=sensitive)
    eod = equalized_odds_difference(
        y_true=y_te, y_pred=preds_te, sensitive_features=sensitive)
    print(f"Demographic parity difference (race): {dpd:.4f}")
    print(f"Equalized odds difference     (race): {eod:.4f}")
    print()

    # Subgroup AUC (equivalent to RISED Inclusivity, computed via Fairlearn)
    def _auc(y_true, y_score):
        if len(set(y_true)) < 2:
            return float("nan")
        return roc_auc_score(y_true, y_score)

    mf_auc = MetricFrame(
        metrics=_auc, y_true=y_te, y_pred=scores_te,
        sensitive_features=sensitive,
    )
    print("Subgroup AUC by race (Fairlearn):")
    print(mf_auc.by_group)
    print(f"AUC parity gap (max-min): {mf_auc.by_group.max() - mf_auc.by_group.min():.4f}")
    print()

    # --- Comparison narrative ---
    print("=== What RISED catches that Fairlearn does NOT ===")
    print(" * Reliability: input-perturbation flip rate (JSS)")
    print(" * Sensitivity: threshold-shift flip rate")
    print(" * Equity: need-prediction correlation under independent proxy")
    print(" * Deployability: inference latency + SHAP explanation faithfulness")
    print()
    print("=== What Fairlearn provides that RISED's Inclusivity does not ===")
    print(" * Demographic parity, equalized odds, selection rate by group")
    print(" * Mitigation algorithms (ExponentiatedGradient, ThresholdOptimizer)")
    print(" * Group-level FPR / FNR breakdown")
    print()
    print("Conclusion: RISED's Inclusivity dimension overlaps with Fairlearn on")
    print("subgroup AUC, but the other four RISED dimensions (Reliability,")
    print("Sensitivity, Equity-via-need-proxy, Deployability) are not measured")
    print("by Fairlearn. The two tools are complementary.")


if __name__ == "__main__":
    main()
