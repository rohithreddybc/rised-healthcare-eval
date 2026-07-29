"""
Cross-domain demo: RISED evaluation on the Folktables ACS-Income cohort.

Reference: Ding, Hardt, Miller, Schmidt. "Retiring Adult: New Datasets for
Fair Machine Learning." NeurIPS 2021.
    https://arxiv.org/abs/2108.04884
    https://github.com/socialfoundations/folktables

Folktables ACS-Income is the modern, NeurIPS-vetted replacement for the
UCI Adult Income cohort (Kohavi 1996). It exposes US Census ACS PUMS data
through a curated API with state-level partitioning, removes the
documented coding problems in Adult, and is the dataset the algorithmic
fairness community now defaults to. We include this demo as the
forward-looking cross-domain reference for the RISED protocol.

If `folktables` is not installed, the script falls back to a one-state
ACS extract bundled with the repository at
    examples/data/acs_income_ca_2018_sample.csv
so that the demo still runs in restricted-network environments.

Domain-calibrated thresholds applied here:
    Reliability  PSS < 0.05                (RISED default)
    Inclusivity  selection-rate ratio >= 0.80   (EEOC four-fifths rule)
    Sensitivity  max TFR < 10%             (RISED default, +/- 5pp sweep)
    Equity       proxy = SCHL (educational attainment, ordinal scale)
    Deployability latency < 100 ms         (RISED default)

Run:
    python folktables_acs_income_demo.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

import rised
from rised.equity import evaluate_equity


def _load_folktables_acs_income(state: str = "CA", year: int = 2018):
    """Load the ACS-Income subset via the `folktables` package if available.

    Returns (features_df, label_array, group_array) matching the Folktables
    contract: features are the ACSIncome feature list, label is income > $50k,
    group is RAC1P (race recoded).
    """
    from folktables import ACSDataSource, ACSIncome  # type: ignore
    data_source = ACSDataSource(
        survey_year=str(year), horizon="1-Year", survey="person")
    acs_data = data_source.get_data(states=[state], download=True)
    features, label, group = ACSIncome.df_to_pandas(acs_data)
    return features, label, group


def _load_local_fallback() -> pd.DataFrame:
    """Fallback: a bundled small ACS extract so the demo runs offline."""
    path = os.path.join(os.path.dirname(__file__), "data",
                        "acs_income_ca_2018_sample.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            "folktables not installed and bundled fallback CSV not found. "
            "Install folktables with `pip install folktables` to fetch ACS data."
        )
    return pd.read_csv(path)


def selection_rate_ratio(y_pred: np.ndarray, group: pd.Series) -> float:
    rates = {}
    for g in group.unique():
        mask = group == g
        rates[g] = float(y_pred[mask].mean()) if mask.sum() > 0 else 0.0
    if max(rates.values()) == 0:
        return 1.0
    return min(rates.values()) / max(rates.values())


def main():
    # 1. Load ACS-Income (Folktables, Ding et al. 2021).
    try:
        feat_df, y_arr, group_arr = _load_folktables_acs_income()
        df = feat_df.copy()
        # y_arr and group_arr come back as 2D arrays of shape (n, 1) from the
        # Folktables df_to_pandas helper; flatten before assigning to columns.
        df["target"] = np.asarray(y_arr).ravel().astype(int)
        df["RAC1P"] = np.asarray(group_arr).ravel()
        # The full California ACS-2018 PUMS has ~195k records; for the demo we
        # subsample to 20k (stratified on outcome) for tractable bootstrap
        # runtime. Downstream verdicts are stable at this sample size by the
        # power analysis in Appendix A.
        if len(df) > 20000:
            rng = np.random.RandomState(42)
            p1 = df[df["target"] == 1].sample(
                n=int(20000 * df["target"].mean()), random_state=rng)
            p0 = df[df["target"] == 0].sample(
                n=20000 - len(p1), random_state=rng)
            df = pd.concat([p0, p1]).reset_index(drop=True)
        used = "folktables (live ACS data, stratified 20k subsample)"
    except Exception as e:  # noqa: BLE001
        print(f"[folktables not available: {e}; falling back to bundled sample]")
        df = _load_local_fallback()
        used = "bundled fallback sample"

    # 2. Derive protected attributes for subgroup analysis.
    df["sex_str"] = np.where(df["SEX"].astype(int) == 1, "Male", "Female")
    # Race: 1 = White alone, 2 = Black alone (per ACS coding); collapse to binary
    # majority/minority for a worst-case adverse-impact test in this demo.
    df["race_bin"] = np.where(df["RAC1P"].astype(int) == 1, "White", "Non-White")
    df["age_group"] = np.where(df["AGEP"].astype(int) >= 40, "40+", "<40")

    feature_cols = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP",
                    "WKHP", "SEX", "RAC1P"]
    # Defensive: keep only present feature columns (bundled sample may differ)
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols + ["target"])

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "race_bin", "age_group"]].rename(
        columns={"sex_str": "sex", "race_bin": "race"}
    )

    print(f"Cohort source: {used}")
    print(f"Cohort: n={len(df)}, prevalence={y.mean():.3f}")
    print(f"Subgroups: sex {dict(demo['sex'].value_counts())}")
    print(f"           race {dict(demo['race'].value_counts())}")
    print(f"           age  {dict(demo['age_group'].value_counts())}")

    # 3. Train / test split (80/20), stratified on outcome
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Logistic-regression baseline (standard for income-classification audits)
    model = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    model.fit(X_tr, y_tr)

    scores_te = model.predict_proba(X_te)[:, 1]
    print(f"\nTest AUROC: {roc_auc_score(y_te, scores_te):.3f}")
    print(f"Test Brier:  {float(np.mean((scores_te - y_te) ** 2)):.3f}")

    # 5. RISED evaluation with ACS-style perturbations.
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Noise +10%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "Age +5%"},
        {"type": "unit_rescaling", "feature_index": feature_cols.index("WKHP")
         if "WKHP" in feature_cols else 0, "factor": 1.05,
         "label": "Hours-per-week +5%"},
    ]

    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000,
    )

    # 6. Equity with educational attainment (SCHL) as independent need proxy.
    demo_with_need = d_te.reset_index(drop=True).copy()
    if "SCHL" in feature_cols:
        schl_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(drop=True)["SCHL"]
        demo_with_need["schl_proxy"] = schl_te.values
        eq_independent = evaluate_equity(
            model, X_te, y_te, demo_with_need, need_column="schl_proxy"
        )
    else:
        eq_independent = None

    # 7. EEOC-style adverse-impact ratios at the >=0.5 decision cutoff
    y_pred_te = (scores_te >= 0.5).astype(int)
    sr_sex = selection_rate_ratio(y_pred_te, d_te["sex"].reset_index(drop=True))
    sr_race = selection_rate_ratio(y_pred_te, d_te["race"].reset_index(drop=True))
    sr_age = selection_rate_ratio(y_pred_te, d_te["age_group"].reset_index(drop=True))

    # 8. Scorecard
    print("\n=== RISED scorecard on Folktables ACS-Income (Ding et al. 2021) ===")
    print(f"  Reliability JSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    print(f"  Inclusivity (EEOC four-fifths rule on selection-rate ratio):")
    print(f"    sex  ratio = {sr_sex:.3f}  "
          f"{'PASS' if sr_sex >= 0.80 else 'FAIL'} (threshold 0.80)")
    print(f"    race ratio = {sr_race:.3f}  "
          f"{'PASS' if sr_race >= 0.80 else 'FAIL'} (threshold 0.80)")
    print(f"    age  ratio = {sr_age:.3f}  "
          f"{'PASS' if sr_age >= 0.80 else 'FAIL'} (threshold 0.80)")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  "
          f"95% CI {tuple(round(x*100,1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None}")
    # Equity against y_true is withdrawn: with a binary outcome proxy the
    # statistic is an affine reparameterisation of AUROC (see rised.equity).
    if eq_independent is not None:
        print(f"  Equity rho_need (SCHL proxy)  = "
              f"{eq_independent.need_prediction_correlation:.4f}")
    print(f"  Deployability batch scoring time (whole cohort) = "
          f"{report.deployability.batch_scoring_time_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
