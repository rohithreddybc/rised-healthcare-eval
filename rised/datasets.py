"""
Dataset generation and loading utilities for RISED demonstrations.

Primary dataset: synthetic patient cohort inspired by Synthea
(Walonoski et al., 2018, JAMIA 25(3):230-238).

All data generated here is entirely synthetic. No real patient data is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ── Column definitions ────────────────────────────────────────────────────────

FEATURE_COLS = [
    "age", "sex_male", "cci_score",
    "has_hypertension", "has_diabetes", "has_chf", "has_ckd", "has_copd",
    "has_mi", "has_cvd", "has_dementia", "has_cancer", "has_metastatic",
    "prior_hosp_count", "ed_visits_count", "bmi", "adi_score",
    "ins_medicare", "ins_medicaid", "ins_private",
]

DEMOGRAPHIC_COLS = ["age_group", "sex", "race", "insurance"]
LABEL_COL = "high_need"

# Charlson weights (Charlson 1987; Quan 2011 updates)
_CCI_WEIGHTS = {
    "has_chf": 1, "has_copd": 1, "has_cvd": 1,
    "has_dementia": 1, "has_mi": 1, "has_diabetes": 1,
    "has_ckd": 2, "has_cancer": 2, "has_metastatic": 4,
}

# Default bundled CSV path
_DEFAULT_CSV = Path(__file__).parent.parent / "examples" / "synthetic_cohort_10k.csv"


# ── Public API ────────────────────────────────────────────────────────────────

def load_synthea_cohort(
    csv_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the preprocessed Synthea-inspired synthetic patient cohort.

    If ``csv_path`` is None and the bundled file does not yet exist, the
    cohort is generated on-the-fly with ``generate_synthea_cohort()``.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the cohort CSV. Defaults to
        ``examples/synthetic_cohort_5k.csv`` relative to the repo root.

    Returns
    -------
    X : pd.DataFrame  (n_patients × n_features)
    y : pd.Series     binary high-need label
    demographic_df : pd.DataFrame  demographic/subgroup columns
    """
    path = Path(csv_path) if csv_path is not None else _DEFAULT_CSV

    if not path.exists():
        df = generate_synthea_cohort()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    else:
        df = pd.read_csv(path)

    return build_feature_matrix(df)


def generate_synthea_cohort(
    n: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic patient cohort with Synthea-inspired demographics
    and clinical features.

    Generates a population weighted toward Medicare/Medicaid enrollees
    (older adults, higher comorbidity burden) with realistic prevalence
    rates for common chronic conditions.

    Parameters
    ----------
    n : int
        Number of synthetic patients. Default 10 000.
    random_state : int
        Seed for reproducibility. Default 42.

    Returns
    -------
    pd.DataFrame
        One row per patient. Includes feature columns, demographic columns,
        and the binary ``high_need`` label.
    """
    rng = np.random.default_rng(random_state)

    # ── Age distribution (Medicare/Medicaid-weighted) ─────────────────────
    age_group_idx = rng.choice([0, 1, 2, 3], n, p=[0.18, 0.25, 0.28, 0.29])
    age_ranges = [(18, 44), (45, 64), (65, 74), (75, 95)]
    age = np.array([int(rng.integers(*age_ranges[g])) for g in age_group_idx])
    age_group = np.array(["18-44", "45-64", "65-74", "75+"])[age_group_idx]

    # ── Demographics ──────────────────────────────────────────────────────
    sex = rng.choice(["M", "F"], n, p=[0.45, 0.55])
    race = rng.choice(
        ["White", "Black", "Hispanic", "Asian", "Other"],
        n, p=[0.64, 0.13, 0.13, 0.06, 0.04],
    )

    # Insurance correlated with age group
    _ins_probs = {
        0: (["Medicaid", "Private", "Uninsured"], [0.30, 0.55, 0.15]),
        1: (["Medicare", "Medicaid", "Private", "Uninsured"], [0.05, 0.25, 0.55, 0.15]),
        2: (["Medicare", "Medicaid", "Private", "Uninsured"], [0.80, 0.05, 0.10, 0.05]),
        3: (["Medicare", "Medicaid", "Private", "Uninsured"], [0.85, 0.05, 0.08, 0.02]),
    }
    insurance = np.array([
        rng.choice(_ins_probs[g][0], p=_ins_probs[g][1])
        for g in age_group_idx
    ])

    # ── Comorbidities (age-correlated prevalence) ─────────────────────────
    age_f = np.clip((age - 40) / 40.0, 0.0, 1.0)

    def _bern(base: float, slope: float = 0.0) -> np.ndarray:
        return (rng.random(n) < np.clip(base + slope * age_f, 0, 0.99)).astype(int)

    has_hypertension = _bern(0.30, 0.30)
    has_diabetes     = _bern(0.08, 0.12)
    has_chf          = _bern(0.02, 0.12)
    has_ckd          = _bern(0.08, 0.15)
    has_copd         = _bern(0.03, 0.10)
    has_mi           = _bern(0.02, 0.08)
    has_cvd          = _bern(0.01, 0.07)
    has_dementia     = _bern(0.01, 0.15)
    has_cancer       = _bern(0.02, 0.08)
    has_metastatic   = (_bern(0.005, 0.015) & has_cancer).astype(int)

    # ── Charlson Comorbidity Index ────────────────────────────────────────
    cci = (
        has_chf * 1 + has_copd * 1 + has_cvd * 1 + has_dementia * 1
        + has_mi * 1 + has_diabetes * 1
        + has_ckd * 2 + has_cancer * 2 + has_metastatic * 4
    )

    # ── Utilization ───────────────────────────────────────────────────────
    hosp_prob = np.clip(0.10 + 0.50 * age_f + 0.20 * (cci > 2), 0, 0.90)
    prior_hosp = np.clip(
        (rng.integers(0, 6, n) * (rng.random(n) < hosp_prob)).astype(int), 0, 5
    )
    ed_prob = np.clip(0.15 + 0.40 * age_f + 0.15 * has_hypertension, 0, 0.90)
    ed_visits = np.clip(
        (rng.integers(0, 6, n) * (rng.random(n) < ed_prob)).astype(int), 0, 5
    )

    # ── Physical / social determinants ───────────────────────────────────
    bmi = np.clip(rng.normal(28.5, 6.0, n), 15.0, 55.0).round(1)
    adi_score = np.clip(rng.normal(50.0, 25.0, n), 1.0, 100.0).round(1)

    # ── Outcome: high-need (top-30% risk by logistic score) ───────────────
    log_odds = (
        -2.0
        + 0.03 * (age - 60)
        + 0.6  * has_diabetes
        + 1.2  * has_chf
        + 0.7  * has_ckd
        + 0.5  * has_copd
        + 0.4  * has_mi
        + 0.3  * (cci / 5.0)
        + 0.8  * (prior_hosp / 3.0)
        + 0.3  * (ed_visits / 3.0)
        + 0.2  * (adi_score / 100.0)
        + rng.normal(0.0, 0.5, n)
    )
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    high_need = (prob >= np.percentile(prob, 70)).astype(int)  # 30% prevalence

    # ── Assemble DataFrame ────────────────────────────────────────────────
    return pd.DataFrame({
        "patient_id":       [f"P{i:05d}" for i in range(1, n + 1)],
        # Demographic columns
        "age_group":        age_group,
        "sex":              sex,
        "race":             race,
        "insurance":        insurance,
        # Feature columns
        "age":              age,
        "sex_male":         (sex == "M").astype(int),
        "cci_score":        cci,
        "has_hypertension": has_hypertension,
        "has_diabetes":     has_diabetes,
        "has_chf":          has_chf,
        "has_ckd":          has_ckd,
        "has_copd":         has_copd,
        "has_mi":           has_mi,
        "has_cvd":          has_cvd,
        "has_dementia":     has_dementia,
        "has_cancer":       has_cancer,
        "has_metastatic":   has_metastatic,
        "prior_hosp_count": prior_hosp,
        "ed_visits_count":  ed_visits,
        "bmi":              bmi,
        "adi_score":        adi_score,
        "ins_medicare":     (insurance == "Medicare").astype(int),
        "ins_medicaid":     (insurance == "Medicaid").astype(int),
        "ins_private":      (insurance == "Private").astype(int),
        # Outcome
        LABEL_COL:          high_need,
    })


def build_feature_matrix(
    patients_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Split a patients DataFrame into (X, y, demographic_df).

    Parameters
    ----------
    patients_df : pd.DataFrame
        Output of ``generate_synthea_cohort()`` or a compatible CSV.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    demographic_df : pd.DataFrame
    """
    X = patients_df[FEATURE_COLS].copy().reset_index(drop=True)
    y = patients_df[LABEL_COL].copy().reset_index(drop=True)
    y.name = "high_need"
    demographic_df = patients_df[DEMOGRAPHIC_COLS].copy().reset_index(drop=True)
    return X, y, demographic_df


def charlson_comorbidity_index(df: pd.DataFrame) -> pd.Series:
    """
    Compute the Charlson Comorbidity Index score per patient.

    Applies the weights from Charlson et al. (1987) with updates
    from Quan et al. (2011) to binary comorbidity flag columns present
    in ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least some of the columns:
        has_chf, has_copd, has_cvd, has_dementia, has_mi, has_diabetes,
        has_ckd, has_cancer, has_metastatic.

    Returns
    -------
    pd.Series  (dtype int)
        CCI score per row.
    """
    cci = pd.Series(0, index=df.index, name="cci_score", dtype=int)
    for col, weight in _CCI_WEIGHTS.items():
        if col in df.columns:
            cci = cci + df[col].fillna(0).astype(int) * weight
    return cci


def train_baseline_model(X_train, y_train):
    """
    Train an XGBoost gradient-boosted classifier on the given training data.

    Falls back to sklearn's HistGradientBoostingClassifier if XGBoost
    is not installed.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
    y_train : array-like of shape (n_samples,)

    Returns
    -------
    Fitted classifier with a ``predict_proba`` method.
    """
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )
    model.fit(X_train, y_train)
    return model
