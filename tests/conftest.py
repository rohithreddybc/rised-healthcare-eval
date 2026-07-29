"""
Shared pytest fixtures for RISED test suite.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def small_cohort():
    """200-patient synthetic cohort with 10 features and binary outcome."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(rng.standard_normal((n, 10)), columns=[f"f{i}" for i in range(10)])
    y = pd.Series((rng.random(n) > 0.7).astype(int), name="label")
    return X, y


@pytest.fixture
def demographic_df(small_cohort):
    """Demographic DataFrame aligned to the small_cohort fixture.

    Every level of every column has n >= 30 under seed 0, so the default
    min_subgroup_n excludes nothing unless a test arranges it.
    """
    X, y = small_cohort
    rng = np.random.default_rng(0)
    n = len(X)
    return pd.DataFrame(
        {
            "race": rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], n),
            "sex": rng.choice(["M", "F"], n),
            "age_group": rng.choice(["18-44", "45-64", "65+"], n),
        }
    )


@pytest.fixture
def demographic_with_need(demographic_df, small_cohort):
    """Demographics plus an independent clinical-need proxy.

    ``comorbidity`` is drawn independently of the outcome, so it is a valid
    Equity proxy. ``outcome_copy`` and ``outcome_rescaled`` are deliberately
    derived from ``y_true`` so tests can assert that such proxies are rejected.
    """
    X, y = small_cohort
    rng = np.random.default_rng(7)
    n = len(X)
    out = demographic_df.copy()
    out["comorbidity"] = rng.integers(0, 10, size=n).astype(float)
    out["outcome_copy"] = np.asarray(y, dtype=float)
    out["outcome_rescaled"] = np.asarray(y, dtype=float) * 3.0 + 1.0
    return out


@pytest.fixture
def fitted_lr(small_cohort):
    """Logistic regression fitted on the small_cohort fixture."""
    X, y = small_cohort
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture
def clustered_cohort():
    """Cohort whose rows are repeated encounters on a smaller set of patients."""
    rng = np.random.default_rng(11)
    n_patients = 60
    rows_per_patient = rng.integers(1, 5, size=n_patients)
    groups = np.repeat(np.arange(n_patients), rows_per_patient)
    n = len(groups)
    patient_effect = rng.standard_normal(n_patients)[groups]
    X = pd.DataFrame(
        rng.standard_normal((n, 6)) + patient_effect[:, None],
        columns=[f"f{i}" for i in range(6)],
    )
    y = pd.Series((rng.random(n) + 0.2 * patient_effect > 0.6).astype(int))
    clf = LogisticRegression(max_iter=500, random_state=0).fit(X, y)
    demo = pd.DataFrame({"site": rng.choice(["A", "B"], n)})
    return X, y, groups, demo, clf


@pytest.fixture
def mixed_type_cohort():
    """Cohort mixing continuous, binary and categorical columns.

    Column layout: 0-1 continuous, 2 binary, 3 categorical (5 integer levels).
    """
    rng = np.random.default_rng(5)
    n = 240
    cont = rng.standard_normal((n, 2)) * 10.0 + 50.0
    binary = (rng.random(n) > 0.5).astype(float)
    categorical = rng.integers(0, 5, size=n).astype(float)
    X = np.column_stack([cont, binary, categorical])
    y = (rng.random(n) > 0.6).astype(int)
    clf = LogisticRegression(max_iter=500, random_state=0).fit(X, y)
    names = ["lab_a", "lab_b", "smoker", "severity_code"]
    return X, y, names, clf


@pytest.fixture
def low_dimensional_cohort():
    """Cohort with d = 3, where the old top-3 explanation metrics were 1.0."""
    rng = np.random.default_rng(3)
    n = 200
    X = rng.standard_normal((n, 3))
    y = (X[:, 0] + rng.standard_normal(n) * 0.5 > 0).astype(int)
    clf = LogisticRegression(max_iter=500, random_state=0).fit(X, y)
    return X, y, clf
