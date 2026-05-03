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
    """Demographic DataFrame aligned to the small_cohort fixture."""
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
def fitted_lr(small_cohort):
    """Logistic regression fitted on the small_cohort fixture."""
    X, y = small_cohort
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(X, y)
    return clf
