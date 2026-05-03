"""
Tests for rised.datasets — cohort generation, feature matrix construction,
Charlson index computation, and baseline model training.
"""

import numpy as np
import pandas as pd
import pytest

from rised.datasets import (
    DEMOGRAPHIC_COLS,
    FEATURE_COLS,
    LABEL_COL,
    build_feature_matrix,
    charlson_comorbidity_index,
    generate_synthea_cohort,
    load_synthea_cohort,
    train_baseline_model,
)


def test_generate_shape():
    df = generate_synthea_cohort(n=100, random_state=0)
    assert df.shape[0] == 100
    assert df.shape[1] >= 25


def test_generate_required_columns():
    df = generate_synthea_cohort(n=50, random_state=0)
    for col in FEATURE_COLS:
        assert col in df.columns, f"Missing feature column: {col}"
    for col in DEMOGRAPHIC_COLS:
        assert col in df.columns, f"Missing demographic column: {col}"
    assert LABEL_COL in df.columns


def test_generate_prevalence_approximately_30pct():
    df = generate_synthea_cohort(n=2000, random_state=42)
    prevalence = df[LABEL_COL].mean()
    assert 0.25 <= prevalence <= 0.35, f"Prevalence {prevalence:.3f} outside expected range [0.25, 0.35]"


def test_generate_reproducible():
    df1 = generate_synthea_cohort(n=200, random_state=7)
    df2 = generate_synthea_cohort(n=200, random_state=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_generate_different_seeds_differ():
    df1 = generate_synthea_cohort(n=200, random_state=1)
    df2 = generate_synthea_cohort(n=200, random_state=2)
    assert not df1["age"].equals(df2["age"])


def test_build_feature_matrix_shapes():
    df = generate_synthea_cohort(n=150, random_state=0)
    X, y, demo = build_feature_matrix(df)
    assert X.shape == (150, len(FEATURE_COLS))
    assert len(y) == 150
    assert demo.shape == (150, len(DEMOGRAPHIC_COLS))


def test_charlson_known_input():
    df = pd.DataFrame({
        "has_ckd": [1, 0, 0],
        "has_cancer": [0, 1, 0],
        "has_metastatic": [0, 0, 1],
        "has_chf": [0, 0, 0],
    })
    cci = charlson_comorbidity_index(df)
    assert cci.tolist() == [2, 2, 4]


def test_train_baseline_model_predict_proba():
    df = generate_synthea_cohort(n=300, random_state=42)
    X, y, _ = build_feature_matrix(df)
    model = train_baseline_model(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (300, 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_load_synthea_cohort_uses_existing_csv(tmp_path):
    df_orig = generate_synthea_cohort(n=80, random_state=5)
    csv_path = tmp_path / "test_cohort.csv"
    df_orig.to_csv(csv_path, index=False)
    X, y, demo = load_synthea_cohort(csv_path=csv_path)
    assert len(X) == 80
    assert len(y) == 80
    assert len(demo) == 80
