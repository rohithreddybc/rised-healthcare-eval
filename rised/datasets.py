"""
Dataset loading and preprocessing utilities for RISED demonstrations.

Primary dataset: synthetic patient cohort generated with Synthea
(Walonoski et al., 2018, JAMIA 25(3):230-238).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def load_synthea_cohort(
    csv_path: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load the preprocessed Synthea synthetic patient cohort.

    Parameters
    ----------
    csv_path : str or Path, optional
        Path to the cohort CSV. If None, loads the bundled example file
        from ``examples/synthetic_cohort_5k.csv``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (one row per patient).
    y : pd.Series
        Binary outcome label (high-risk = 1).
    demographic_df : pd.DataFrame
        Demographic and subgroup columns aligned to X.
    """
    raise NotImplementedError("load_synthea_cohort() will be implemented in Session 4.")


def build_feature_matrix(patients_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features from a raw Synthea patients DataFrame.

    Includes: Charlson Comorbidity Index components (Charlson 1987;
    Quan et al. 2011), age, sex, encounter frequency, condition counts.

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    raise NotImplementedError("build_feature_matrix() will be implemented in Session 4.")


def charlson_comorbidity_index(conditions_df: pd.DataFrame) -> pd.Series:
    """
    Compute the Charlson Comorbidity Index score per patient.

    References
    ----------
    Charlson et al. (1987). Journal of Chronic Diseases, 40(5):373-383.
    Quan et al. (2011). American Journal of Epidemiology, 173(6):676-682.
    """
    raise NotImplementedError("charlson_comorbidity_index() will be implemented in Session 4.")
