"""
Cohort loaders for the 0.1.0 -> 0.2.0 recomputation.

Every loader reproduces the data preparation of the corresponding script in
``examples/`` *exactly* -- same source, same cleaning, same feature list, same
``train_test_split(test_size=0.2, random_state=42, stratify=y)``, same
estimator and hyperparameters. The preparation is transcribed here rather than
imported because each example script buries it inside a monolithic ``main()``
that also runs the evaluation and prints a report.

The one deliberate departure is Diabetes 130, where ``patient_nbr`` is retained
and the split becomes a group split. See :func:`load_diabetes130`.

All loaders run offline from the caches already present in the working tree:
``~/scikit_learn_data/openml`` (UCI Heart, Diabetes 130, Adult, German Credit),
``examples/adult24.csv`` (NHIS 2024), ``nhis_cache/adult23.csv`` (NHIS 2023),
``nhanes_cache/*.xpt`` (NHANES 2021-2023), ``brfss_cache/LLCP2024.XPT``
(BRFSS 2024) and ``data/2018/1-Year/psam_p06.csv`` (Folktables ACS).
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

SEED = 42
TEST_SIZE = 0.2


# ── Model builders (verbatim from the example scripts) ───────────────────────
def _xgb():
    """The XGBoost baseline used by every clinical cohort script."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.80,
        eval_metric="logloss", random_state=42, verbosity=0, seed=42,
    )


def _logreg():
    """The logistic-regression baseline used by the cross-domain demos."""
    return Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
    ])


def _yn(s: pd.Series) -> pd.Series:
    """Survey yes/no recode: 1=Yes->1.0, 2=No->0.0, everything else -> NaN."""
    return s.map({1: 1.0, 2: 0.0}).astype(float)


def _fit(model, X_tr, y_tr):
    model.fit(X_tr, y_tr)
    return model


def _pack(
    name, X_tr, X_te, y_tr, y_te, d_tr, d_te, feature_cols, model,
    n_total, groups_te=None, need_values=None, need_name=None, extra=None,
) -> Dict[str, Any]:
    out = {
        "name": name,
        "X_train": X_tr, "X_test": X_te,
        "y_train": np.asarray(y_tr).astype(int),
        "y_test": np.asarray(y_te).astype(int),
        "demo_train": d_tr.reset_index(drop=True),
        "demo_test": d_te.reset_index(drop=True),
        "feature_names": list(feature_cols),
        "model": model,
        "n_total": int(n_total),
        "groups_test": groups_te,
    }
    demo_test = out["demo_test"].copy()
    if need_values is not None and need_name is not None:
        demo_test[need_name] = np.asarray(need_values)
        out["need_column"] = need_name
    else:
        out["need_column"] = None
    out["demo_test_with_need"] = demo_test
    #: Columns that form the Inclusivity partitions (the need proxy must not).
    out["subgroup_columns"] = list(out["demo_test"].columns)
    if extra:
        out.update(extra)
    return out


# ── 1. Synthetic baseline ────────────────────────────────────────────────────
def load_synthetic() -> Dict[str, Any]:
    """The 10,000-patient Synthea-inspired cohort (README Quickstart)."""
    from rised.datasets import load_synthea_cohort, train_baseline_model

    X, y, demo = load_synthea_cohort()
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.20, random_state=42, stratify=y)
    model = train_baseline_model(X_tr, y_tr)

    # Charlson comorbidity index: a measured clinical quantity, not derived
    # from the outcome. It is however also a model input feature (see the
    # proxy-validity note in the report).
    cci_te = pd.DataFrame(
        np.asarray(X_te), columns=list(X.columns)
    ).reset_index(drop=True)["cci_score"]

    return _pack(
        "synthetic", np.asarray(X_tr), np.asarray(X_te), y_tr, y_te,
        d_tr, d_te, list(X.columns), model, len(X),
        need_values=cci_te.values, need_name="cci_proxy",
    )


# ── 2. UCI Heart Disease (Cleveland) ─────────────────────────────────────────
def load_uci_heart() -> Dict[str, Any]:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="heart-disease", version=1, as_frame=True, parser="auto")
    df = data.frame.copy().dropna()
    df["sex_str"] = df["sex"].map({0: "F", 1: "M"})
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 50, 60, 200],
        labels=["<=50", "51-60", ">60"], include_lowest=True,
    ).astype(str)

    feature_cols = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                    "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "age_group"]].rename(columns={"sex_str": "sex"})

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_xgb(), X_tr, y_tr)

    chol_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(drop=True)["chol"]
    return _pack("uci_heart", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=chol_te.values, need_name="chol_proxy")


# ── 3. Diabetes 130-US Hospitals (grouped on patient_nbr) ────────────────────
def load_diabetes130() -> Dict[str, Any]:
    """Diabetes 130 with ``patient_nbr`` RETAINED and a group split.

    The example script never selects ``patient_nbr``, so the patient identity is
    discarded before the split and 42.12% of test rows belong to a patient who
    also appears in training (measured in ``verification/verify_p6.py``). Here
    the column is kept, the split is a ``GroupShuffleSplit`` on it, and the
    group vector is handed to RISED so the bootstrap resamples whole patients
    and the jackknife deletes whole patients.

    ``GroupShuffleSplit`` cannot stratify, so test prevalence drifts slightly
    from the row-level split; that drift is reported.

    The original leaky row-level split is also returned (under the ``row_*``
    keys) so the published 0.1.0 figures remain traceable.
    """
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="Diabetes130US", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)

    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses",
    ]
    df["A1Cresult_encoded"] = df["A1Cresult"].astype(str).map(
        {"None": 0, "Norm": 1, ">7": 2, ">8": 3, "nan": 0}).fillna(0)
    df["max_glu_serum_encoded"] = df["max_glu_serum"].astype(str).map(
        {"None": 0, "Norm": 1, ">200": 2, ">300": 3, "nan": 0}).fillna(0)
    df["change_encoded"] = (df["change"] == "Ch").astype(int)
    df["diabetesMed_encoded"] = (df["diabetesMed"] == "Yes").astype(int)
    df["insulin_used"] = (df["insulin"].astype(str) != "No").astype(int)

    feature_cols = numeric_cols + [
        "A1Cresult_encoded", "max_glu_serum_encoded",
        "change_encoded", "diabetesMed_encoded", "insulin_used",
    ]
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    df["age_numeric"] = df["age"].map(age_map).fillna(55).astype(float)
    feature_cols = ["age_numeric"] + feature_cols

    df = df.dropna(subset=feature_cols + ["target", "race", "gender", "age"])
    df = df[df["race"] != "?"]
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df.reset_index(drop=True)

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["race", "gender", "age"]].copy().rename(columns={"age": "age_group"})
    for c in demo.columns:
        demo[c] = demo[c].astype(str)

    # ── the retained identity ────────────────────────────────────────────────
    pid = df["patient_nbr"].values

    # Primary: group split on patient_nbr.
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(gss.split(X, y, groups=pid))
    X_tr, X_te = X[tr], X[te]
    y_tr, y_te = y[tr], y[te]
    d_tr, d_te = demo.iloc[tr], demo.iloc[te]
    pid_te = pid[te]
    model = _fit(_xgb(), X_tr, y_tr)

    n_inp_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["number_inpatient"]

    # Secondary: the original leaky row-level split, for traceability only.
    # Split the index array so the patient ids of the row-level test set stay
    # recoverable and the leakage can be measured rather than asserted.
    idx = np.arange(len(y))
    rtr, rte = train_test_split(
        idx, test_size=0.2, random_state=42, stratify=y)
    Xr_tr, Xr_te = X[rtr], X[rte]
    yr_tr, yr_te = y[rtr], y[rte]
    dr_te = demo.iloc[rte]
    row_model = _fit(_xgb(), Xr_tr, yr_tr)
    row_leak = float(np.isin(pid[rte], np.unique(pid[rtr])).mean())

    n_patients = int(len(np.unique(pid)))
    leaked = float(np.isin(pid_te, np.unique(pid[tr])).mean())

    return _pack(
        "diabetes130", X_tr, X_te, y_tr, y_te, d_tr, d_te,
        feature_cols, model, len(df),
        groups_te=pid_te,
        need_values=n_inp_te.values, need_name="n_inpatient_proxy",
        extra={
            "n_patients": n_patients,
            "n_test_patients": int(len(np.unique(pid_te))),
            "group_split_test_row_leakage": leaked,
            "row_split": {
                "X_train": Xr_tr, "X_test": Xr_te,
                "y_train": yr_tr, "y_test": yr_te,
                "demo_test": dr_te.reset_index(drop=True),
                "model": row_model,
                "row_leakage_fraction": row_leak,
            },
        },
    )


# ── 4. NHIS 2024 (cardiovascular) ────────────────────────────────────────────
def load_nhis2024() -> Dict[str, Any]:
    path = Path(os.environ.get("NHIS_PATH", REPO / "examples" / "adult24.csv"))
    if not path.exists():
        raise FileNotFoundError(f"NHIS 2024 file not found at {path}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper() for c in df.columns]

    RACE_LABELS = {1: "Hispanic", 2: "NH-White", 3: "NH-Black", 4: "NH-Asian",
                   5: "NH-AIAN", 6: "NH-AIAN+other", 7: "NH-Other/Multi"}
    AGE_BUCKETS = [(18, 25, "18-24"), (25, 35, "25-34"), (35, 45, "35-44"),
                   (45, 55, "45-54"), (55, 65, "55-64"), (65, 200, "65+")]

    def _age_bucket(age):
        if pd.isna(age):
            return np.nan
        for lo, hi, label in AGE_BUCKETS:
            if lo <= age < hi:
                return label
        return np.nan

    chdev = _yn(df.get("CHDEV_A", pd.Series(np.nan, index=df.index)))
    miev = _yn(df.get("MIEV_A", pd.Series(np.nan, index=df.index)))
    df["target"] = ((chdev == 1) | (miev == 1)).astype(int)
    df.loc[chdev.isna() & miev.isna(), "target"] = np.nan

    df["age_group"] = df["AGEP_A"].apply(_age_bucket)
    df["sex"] = df["SEX_A"].map({1: "M", 2: "F"})
    df["race"] = df["HISPALLP_A"].map(RACE_LABELS)
    rat = df.get("RATCAT_A")
    if rat is not None:
        df["income"] = pd.cut(
            rat.where(rat <= 14),
            bins=[0, 3.5, 6.5, 9.5, 12.5, 14.5],
            labels=["<$35K", "$35-50K", "$50-75K", "$75-100K", ">=$100K"],
        ).astype(str).replace({"nan": np.nan})
    else:
        df["income"] = np.nan
    df["insurance"] = df["NOTCOV_A"].map({1: "Uninsured", 2: "Insured"})

    df["age_numeric"] = df["AGEP_A"].where(df["AGEP_A"] <= 85)
    df["sex_male"] = (df["SEX_A"] == 1).astype(float)
    df["bmi_cat"] = df["BMICAT_A"].where(df["BMICAT_A"] <= 4)
    df["genhlth"] = df["PHSTAT_A"].where(df["PHSTAT_A"] <= 5)
    df["smoker"] = _yn(df["SMKEV_A"])
    df["current_smoker"] = (df["SMKCIGST_A"].isin([1, 2])).astype(float)
    df["heavy_drink"] = _yn(df["DRKHVY12M_A"])
    df["phys_active"] = _yn(df["PA18_02R_A"])
    df["hypertension"] = _yn(df["HYPEV_A"])
    df["high_chol"] = _yn(df["CHLEV_A"])
    dibev_recoded = df["DIBEV_A"].map(
        {1: 1, 2: 2, 3: 2, 7: np.nan, 8: np.nan, 9: np.nan})
    df["diabetes"] = _yn(dibev_recoded)
    df["asthma"] = _yn(df["ASEV_A"])
    df["stroke"] = _yn(df["STREV_A"])
    df["copd"] = _yn(df["COPDEV_A"])
    df["arthritis"] = _yn(df["ARTHEV_A"])
    df["depression"] = _yn(df["DEPEV_A"])
    df["kidney"] = _yn(df["KIDWEAKEV_A"])
    df["medcost"] = _yn(df["MEDDL12M_A"])
    df["usual_care"] = _yn(df["USUALPL_A"])

    feature_cols = [
        "age_numeric", "sex_male", "bmi_cat", "genhlth",
        "smoker", "current_smoker", "heavy_drink", "phys_active",
        "hypertension", "high_chol", "diabetes",
        "asthma", "stroke", "copd", "arthritis", "depression", "kidney",
        "medcost", "usual_care",
    ]
    demo_cols = ["age_group", "sex", "race", "income", "insurance"]
    df = df.dropna(subset=["target"] + demo_cols + feature_cols).copy()

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_xgb(), X_tr, y_tr)

    genhlth_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["genhlth"]
    return _pack("nhis2024", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=genhlth_te.values, need_name="genhlth_proxy")


# ── 5. NHIS 2023 (diabetes) ──────────────────────────────────────────────────
def load_nhis2023() -> Dict[str, Any]:
    cache = Path(os.environ.get("NHIS_CACHE_DIR", REPO / "nhis_cache")) / "adult23.csv"
    if not cache.exists():
        raise FileNotFoundError(f"NHIS 2023 cache not found at {cache}")
    df = pd.read_csv(cache, low_memory=False)
    df.columns = [c.upper() for c in df.columns]

    df["target"] = np.where(df["DIBEV_A"] == 1, 1,
                            np.where(df["DIBEV_A"] == 2, 0, np.nan))

    RACE_MAP = {1: "Hispanic", 2: "NH-White", 3: "NH-Black", 4: "NH-Asian",
                5: "NH-AIAN", 6: "NH-AIAN+other", 7: "NH-Other/Multi"}
    AGE_BUCKETS = [(18, 35, "18-34"), (35, 50, "35-49"),
                   (50, 65, "50-64"), (65, 200, "65+")]

    def _age_group(age):
        if pd.isna(age):
            return np.nan
        for lo, hi, label in AGE_BUCKETS:
            if lo <= age < hi:
                return label
        return np.nan

    df["age_group"] = df["AGEP_A"].apply(_age_group)
    df["sex"] = df["SEX_A"].map({1: "Male", 2: "Female"})
    df["race"] = df["HISPALLP_A"].map(RACE_MAP)
    df["insured"] = df["NOTCOV_A"].map({1: "Uninsured", 2: "Insured"})

    df["age"] = df["AGEP_A"].astype(float)
    df["sex_male"] = (df["SEX_A"] == 1).astype(float)
    df["bmi_cat"] = df["BMICAT_A"].where(df["BMICAT_A"] <= 4)
    df["genhlth"] = df["PHSTAT_A"].where(df["PHSTAT_A"] <= 5)
    df["smoker"] = _yn(df["SMKEV_A"])
    df["current_smk"] = (df["SMKCIGST_A"].isin([1, 2])).astype(float)
    _na = pd.Series(np.nan, index=df.index)
    df["heavy_drink"] = _yn(df.get("DRKHVY12M_A", _na))
    df["phys_active"] = _yn(df.get("PA18_02R_A", _na))
    df["hypertension"] = _yn(df["HYPEV_A"])
    df["high_chol"] = _yn(df["CHLEV_A"])
    df["stroke"] = _yn(df["STREV_A"])
    df["arthritis"] = _yn(df.get("ARTHEV_A", _na))
    df["depression"] = _yn(df.get("DEPEV_A", _na))
    df["kidney"] = _yn(df.get("KIDWEAKEV_A", _na))
    df["medcost"] = _yn(df["MEDDL12M_A"])
    df["usual_care"] = _yn(df["USUALPL_A"])
    df["insured_num"] = (df["NOTCOV_A"] == 2).astype(float)

    candidate_cols = [
        "age", "sex_male", "bmi_cat", "genhlth",
        "smoker", "current_smk", "heavy_drink", "phys_active",
        "hypertension", "high_chol",
        "stroke", "arthritis", "depression", "kidney",
        "medcost", "usual_care", "insured_num",
    ]
    feature_cols = [c for c in candidate_cols if df[c].notna().mean() >= 0.20]
    demo_cols = ["age_group", "sex", "race", "insured"]

    df = df[df["AGEP_A"] >= 18].copy()
    df = df.dropna(subset=["target"] + feature_cols + demo_cols).copy()

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_xgb(), X_tr, y_tr)

    genhlth_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["genhlth"]
    return _pack("nhis2023", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=genhlth_te.values, need_name="genhlth_proxy")


# ── 6. NHANES 2021-2023 ──────────────────────────────────────────────────────
def load_nhanes() -> Dict[str, Any]:
    FILES = {
        "DEMO": "DEMO_L.xpt", "DIQ": "DIQ_L.xpt", "GHB": "GHB_L.xpt",
        "TCHOL": "TCHOL_L.xpt", "BMX": "BMX_L.xpt", "BPQ": "BPQ_L.xpt",
        "BPXO": "BPXO_L.xpt", "SMQ": "SMQ_L.xpt", "ALQ": "ALQ_L.xpt",
        "PAQ": "PAQ_L.xpt", "MCQ": "MCQ_L.xpt", "HIQ": "HIQ_L.xpt",
    }
    cache_dir = Path(os.environ.get("NHANES_CACHE_DIR", REPO / "nhanes_cache"))

    def _load_xpt(key: str) -> pd.DataFrame:
        p = cache_dir / FILES[key]
        if not p.exists():
            raise FileNotFoundError(f"NHANES cache missing {p}")
        return pd.read_sas(io.BytesIO(p.read_bytes()), format="xport")

    demo_f = _load_xpt("DEMO")
    df = (demo_f
          .merge(_load_xpt("DIQ")[["SEQN", "DIQ010"]], on="SEQN", how="left")
          .merge(_load_xpt("GHB")[["SEQN", "LBXGH"]], on="SEQN", how="left")
          .merge(_load_xpt("TCHOL")[["SEQN", "LBXTC"]], on="SEQN", how="left")
          .merge(_load_xpt("BMX")[["SEQN", "BMXBMI"]], on="SEQN", how="left")
          .merge(_load_xpt("BPQ")[["SEQN", "BPQ020"]], on="SEQN", how="left")
          .merge(_load_xpt("BPXO")[["SEQN", "BPXOSY1", "BPXODI1"]], on="SEQN", how="left")
          .merge(_load_xpt("SMQ")[["SEQN", "SMQ020"]], on="SEQN", how="left")
          .merge(_load_xpt("ALQ")[["SEQN", "ALQ151"]], on="SEQN", how="left")
          .merge(_load_xpt("PAQ")[["SEQN", "PAD680"]], on="SEQN", how="left")
          .merge(_load_xpt("MCQ")[["SEQN", "MCQ160C", "MCQ160F"]], on="SEQN", how="left")
          .merge(_load_xpt("HIQ")[["SEQN", "HIQ011"]], on="SEQN", how="left"))

    df = df[df["RIDAGEYR"] >= 18].copy()
    df["target"] = np.where(df["DIQ010"] == 1, 1,
                            np.where(df["DIQ010"] == 2, 0, np.nan))

    RACE_MAP = {1: "Mexican-American", 2: "Other-Hispanic", 3: "NH-White",
                4: "NH-Black", 5: "NH-Asian", 6: "NH-Other", 7: "NH-Other"}
    df["race"] = df["RIDRETH3"].map(RACE_MAP)
    df["sex"] = df["RIAGENDR"].map({1: "Male", 2: "Female"})
    df["age_group"] = pd.cut(
        df["RIDAGEYR"], bins=[18, 35, 50, 65, 200],
        labels=["18-34", "35-49", "50-64", "65+"], include_lowest=True,
    ).astype(str)
    df["insured"] = df["HIQ011"].map({1: "Insured", 2: "Uninsured"})

    df["age"] = df["RIDAGEYR"].astype(float)
    df["sex_male"] = (df["RIAGENDR"] == 1).astype(float)
    df["bmi"] = df["BMXBMI"].astype(float)
    df["hba1c"] = df["LBXGH"].astype(float)
    df["chol"] = df["LBXTC"].astype(float)
    df["sbp"] = df["BPXOSY1"].astype(float)
    df["dbp"] = df["BPXODI1"].astype(float)
    df["htn_dx"] = _yn(df["BPQ020"])
    df["ever_smoked"] = _yn(df["SMQ020"])
    df["heavy_drink"] = _yn(df["ALQ151"])
    df["inactive"] = (df["PAD680"] >= 480).astype(float)
    df["chd_dx"] = _yn(df["MCQ160C"])
    df["stroke_dx"] = _yn(df["MCQ160F"])
    df["insured_num"] = (df["HIQ011"] == 1).astype(float)

    feature_cols = ["age", "sex_male", "bmi", "hba1c", "chol", "sbp", "dbp",
                    "htn_dx", "ever_smoked", "heavy_drink", "inactive",
                    "chd_dx", "stroke_dx", "insured_num"]
    demo_cols = ["age_group", "sex", "race", "insured"]
    df = df.dropna(subset=["target"] + feature_cols + demo_cols).copy()

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_xgb(), X_tr, y_tr)

    hba1c_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["hba1c"]
    return _pack("nhanes2123", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=hba1c_te.values, need_name="hba1c_proxy")


# ── 7. BRFSS 2024 ────────────────────────────────────────────────────────────
def load_brfss2024() -> Dict[str, Any]:
    path = Path(os.environ.get("BRFSS_PATH", REPO / "brfss_cache" / "LLCP2024.XPT"))
    if not path.exists():
        raise FileNotFoundError(f"BRFSS 2024 XPT not found at {path}")
    df = pd.read_sas(path, format="xport", encoding="latin-1")

    RACE_LABELS = {1: "White", 2: "Black", 3: "AIAN", 4: "Asian",
                   5: "NHPI", 6: "Other", 7: "Multiracial", 8: "Hispanic", 9: None}
    AGE_LABELS = {1: "18-24", 2: "25-34", 3: "35-44", 4: "45-54",
                  5: "55-64", 6: "65+"}
    INCOME_LABELS = {1: "<$15K", 2: "$15-25K", 3: "$25-35K", 4: "$35-50K",
                     5: "$50-100K", 6: "$100-200K", 7: ">=$200K", 9: None}
    HEALTHPLAN_LABELS = {1: "Insured", 2: "Uninsured", 9: None}

    def _yes_no_to_int(s):
        return s.map({1: 1, 2: 0}).astype(float)

    df["target"] = df["_MICHD"].map({1.0: 1, 2.0: 0})
    df["age_group"] = df["_AGE_G"].map(AGE_LABELS)
    df["sex"] = df["SEXVAR"].map({1: "M", 2: "F"})
    df["race"] = df["_RACE"].map(RACE_LABELS)
    df["income"] = df["_INCOMG1"].map(INCOME_LABELS)
    df["health_plan"] = df["_HLTHPL2"].map(HEALTHPLAN_LABELS)

    df["bmi"] = df["_BMI5"] / 100.0
    df["age_numeric"] = df["_AGE80"]
    df["genhlth"] = df["GENHLTH"].where(df["GENHLTH"] <= 5)
    df["physhlth"] = df["PHYSHLTH"].replace({77: np.nan, 88: 0, 99: np.nan})
    df["menthlth"] = df["MENTHLTH"].replace({77: np.nan, 88: 0, 99: np.nan})
    df["sex_male"] = (df["SEXVAR"] == 1).astype(float)
    df["smoker"] = _yes_no_to_int(df["_RFSMOK3"])
    df["heavy_drink"] = _yes_no_to_int(df["_RFDRHV9"])
    df["phys_active"] = _yes_no_to_int(df["_TOTINDA"])
    df["diabetes"] = _yes_no_to_int(df["DIABETE4"].map(
        {1: 1, 2: 1, 3: 0, 4: 0, 7: np.nan, 9: np.nan}))
    df["asthma"] = _yes_no_to_int(df["_LTASTH1"])
    df["stroke"] = _yes_no_to_int(df["CVDSTRK3"])
    df["kidney"] = _yes_no_to_int(df["CHCKDNY2"])
    df["copd"] = _yes_no_to_int(df["CHCCOPD3"])
    df["arthritis"] = _yes_no_to_int(df["_DRDXAR2"])
    df["depression"] = _yes_no_to_int(df["ADDEPEV3"])
    df["medcost"] = _yes_no_to_int(df["MEDCOST1"])
    df["any_insurance"] = _yes_no_to_int(df["_HLTHPL2"])
    df["checkup_recent"] = (df["CHECKUP1"].isin([1, 2])).astype(float)

    feature_cols = [
        "age_numeric", "sex_male", "bmi", "genhlth", "physhlth", "menthlth",
        "smoker", "heavy_drink", "phys_active",
        "diabetes", "asthma", "stroke",
        "kidney", "copd", "arthritis", "depression",
        "medcost", "any_insurance", "checkup_recent",
    ]
    demo_cols = ["age_group", "sex", "race", "income", "health_plan"]
    df = df.dropna(subset=["target"] + demo_cols + feature_cols).copy()

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[demo_cols].astype(str)

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_xgb(), X_tr, y_tr)

    physhlth_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["physhlth"]
    return _pack("brfss2024", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=physhlth_te.values, need_name="physhlth_proxy")


# ── 8. UCI Adult Income ──────────────────────────────────────────────────────
def load_adult_income() -> Dict[str, Any]:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()
    df["target"] = (
        df["class"].astype(str).str.strip().str.startswith(">50K")).astype(int)
    df = df.drop(columns=["class"]).dropna()

    df["sex_str"] = df["sex"].astype(str).str.strip()
    df["race_str"] = df["race"].astype(str).str.strip()
    df["age_group"] = np.where(df["age"] >= 40, "40+", "<40")
    df["sex_bin"] = (df["sex_str"] == "Male").astype(int)
    df["white_bin"] = (df["race_str"] == "White").astype(int)
    df["married_bin"] = df["marital-status"].astype(str).str.contains(
        "Married").astype(int)
    df["us_native_bin"] = (
        df["native-country"].astype(str).str.strip() == "United-States").astype(int)

    feature_cols = [
        "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
        "hours-per-week", "sex_bin", "white_bin", "married_bin", "us_native_bin",
    ]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "race_str", "age_group"]].rename(
        columns={"sex_str": "sex", "race_str": "race"})

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_logreg(), X_tr, y_tr)

    edu_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["education-num"]
    return _pack("adult_income", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=edu_te.values, need_name="education_proxy")


# ── 9. Folktables ACS-Income ─────────────────────────────────────────────────
def load_acs_income() -> Dict[str, Any]:
    from folktables import ACSDataSource, ACSIncome

    # ACSDataSource resolves root_dir relative to the CWD; force the repo root
    # so the cached data/2018/1-Year/psam_p06.csv is found regardless of CWD.
    src = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person",
                        root_dir=str(REPO / "data"))
    acs = src.get_data(states=["CA"], download=False)
    feat_df, y_arr, group_arr = ACSIncome.df_to_pandas(acs)
    df = feat_df.copy()
    df["target"] = np.asarray(y_arr).ravel().astype(int)
    df["RAC1P"] = np.asarray(group_arr).ravel()

    if len(df) > 20000:
        rng = np.random.RandomState(42)
        p1 = df[df["target"] == 1].sample(
            n=int(20000 * df["target"].mean()), random_state=rng)
        p0 = df[df["target"] == 0].sample(n=20000 - len(p1), random_state=rng)
        df = pd.concat([p0, p1]).reset_index(drop=True)

    df["sex_str"] = np.where(df["SEX"].astype(int) == 1, "Male", "Female")
    df["race_bin"] = np.where(df["RAC1P"].astype(int) == 1, "White", "Non-White")
    df["age_group"] = np.where(df["AGEP"].astype(int) >= 40, "40+", "<40")

    feature_cols = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP",
                    "WKHP", "SEX", "RAC1P"]
    feature_cols = [c for c in feature_cols if c in df.columns]
    df = df.dropna(subset=feature_cols + ["target"])

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "race_bin", "age_group"]].rename(
        columns={"sex_str": "sex", "race_bin": "race"})

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)
    model = _fit(_logreg(), X_tr, y_tr)

    schl_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(
        drop=True)["SCHL"]
    return _pack("acs_income", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=schl_te.values, need_name="schl_proxy",
                 extra={"wkhp_index": feature_cols.index("WKHP")
                        if "WKHP" in feature_cols else 0})


# ── 10. Statlog German Credit ────────────────────────────────────────────────
def load_german_credit() -> Dict[str, Any]:
    from sklearn.datasets import fetch_openml

    data = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
    df = data.frame.copy()
    df["target"] = (df["class"].astype(str).str.strip() == "good").astype(int)
    df = df.drop(columns=["class"]).dropna()

    df["sex_str"] = np.where(
        df["personal_status"].astype(str).str.strip().str.startswith("male"),
        "Male", "Female")
    df["age_group"] = np.where(df["age"] >= 40, "40+", "<40")
    df["sex_bin"] = (df["sex_str"] == "Male").astype(int)
    df["owns_property_bin"] = df["property_magnitude"].astype(str).str.contains(
        "real estate|building society|life insurance", case=False, regex=True
    ).astype(int)
    df["foreign_bin"] = (
        df["foreign_worker"].astype(str).str.strip() == "yes").astype(int)
    df["job_skilled_bin"] = df["job"].astype(str).str.contains(
        "skilled", case=False, regex=False).astype(int)

    savings_order = {"no known savings": 0, "<100": 1, "100<=X<500": 2,
                     "500<=X<1000": 3, ">=1000": 4}
    df["savings_num"] = df["savings_status"].astype(str).str.strip().map(savings_order)
    df["savings_num"] = df["savings_num"].fillna(df["savings_num"].median())

    feature_cols = [
        "duration", "credit_amount", "installment_commitment",
        "residence_since", "age", "existing_credits", "num_dependents",
        "sex_bin", "owns_property_bin", "foreign_bin", "job_skilled_bin",
    ]
    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values
    demo = df[["sex_str", "age_group"]].rename(columns={"sex_str": "sex"})

    X_tr, X_te, y_tr, y_te, d_tr, d_te, sav_tr, sav_te = train_test_split(
        X, y, demo, df["savings_num"].values,
        test_size=0.2, random_state=42, stratify=y)
    model = _fit(_logreg(), X_tr, y_tr)

    # The only proxy in the whole study that is NOT also a model input.
    return _pack("german_credit", X_tr, X_te, y_tr, y_te, d_tr, d_te,
                 feature_cols, model, len(df),
                 need_values=sav_te, need_name="savings_proxy")


LOADERS = {
    "synthetic": load_synthetic,
    "uci_heart": load_uci_heart,
    "diabetes130": load_diabetes130,
    "nhis2024": load_nhis2024,
    "nhis2023": load_nhis2023,
    "nhanes2123": load_nhanes,
    "brfss2024": load_brfss2024,
    "adult_income": load_adult_income,
    "acs_income": load_acs_income,
    "german_credit": load_german_credit,
}
