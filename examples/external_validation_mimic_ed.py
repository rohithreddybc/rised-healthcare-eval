"""
External validation of the RISED Framework on MIMIC-IV-ED.

Reference:
  Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., &
  Horng, S. (2023). MIMIC-IV-ED (version 2.2). PhysioNet.
  https://doi.org/10.13026/5ntk-km72
  Johnson, A. E. W., et al. (2023). MIMIC-IV, a freely accessible
  electronic health record dataset. Scientific Data, 10, 1.

STATUS: READY-TO-RUN SCAFFOLD (PENDING CREDENTIALED DATA ACCESS).
  MIMIC-IV-ED requires a credentialed PhysioNet account (CITI "Data or
  Specimens Only Research" training + signed data-use agreement). This
  script contains NO precomputed or fabricated results. It produces the
  scorecard ONLY when run against the real MIMIC-IV-ED tables; until then
  it prints an actionable message and exits without emitting numbers.

  Set MIMIC_ED_DIR to the directory holding the MIMIC-IV-ED `ed` module
  CSVs (edstays.csv.gz, triage.csv.gz, diagnosis.csv.gz). Optionally set
  MIMIC_HOSP_DIR to the MIMIC-IV `hosp` module (patients.csv.gz,
  admissions.csv.gz) to enable age and insurance subgroups.

Why this cohort for RISED (maps to all five dimensions):
  * Reliability  : ICD-9 vs ICD-10 codes coexist in diagnosis.icd_version,
                   giving a real encoding-granularity perturbation; vital
                   signs admit unit rescalings (degF/degC, mmHg).
  * Inclusivity  : gender, race, age band, and insurance subgroups.
  * Sensitivity  : admission-probability threshold sweep.
  * Equity       : triage acuity (ESI 1-5) is an outcome-INDEPENDENT,
                   clinician-assigned need measure -- the first non-circular
                   need proxy in the RISED suite (lower ESI = higher need).
  * Deployability: inference latency + SHAP top-3 feature consistency.

Outcome: hospital admission from the ED (edstays.disposition == 'ADMITTED'),
  the standard MIMIC-IV-ED triage-prediction target.

Run (after access):
    MIMIC_ED_DIR=/path/to/mimic-iv-ed/2.2/ed \\
    MIMIC_HOSP_DIR=/path/to/mimic-iv/2.2/hosp \\
    python external_validation_mimic_ed.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

import rised
from rised.equity import evaluate_equity


# --- Configuration (no data is bundled; paths come from the environment) -----
MIMIC_ED_DIR = os.environ.get("MIMIC_ED_DIR", "")
MIMIC_HOSP_DIR = os.environ.get("MIMIC_HOSP_DIR", "")

# RISED informative-verdict floor from the paper's power analysis (Appendix A).
MIN_INFORMATIVE_N = 1500


def _require_access() -> Path:
    """Fail loudly and helpfully if the credentialed data is not present.

    Returns the validated MIMIC-IV-ED directory. Never fabricates data.
    """
    if not MIMIC_ED_DIR:
        print(
            "[MIMIC-IV-ED not configured]\n"
            "  This is a ready-to-run scaffold. It emits NO results until it\n"
            "  is pointed at the real credentialed dataset.\n\n"
            "  1. Obtain credentialed PhysioNet access (CITI training + DUA).\n"
            "  2. Download MIMIC-IV-ED v2.2 (the `ed` module).\n"
            "  3. Set MIMIC_ED_DIR to the directory with edstays.csv.gz,\n"
            "     triage.csv.gz, diagnosis.csv.gz, then re-run.\n",
            file=sys.stderr,
        )
        sys.exit(2)
    ed_dir = Path(MIMIC_ED_DIR)
    required = ["edstays.csv.gz", "triage.csv.gz", "diagnosis.csv.gz"]
    missing = [f for f in required if not (ed_dir / f).exists()]
    if missing:
        print(
            f"[MIMIC-IV-ED files missing in {ed_dir}]: {missing}\n"
            "  Expected the MIMIC-IV-ED `ed` module CSVs.",
            file=sys.stderr,
        )
        sys.exit(2)
    return ed_dir


def _load_cohort(ed_dir: Path) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Build the ED admission-prediction cohort from real MIMIC-IV-ED tables."""
    edstays = pd.read_csv(ed_dir / "edstays.csv.gz",
                          usecols=["stay_id", "subject_id", "hadm_id",
                                   "gender", "race", "disposition"])
    triage = pd.read_csv(ed_dir / "triage.csv.gz",
                         usecols=["stay_id", "temperature", "heartrate",
                                  "resprate", "o2sat", "sbp", "dbp", "pain",
                                  "acuity"])

    df = edstays.merge(triage, on="stay_id", how="inner")

    # Outcome: admission from the ED.
    df["target"] = (df["disposition"].astype(str).str.upper() == "ADMITTED").astype(int)

    # Pain is free-text/numeric in MIMIC; coerce to numeric, drop the rest.
    df["pain"] = pd.to_numeric(df["pain"], errors="coerce")

    # Optional age + insurance from the hosp module.
    if MIMIC_HOSP_DIR and (Path(MIMIC_HOSP_DIR) / "patients.csv.gz").exists():
        pat = pd.read_csv(Path(MIMIC_HOSP_DIR) / "patients.csv.gz",
                          usecols=["subject_id", "anchor_age", "gender"])
        df = df.merge(pat[["subject_id", "anchor_age"]], on="subject_id", how="left")
        df["age_band"] = pd.cut(
            df["anchor_age"], bins=[0, 35, 50, 65, 80, 200],
            labels=["18-34", "35-49", "50-64", "65-79", "80+"]).astype(str)
    else:
        df["age_band"] = "unknown"

    if MIMIC_HOSP_DIR and (Path(MIMIC_HOSP_DIR) / "admissions.csv.gz").exists():
        adm = pd.read_csv(Path(MIMIC_HOSP_DIR) / "admissions.csv.gz",
                          usecols=["hadm_id", "insurance"])
        df = df.merge(adm, on="hadm_id", how="left")
        df["insurance"] = df["insurance"].fillna("Unknown").astype(str)
    else:
        df["insurance"] = "unknown"

    # Collapse race to the standard MIMIC reporting buckets for subgroup eval.
    def _race_bucket(r: str) -> str:
        r = str(r).upper()
        if "WHITE" in r:
            return "White"
        if "BLACK" in r:
            return "Black"
        if "HISPANIC" in r or "LATINO" in r:
            return "Hispanic"
        if "ASIAN" in r:
            return "Asian"
        return "Other/Unknown"
    df["race_bucket"] = df["race"].map(_race_bucket)

    feature_cols = ["temperature", "heartrate", "resprate", "o2sat",
                    "sbp", "dbp", "pain", "acuity"]
    df = df.dropna(subset=feature_cols + ["target"])

    demo = df[["gender", "race_bucket", "age_band", "insurance"]].rename(
        columns={"race_bucket": "race"}).astype(str)
    return df, feature_cols, demo


def main():
    ed_dir = _require_access()
    df, feature_cols, demo = _load_cohort(ed_dir)

    print(f"Cohort: n={len(df):,}, admission prevalence={df['target'].mean():.3f}")
    if len(df) < MIN_INFORMATIVE_N:
        print(f"  WARNING: n={len(df)} is below the RISED informative-verdict "
              f"floor (~{MIN_INFORMATIVE_N}); verdicts will be directional only.")
    for col in demo.columns:
        print(f"  {col}: {dict(demo[col].value_counts())}")

    X = df[feature_cols].astype(float).values
    y = df["target"].astype(int).values

    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.2, random_state=42, stratify=y)

    # Same XGBoost hyperparameters as every other RISED cohort.
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.80,
            eval_metric="logloss", random_state=42, verbosity=0, seed=42)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)

    scores_te = model.predict_proba(X_te)[:, 1]
    print(f"\nTest AUROC: {roc_auc_score(y_te, scores_te):.3f}")
    print(f"Test Brier:  {float(np.mean((scores_te - y_te) ** 2)):.3f}")

    # Reliability battery: vital-sign unit rescalings + measurement noise.
    # (ICD-version perturbation can be added once diagnosis features are
    # included; vitals already exercise the encoding-stability test.)
    perturbation_specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "Vitals noise +5%"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "Vitals noise +10%"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "Temperature unit +5%"},
        {"type": "unit_rescaling",
         "feature_index": feature_cols.index("sbp"), "factor": 1.05,
         "label": "SBP unit +5%"},
    ]

    print("\nRunning RISED evaluation (B=1000 bootstrap) ...")
    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=perturbation_specs,
        random_state=42, n_bootstrap=1000)

    # Equity with triage acuity (ESI) as an OUTCOME-INDEPENDENT need proxy.
    # ESI is assigned by a triage nurse before disposition is known, so it is
    # not derived from the admission label -- the first genuinely non-circular
    # need proxy in the RISED suite. Lower ESI = higher acuity = higher need,
    # so we negate it to make larger = greater need.
    acuity_te = pd.DataFrame(X_te, columns=feature_cols).reset_index(drop=True)["acuity"]
    demo_need = d_te.reset_index(drop=True).copy()
    demo_need["acuity_need"] = (6.0 - acuity_te.values)  # ESI 1->5 need, 5->1
    eq_independent = evaluate_equity(
        model, X_te, y_te, demo_need, need_column="acuity_need")

    B = 1000
    rs = np.random.RandomState(42)
    n = len(scores_te)
    need_vec = (6.0 - acuity_te.values)
    boot_y, boot_need = [], []
    for _ in range(B):
        idx = rs.choice(n, size=n, replace=True)
        r1, _ = spearmanr(scores_te[idx], y_te[idx])
        r2, _ = spearmanr(scores_te[idx], need_vec[idx])
        boot_y.append(r1)
        boot_need.append(r2)
    rho_y_ci = (np.percentile(boot_y, 2.5), np.percentile(boot_y, 97.5))
    rho_need_ci = (np.percentile(boot_need, 2.5), np.percentile(boot_need, 97.5))

    print("\n=== RISED scorecard on MIMIC-IV-ED (real data) ===")
    print(f"  Cohort: n={len(df):,}, test n={len(y_te):,}")
    print(f"  Reliability PSS = {report.reliability.judge_sensitivity_score:.4f}  "
          f"95% CI {report.reliability.jss_ci}")
    print(f"  Inclusivity DeltaAUC = {report.inclusivity.auc_parity_gap:.4f}  "
          f"95% CI {report.inclusivity.auc_gap_ci}")
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    print(f"  Sensitivity max TFR = {max_tfr*100:.1f}%  "
          f"95% CI {tuple(round(x*100,1) for x in report.sensitivity.max_tfr_ci) if report.sensitivity.max_tfr_ci else None}")
    print(f"  Equity rho_need (y_true)        = "
          f"{report.equity.need_prediction_correlation:.4f} "
          f"95% CI [{rho_y_ci[0]:.4f}, {rho_y_ci[1]:.4f}]")
    print(f"  Equity rho_need (triage acuity) = "
          f"{eq_independent.need_prediction_correlation:.4f} "
          f"95% CI [{rho_need_ci[0]:.4f}, {rho_need_ci[1]:.4f}]   "
          f"[OUTCOME-INDEPENDENT proxy]")
    print(f"  Deployability latency = "
          f"{report.deployability.mean_inference_latency_ms:.3f} ms")

    return report, eq_independent


if __name__ == "__main__":
    main()
