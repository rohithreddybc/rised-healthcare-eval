"""Threshold-sensitivity sweep for the RISED Reliability and Sensitivity verdicts.

Referenced in Section 3.6 of the paper ("Threshold sensitivity and metric
monotonicity"). Reproduces:

  - Reliability verdict at PSS thresholds in {0.025, 0.05, 0.075, 0.10}
  - Sensitivity verdict at max-TFR thresholds in {0.05, 0.075, 0.10, 0.125, 0.15}
  - PSS monotonicity check under Gaussian noise sigma in
    {0, 0.025, 0.05, 0.10}

Run:
    python threshold_sensitivity.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np

import rised
from rised.datasets import load_synthea_cohort, train_baseline_model
from sklearn.model_selection import train_test_split


def main():
    X, y, demo = load_synthea_cohort()
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X, y, demo, test_size=0.20, random_state=42, stratify=y
    )
    model = train_baseline_model(X_tr, y_tr)

    # 1. Headline RISED evaluation (default perturbations)
    specs = [
        {"type": "gaussian_noise", "scale": 0.05, "random_state": 0,
         "label": "noise_5pct"},
        {"type": "gaussian_noise", "scale": 0.10, "random_state": 1,
         "label": "noise_10pct"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05,
         "label": "age_+5pct"},
        {"type": "unit_rescaling", "feature_index": 0, "factor": 1.06,
         "label": "age_+6pct"},
    ]
    report = rised.evaluate_all(
        model, X_te, y_te, d_te,
        perturbation_specs=specs,
        random_state=42, n_bootstrap=1000,
    )

    pss = float(report.reliability.judge_sensitivity_score)
    pss_ci = report.reliability.jss_ci
    max_tfr = max(report.sensitivity.threshold_flip_rates.values())
    tfr_ci = report.sensitivity.max_tfr_ci

    print("=== Headline point estimates ===")
    print(f"PSS = {pss:.4f}  95% CI {pss_ci}")
    print(f"max TFR = {max_tfr*100:.2f}%  95% CI "
          f"{tuple(round(x*100, 2) for x in tfr_ci) if tfr_ci else None}")

    def verdict(ci, threshold, direction="upper"):
        if ci is None:
            return "N/A"
        lo, hi = ci
        if direction == "upper":
            if hi < threshold:
                return "PASS"
            if lo > threshold:
                return "FAIL"
            return "INCONCLUSIVE"
        else:  # lower-bounded sub-criterion
            if lo > threshold:
                return "PASS"
            if hi < threshold:
                return "FAIL"
            return "INCONCLUSIVE"

    # 2. Reliability verdict sweep
    print("\n=== Reliability (PSS) verdict sweep ===")
    for thr in [0.025, 0.05, 0.075, 0.10]:
        v = verdict(pss_ci, thr, direction="upper")
        print(f"  threshold = {thr:.3f}  -> {v}")

    # 3. Sensitivity verdict sweep
    print("\n=== Sensitivity (max TFR) verdict sweep ===")
    for thr in [0.05, 0.075, 0.10, 0.125, 0.15]:
        v = verdict(tfr_ci, thr, direction="upper")
        print(f"  threshold = {thr:.3f}  -> {v}")

    # 4. PSS monotonicity check under increasing noise
    print("\n=== PSS monotonicity check under Gaussian noise ===")
    for sigma in [0.0, 0.025, 0.05, 0.10]:
        if sigma == 0.0:
            print(f"  sigma = {sigma:.3f}  PSS = 0.0000 (no perturbation)")
            continue
        specs_sigma = [
            {"type": "gaussian_noise", "scale": sigma, "random_state": 0,
             "label": f"noise_{int(sigma*1000)}per_mille"},
        ]
        rep = rised.evaluate_all(
            model, X_te, y_te, d_te,
            perturbation_specs=specs_sigma,
            random_state=42, n_bootstrap=200,  # cheaper for sweep
        )
        print(f"  sigma = {sigma:.3f}  PSS = "
              f"{rep.reliability.judge_sensitivity_score:.4f}")


if __name__ == "__main__":
    main()
