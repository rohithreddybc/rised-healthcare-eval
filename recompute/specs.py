"""
Per-cohort perturbation specifications and equity-proxy provenance.

The perturbation lists are transcribed verbatim from the example scripts so the
0.1.0 and 0.2.0 runs receive identical input. Under 0.1.0 all four specs are
pooled into one Judge Sensitivity Score and Gaussian noise is applied to every
column regardless of type; under 0.2.0 the ``unit_rescaling`` entries are
classified as covariate shift and excluded from JSS, and noise reaches
continuous columns only. That divergence is the correction under test.
"""

from __future__ import annotations

from typing import Any, Dict, List

NOISE = [
    {"type": "gaussian_noise", "scale": 0.05, "random_state": 0, "label": "Noise +5%"},
    {"type": "gaussian_noise", "scale": 0.10, "random_state": 1, "label": "Noise +10%"},
]


def _rescale(index: int, factor: float, label: str) -> Dict[str, Any]:
    return {"type": "unit_rescaling", "feature_index": index,
            "factor": factor, "label": label}


#: Perturbation specs by cohort. ``acs_income`` is resolved at runtime because
#: its second rescaling index is computed from the surviving feature list.
SPECS: Dict[str, List[Dict[str, Any]]] = {
    # README Quickstart: age x1.05 and age x1.06 on feature index 0.
    "synthetic": NOISE + [_rescale(0, 1.05, "Age +5%"),
                          _rescale(0, 1.06, "Age +6%")],
    "uci_heart": NOISE + [_rescale(0, 1.05, "Age +5%"),
                          _rescale(0, 1.06, "Age +6%")],
    "diabetes130": NOISE + [_rescale(0, 1.05, "Age +5%"),
                            _rescale(0, 1.06, "Age +6%")],
    "nhis2024": NOISE + [_rescale(0, 1.05, "Age +5%"),
                         _rescale(0, 1.06, "Age +6%")],
    "nhis2023": NOISE + [_rescale(0, 1.05, "Age +5%"),
                         _rescale(2, 1.10, "BMI-cat +10%")],
    "nhanes2123": NOISE + [_rescale(0, 1.05, "Age +5%"),
                           _rescale(3, 1.08, "HbA1c +8%")],
    "brfss2024": NOISE + [_rescale(0, 1.05, "Age +5%"),
                          _rescale(0, 1.06, "Age +6%")],
    "adult_income": NOISE + [_rescale(0, 1.05, "Age +5%"),
                             _rescale(3, 1.05, "Capital-gain +5%")],
    "german_credit": NOISE + [_rescale(4, 1.05, "Age +5%"),
                              _rescale(1, 1.05, "Credit-amount +5%")],
    # acs_income filled in by specs_for()
}


def specs_for(cohort: str, bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Resolve the spec list, computing runtime-dependent indices."""
    if cohort == "acs_income":
        return NOISE + [_rescale(0, 1.05, "Age +5%"),
                        _rescale(int(bundle.get("wkhp_index", 0)), 1.05,
                                 "Hours-per-week +5%")]
    return SPECS[cohort]


#: How defensible each cohort's Equity need proxy actually is.
#:
#: ``independent``      -- measured outside the model's feature matrix and not a
#:                         determinant of the outcome. Usable.
#: ``model_input``      -- a legitimate clinical/socioeconomic measurement, but
#:                         also a model input feature, so rho is partly
#:                         mechanical: the model is being correlated with one of
#:                         its own predictors.
#: ``outcome_defining`` -- the proxy is (part of) the diagnostic criterion for
#:                         the outcome. rho then re-expresses discrimination and
#:                         the dimension should be treated as not evaluable.
PROXY_VALIDITY: Dict[str, Dict[str, str]] = {
    "synthetic": {
        "column": "cci_proxy (Charlson comorbidity index)",
        "class": "model_input",
        "note": "CCI is a model input feature and enters the label's "
                "data-generating process; rho is partly mechanical.",
    },
    "uci_heart": {
        "column": "chol_proxy (serum cholesterol)",
        "class": "model_input",
        "note": "A model input feature. Not outcome-derived, but not "
                "independent of the score either.",
    },
    "diabetes130": {
        "column": "n_inpatient_proxy (prior inpatient visits)",
        "class": "model_input",
        "note": "A model input feature and the dominant predictor of "
                "readmission; rho largely re-expresses that dependence.",
    },
    "nhis2024": {
        "column": "genhlth_proxy (self-rated general health)",
        "class": "model_input",
        "note": "A model input feature.",
    },
    "nhis2023": {
        "column": "genhlth_proxy (self-rated general health)",
        "class": "model_input",
        "note": "A model input feature.",
    },
    "nhanes2123": {
        "column": "hba1c_proxy (HbA1c)",
        "class": "outcome_defining",
        "note": "HbA1c >= 6.5% is the diagnostic criterion for the outcome "
                "(diabetes) and is also a model input. The proxy is not "
                "outcome-INDEPENDENT in the sense F8 requires, even though the "
                "library's structural guard does not reject it. Equity should "
                "be treated as not evaluable on this cohort.",
    },
    "brfss2024": {
        "column": "physhlth_proxy (days of poor physical health)",
        "class": "model_input",
        "note": "A model input feature.",
    },
    "adult_income": {
        "column": "education_proxy (education-num)",
        "class": "model_input",
        "note": "A model input feature.",
    },
    "acs_income": {
        "column": "schl_proxy (educational attainment)",
        "class": "model_input",
        "note": "A model input feature.",
    },
    "german_credit": {
        "column": "savings_proxy (savings-status ordinal)",
        "class": "independent",
        "note": "Carried alongside the split and never given to the model. "
                "The only genuinely independent proxy in the study.",
    },
}
