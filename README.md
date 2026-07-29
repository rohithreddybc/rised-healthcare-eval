# RISED Framework

**A measurement toolkit for evaluating high-stakes AI decision-support
systems across five dimensions — Reliability, Inclusivity, Sensitivity,
Equity, Deployability — computing each metric with bootstrap confidence
intervals, plus a configurable advisory policy layer on top. Applied to
healthcare, with cross-domain demonstrations on credit and hiring expert
systems.**

[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-rised--healthcare--eval--dataset-yellow)](https://huggingface.co/datasets/Rohithreddybc/rised-healthcare-eval-dataset)
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F8734-blue)](https://doi.org/10.57967/hf/8734)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)](#tests)
[![Coverage](https://img.shields.io/badge/coverage-82%25-brightgreen)](#tests)

> An XGBoost classifier with **AUROC 0.961** can still fail three of five
> RISED metrics. Aggregate accuracy alone does not surface reliability under
> input perturbation, subgroup performance gaps, threshold instability, or
> proxy-dependent equity findings.

RISED — **R**eliability, **I**nclusivity, **S**ensitivity, **E**quity,
**D**eployability — is a measurement toolkit for high-stakes AI
decision-support systems. Each of the five dimensions is operationalized
as a metric, computed with bootstrap 95% confidence intervals and compared
against a literature-derived reference threshold, packaged as an
open-source Python library. A configurable advisory policy layer turns
each metric/threshold comparison into a pass/fail flag for convenience —
this is a configurable convention supplied by the library, not an
empirically validated deployment-readiness determination, and RISED does
not certify that a model is safe or cleared to deploy. The framework was
developed for clinical AI (where the empirical evidence is densest and the
regulatory discussion is most active) and is demonstrated on
credit-scoring and hiring expert systems via the cross-domain demos in
`examples/`.

📄 **Paper:** *RISED: A Pre-Deployment Evaluation Framework for High-Stakes
AI Decision-Support Systems, with Application to Healthcare* — manuscript
in preparation for submission to the *Journal of Biomedical Informatics*
(Elsevier).
Dataset DOI: [10.57967/hf/8734](https://doi.org/10.57967/hf/8734)

---

## How to Cite

If you use RISED in your own evaluation, please cite the paper:

```bibtex
@unpublished{bellibatlu2026rised,
  title   = {RISED: A Pre-Deployment Evaluation Framework for
             High-Stakes AI Decision-Support Systems, with
             Application to Healthcare},
  author  = {Bellibatlu, Rohith Reddy and Singh, Manpreet and
             Jajoo, Yash and Lakhanpal, Shyamal and Israni, Abhishek},
  note    = {Manuscript in preparation for submission to the
             Journal of Biomedical Informatics},
  year    = {2026},
  url     = {https://github.com/rohithreddybc/rised-healthcare-eval}
}
```

## One-command reproduction

```bash
git clone https://github.com/rohithreddybc/rised-healthcare-eval.git
cd rised-healthcare-eval
conda env create -f environment.yml && conda activate rised
python -m rised.reproduce_all
```

This runs, in sequence: four real-data within-cohort evaluations (UCI Heart
Disease, UCI Diabetes 130, NHIS 2024, NHIS 2023), the multi-model
robustness check, the Fairlearn comparison, and three cross-domain demos
(`adult_income_demo.py`, `folktables_acs_income_demo.py`,
`german_credit_demo.py`). It does **not** currently run the synthetic-cohort
baseline or the NHANES 2021–2023, BRFSS 2024, or MIMIC-IV-ED evaluations —
run those individually with the corresponding script listed in the cohort
table below.

---

## Why RISED

Standard model evaluation reports a single number (AUROC, Brier score,
accuracy) on a held-out test set. That number cannot detect:

- A model that flips one in fifteen patient classifications under semantically
  equivalent input encodings (**Reliability**).
- An AUC parity gap concentrated in the oldest patient subgroup
  (**Inclusivity**).
- Unstable threshold behavior that reclassifies ~20% of patients when the
  binary cutoff is adjusted to balance sensitivity vs. specificity (**Sensitivity**).
- Scoring decisions whose alignment with clinical need depends on the
  proxy chosen, with the verdict flipping under an independent need proxy
  (**Equity**).
- Models whose explanations contradict each other patient to patient or fail
  per-patient latency targets (**Deployability**).

These failure modes are well-documented at scale in production clinical AI
(Obermeyer et al., *Science* 2019; Wong et al., *JAMA Intern Med* 2021;
Finlayson et al., *NEJM* 2021), but no existing toolkit reports all five as
part of a single, reproducible measurement pass.

---

## The Five Dimensions

| Dimension | Primary metric | Pass threshold | Basis |
|-----------|----------------|----------------|-------|
| **R**eliability   | Judge Sensitivity Score (JSS) | < 0.05 | Steyerberg 2010; JudgeSense 2025 |
| **I**nclusivity   | AUC parity gap Δ_AUC          | ≤ 0.05 | FDA AI/ML Action Plan 2021; AIF360 |
| **S**ensitivity   | Max threshold flip rate (TFR) | ≤ 10%  | Wynants 2019 |
| **E**quity        | Need-prediction ρ (Spearman)  | ≥ 0.70 | Cohen 1988; Obermeyer 2019 |
| **D**eployability | Mean inference latency Λ      | ≤ 500 ms | Sutton 2020 |

> The thresholds above are literature-derived conventions used as defaults
> by the advisory policy layer. They are configurable, and RISED does not
> assert that they have been empirically validated against observed
> deployment outcomes.

---

## Quickstart

```bash
git clone https://github.com/rohithreddybc/rised-healthcare-eval.git
cd rised-healthcare-eval
pip install -e .
```

```python
import rised
from rised.datasets import load_synthea_cohort, train_baseline_model
from sklearn.model_selection import train_test_split

# 1. Load data + train any sklearn-compatible classifier with predict_proba
X, y, demo = load_synthea_cohort()
X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
    X, y, demo, test_size=0.20, random_state=42, stratify=y)
model = train_baseline_model(X_tr, y_tr)

# 2. Define perturbations for the Reliability dimension
specs = [
    {"type": "gaussian_noise", "scale": 0.05, "random_state": 0, "label": "noise_5pct"},
    {"type": "gaussian_noise", "scale": 0.10, "random_state": 1, "label": "noise_10pct"},
    {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05, "label": "age_+5pct"},
    {"type": "unit_rescaling", "feature_index": 0, "factor": 1.06, "label": "age_+6pct"},
]

# 3. Run all five dimensions with bootstrap CIs
report = rised.evaluate_all(
    model, X_te, y_te, d_te,
    perturbation_specs=specs,
    random_state=42, n_bootstrap=1000,
)

# 4. Inspect the results
print(report.summary())
# {'reliability': False, 'inclusivity': False, 'sensitivity': False,
#  'equity': True,  'deployability': True}

print(f"JSS = {report.reliability.judge_sensitivity_score:.4f} "
      f"95% CI {report.reliability.jss_ci}")
# Expected output (random_state=42, B=1000):
# JSS = 0.0644 95% CI (0.0576, 0.0704)
```

A scorecard visualization is one call away:

```python
from rised.visualization import plot_framework_dashboard
fig = plot_framework_dashboard(report)
fig.savefig("scorecard.png", dpi=150)
```

![RISED scorecard](docs/img/scorecard.png)

---

## Reproducing the Paper's Results

The paper applies RISED's metric and bootstrap-CI functions to an XGBoost
classifier (AUROC 0.961, Brier 0.073) on the 2,000-patient held-out test
split. The table below reports BCa bootstrap confidence intervals computed
from the library's estimator functions; the Status column follows the
manuscript's own decision rule for that analysis (a metric whose CI
excludes the reference threshold is marked PASS/FAIL, one whose CI spans
it is marked INCONCLUSIVE). That decision rule is applied in the paper's
analysis scripts — it is **not** the same computation as `report.summary()`
above, which compares point estimates against the reference thresholds
directly and does not apply a multiple-comparisons correction.

| Dimension | Value | 95% BCa CI | Status |
|-----------|------:|:----------:|:------:|
| JSS                       | 0.064  | [0.058, 0.070]   | **FAIL** |
| Δ_AUC                     | 0.059  | [0.042, 0.066]   | **INCONCLUSIVE**¹ |
| Max TFR                   | 19.9%  | [18.3%, 21.7%]   | **FAIL** |
| ρ_need (outcome proxy)    | 0.732  | —                | DIAGNOSTIC² |
| ρ_need (CCI proxy)        | 0.599  | —                | DIAGNOSTIC² |
| Λ (per cohort)            | ~1 ms  | —                | PASS |

¹ BCa CI [0.042, 0.066] spans the 0.05 threshold → INCONCLUSIVE under the
  manuscript's CI decision rule.
² Equity is reported as a proxy-dependence diagnostic, not a stand-alone
  gate: the two rows show how the verdict depends on which need proxy is
  chosen, since no independently validated need measure was available for
  this cohort.

Bootstrap CIs from 1,000 iterations with `random_state=42`. Hardware-dependent
latency reported on a single test machine.

The **Reliability** failure is dominated by Gaussian noise perturbations:

![Reliability flip rates](docs/img/reliability.png)

The **Sensitivity** failure spans the full threshold sweep, with peaks at
operationally relevant boundaries (τ = 0.10 and τ ≥ 0.80):

![Threshold sweep](docs/img/threshold_sweep.png)

---

## Dataset

The 10,000-patient synthetic cohort is published as a HuggingFace dataset:

🤗 **[Rohithreddybc/rised-healthcare-eval-dataset](https://huggingface.co/datasets/Rohithreddybc/rised-healthcare-eval-dataset)**

```python
from datasets import load_dataset
ds = load_dataset("Rohithreddybc/rised-healthcare-eval-dataset")
df = ds["train"].to_pandas()  # (10000, 26)
```

Or regenerate deterministically from source:

```python
from rised.datasets import generate_synthea_cohort
df = generate_synthea_cohort(n=10000, random_state=42)
```

The cohort is fully synthetic (Synthea-inspired), Medicare/Medicaid-weighted,
with 30% high-need prevalence. **No real patient records were used at any
stage.** See the [HuggingFace dataset card](https://huggingface.co/datasets/Rohithreddybc/rised-healthcare-eval-dataset)
for the full feature schema and demographic breakdown.

---

## How RISED Compares

RISED is designed to **complement** existing toolkits, not replace them:

| Toolkit | Purpose | Output |
|---------|---------|--------|
| AI Fairness 360 | Menu of fairness metrics for model selection | Metrics (development-time) |
| Fairlearn       | Fairness-aware training algorithms           | Metrics + mitigation (development-time) |
| TRIPOD+AI       | Reporting standard for prediction studies   | Reporting checklist only |
| TEHAI           | Translational evaluation taxonomy            | Qualitative taxonomy |
| **RISED**       | **5-dimension metric suite with bootstrap CIs and a configurable advisory policy layer** | **Metrics + advisory pass/fail flags** |

RISED's five dimensions were chosen to reflect testing concerns raised — but
not specified in operational detail — by the FDA AI/ML Action Plan, the ONC
HTI-1 rule, and the EU AI Act. RISED does not implement, and does not claim,
compliance with any of these frameworks.

---

## Tests

```bash
pytest --tb=short -q
```

81 tests passing, 82% overall line coverage (>90% on the core evaluation
modules: reliability, sensitivity, equity, inclusivity, metrics).

---

## Citation

```bibtex
@unpublished{bellibatlu2026rised,
  title   = {{RISED}: A Pre-Deployment Evaluation Framework for High-Stakes {AI}
             Decision-Support Systems, with Application to Healthcare},
  author  = {Bellibatlu, Rohith Reddy and Singh, Manpreet and
             Jajoo, Yash and Lakhanpal, Shyamal and Israni, Abhishek},
  note    = {Manuscript in preparation for submission to the
             Journal of Biomedical Informatics},
  year    = {2026},
  url     = {https://github.com/rohithreddybc/rised-healthcare-eval}
}
```

---

## Within-cohort evaluation on real datasets

The framework has been re-run unchanged on six publicly available real-data clinical cohorts spanning 35 years of vintage, plus three non-clinical cohorts that demonstrate the protocol is domain-agnostic. In every case the model is trained and tested on a random split **within the same cohort** — there is no cross-site, cross-time, or cross-population holdout, so these are within-cohort evaluations of generalization to a held-out random sample, not external validations:

| Cohort | n | Era | Outcome | Reproduce |
|---|---|---|---|---|
| UCI Heart Disease (Cleveland) | 303 | 1989 | Heart disease presence | `python examples/external_validation_uci_heart.py` |
| UCI Diabetes 130-US Hospitals | 99,492 | 1999–2008 | <30-day readmission | `python examples/external_validation_diabetes130.py` |
| NCHS NHIS 2024 (Sample Adult) | 9,747 | 2024 | Coronary heart disease / MI | `python examples/external_validation_nhis2024.py` |
| NCHS NHIS 2023 (Sample Adult) | 27,114 | 2023 | Physician-diagnosed diabetes | `python examples/external_validation_nhis2023_diabetes.py` |
| NCHS NHANES 2021–2023 | 4,096 | 2021–2023 | Diabetes (with lab HbA1c) | `python examples/external_validation_nhanes2122.py` |
| CDC BRFSS 2024 | 44,888 | 2024 | Coronary heart disease / MI | `python examples/external_validation_brfss2024.py` |
| *Cross-domain:* Statlog German Credit | 1,000 | — | Credit risk | `python examples/german_credit_demo.py` |
| *Cross-domain:* UCI Adult Income | 45,222 | 1994 | Income > $50k | `python examples/adult_income_demo.py` |
| *Cross-domain:* Folktables ACS-Income | 20,000 | 2018 | Income > $50k | `python examples/folktables_acs_income_demo.py` |

The cohorts produce non-uniform pass/fail patterns across these within-cohort evaluations. On Diabetes 130, Reliability passes by three orders of magnitude (PSS = 0.0004) while Inclusivity (ΔAUC = 0.262) and Sensitivity (max TFR = 49.1%) fail decisively; both NHIS cohorts and BRFSS 2024 reproduce the Inclusivity/Sensitivity failure, while NHANES 2021–2023 — with a complete laboratory feature set — reaches INCONCLUSIVE rather than outright failure. The same Reliability-pass / Sensitivity-fail / Inclusivity-fail pattern recurs on the three non-clinical cohorts, showing the protocol runs beyond healthcare data as well. The MIMIC-IV-ED integration (`examples/external_validation_mimic_ed.py`) runs end-to-end on the public MIMIC-IV-ED demo (again a within-cohort split); the full credentialed cohort is the priority next step.

## Roadmap

- [x] Within-cohort evaluation on UCI Heart Disease (Cleveland)
- [x] Within-cohort evaluation on UCI Diabetes 130-US Hospitals
- [x] Within-cohort evaluation on NCHS NHIS 2024, NHIS 2023, NHANES 2021–2023, and CDC BRFSS 2024
- [x] Within-cohort evaluation on non-clinical domains: German Credit, UCI Adult Income, and Folktables ACS-Income
- [x] MIMIC-IV-ED integration implemented and runs end-to-end on the public demo
- [ ] Full MIMIC-IV-ED / eICU evaluation (PhysioNet credential required)
- [ ] True external validation across sites, time periods, or populations
- [ ] Re-run on NHIS 2025 / NHANES 2025–2026 once microdata is released
- [ ] Empirical recalibration of pass/fail thresholds against deployment outcomes
- [ ] Extension to multi-label and time-series risk models
- [ ] Reference implementations for common clinical prediction tasks
- [ ] Certification pathway aligned with FDA SaMD requirements

Contributions welcome — please open an issue or pull request on GitHub.

---

## Contact

**Rohith Reddy Bellibatlu**
[rbell084@fiu.edu](mailto:rbell084@fiu.edu) |
ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)

## License

MIT — see [LICENSE](LICENSE).
