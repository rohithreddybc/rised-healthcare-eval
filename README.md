# RISED Framework

**A pre-deployment evaluation framework for clinical AI decision-support systems
spanning Reliability, Inclusivity, Sensitivity, Equity, and Deployability.**

[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-rised--healthcare--eval--dataset-yellow)](https://huggingface.co/datasets/Rohithreddybc/rised-healthcare-eval-dataset)
[![DOI](https://img.shields.io/badge/DOI-10.57967%2Fhf%2F8734-blue)](https://doi.org/10.57967/hf/8734)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-81%20passing-brightgreen)](#tests)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](#tests)

> An XGBoost classifier with **AUROC 0.961** can still fail three of five
> RISED dimensions. Aggregate accuracy is necessary but not sufficient for safe
> clinical deployment.

RISED — **R**eliability, **I**nclusivity, **S**ensitivity, **E**quity,
**D**eployability — is a structured framework for testing healthcare AI systems
*before* they are deployed in clinical settings. Each of the five dimensions is
operationalized through formal sub-criteria with literature-grounded pass/fail
thresholds and bootstrap 95% confidence intervals, packaged as an open-source
Python library.

📄 **Paper:** *Evaluating AI-Assisted Decision Support in High-Stakes Healthcare:
A Framework for Reliability, Inclusivity, Sensitivity, Equity, and Deployability*
— under review at *Artificial Intelligence in Medicine* (Elsevier, IF 7.5).
Dataset DOI: [10.57967/hf/8734](https://doi.org/10.57967/hf/8734)

---

## Why RISED

Standard model evaluation reports a single number — AUROC, Brier score,
accuracy — on a held-out test set. That number cannot detect:

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
Finlayson et al., *NEJM* 2021), but no existing toolkit captures them as a
gateable pre-deployment test.

---

## The Five Dimensions

| Dimension | Primary metric | Pass threshold | Basis |
|-----------|----------------|----------------|-------|
| **R**eliability   | Judge Sensitivity Score (JSS) | < 0.05 | Steyerberg 2010; JudgeSense 2025 |
| **I**nclusivity   | AUC parity gap Δ_AUC          | ≤ 0.05 | FDA AI/ML Action Plan 2021; AIF360 |
| **S**ensitivity   | Max threshold flip rate (TFR) | ≤ 10%  | Wynants 2019 |
| **E**quity        | Need-prediction ρ (Spearman)  | ≥ 0.70 | Cohen 1988; Obermeyer 2019 |
| **D**eployability | Mean inference latency Λ      | ≤ 500 ms | Sutton 2020 |

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

The paper applies RISED to an XGBoost classifier (AUROC 0.961, Brier 0.073) on
the 2,000-patient held-out test split, using the CI-based decision rule.

| Dimension | Value | 95% CI | Status |
|-----------|------:|:------:|:------:|
| JSS                       | 0.064  | [0.058, 0.070]   | **FAIL** |
| Δ_AUC                     | 0.059  | [0.052, 0.097]   | **INCONCLUSIVE**¹ |
| Max TFR                   | 19.9%  | [18.3%, 21.7%]   | **FAIL** |
| ρ_need (outcome proxy)    | 0.732  | —                | PASS² |
| ρ_need (CCI proxy)        | 0.599  | —                | FAIL² |
| Λ (per cohort)            | ~1 ms  | —                | PASS |

¹ CI [0.052, 0.097] overlaps the 0.05 threshold → INCONCLUSIVE.
² Equity verdict flips under different proxies; recorded as INCONCLUSIVE pending
   an externally validated need measure.

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

| Toolkit | Purpose | Pre-deployment gate? |
|---------|---------|----------------------|
| AI Fairness 360 | Menu of fairness metrics for model selection | No (development-time) |
| Fairlearn       | Fairness-aware training algorithms           | No (development-time) |
| TRIPOD+AI       | Reporting standard for prediction studies   | No (reporting only) |
| TEHAI           | Translational evaluation taxonomy            | No (qualitative) |
| **RISED**       | **5-dim pre-deployment test with pass/fail thresholds** | **Yes** |

RISED operationalizes the testing requirements implied — but not specified —
by the FDA AI/ML Action Plan, the ONC HTI-1 rule, and the EU AI Act.

---

## Tests

```bash
pytest --tb=short -q
```

81 tests, ~93% line coverage across the rised package.

---

## Citation

```bibtex
@article{bellibatlu2026rised,
  author  = {Bellibatlu, Rohith Reddy},
  title   = {{RISED}: A Pre-Deployment Safety Evaluation Framework for Clinical {AI}
             Decision-Support Systems},
  journal = {Artificial Intelligence in Medicine},
  year    = {2026},
  note    = {Under review},
  url     = {https://github.com/rohithreddybc/rised-healthcare-eval}
}
```

---

## External validation on real datasets

The framework has been re-run unchanged on three publicly available real-data clinical cohorts spanning three decades of vintage:

| Cohort | n | Era | Outcome | Reproduce |
|---|---|---|---|---|
| UCI Heart Disease (Cleveland) | 303 | 1989 | Presence of heart disease | `python examples/external_validation_uci_heart.py` |
| UCI Diabetes 130-US Hospitals | 99,492 | 1999–2008 | <30-day readmission | `python examples/external_validation_diabetes130.py` |
| **NCHS NHIS 2024 (Sample Adult)** | **9,747 analytic / 32,629 raw** | **2024** | **Coronary heart disease / MI** | **`python examples/external_validation_nhis2024.py`** |

The three real cohorts produce non-uniform pass/fail patterns: Reliability passes by three orders of magnitude on Diabetes 130 while Inclusivity (ΔAUC = 0.262) and Sensitivity (max TFR = 49.1%) fail decisively, supporting the framework's construct validity. The NHIS 2024 cohort, collected during calendar year 2024 and released by the National Center for Health Statistics in 2025, provides a contemporary nationally representative check on the same dimensions. (NHIS 2025 microdata had not yet been released at the time of submission; an alternative validation script targeting the CDC BRFSS 2024 release is also provided in `examples/external_validation_brfss2024.py`.)

## Roadmap

- [x] External validation on UCI Heart Disease (Cleveland)
- [x] External validation on UCI Diabetes 130-US Hospitals
- [x] External validation on NCHS NHIS 2024 (released 2025) and CDC BRFSS 2024 (released Aug 2025)
- [ ] Validation on MIMIC-IV / eICU (PhysioNet credential required)
- [ ] Re-run on NHIS 2025 / NHANES 2025–2026 once microdata is released
- [ ] Empirical recalibration of pass/fail thresholds against deployment outcomes
- [ ] Extension to multi-label and time-series risk models
- [ ] Reference implementations for common clinical prediction tasks
- [ ] Certification pathway aligned with FDA SaMD requirements

Contributions welcome — please open an issue or pull request on GitHub.

---

## Contact

**Rohith Reddy Bellibatlu** —
[rohithreddybc@gmail.com](mailto:rohithreddybc@gmail.com) —
ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)

## License

MIT — see [LICENSE](LICENSE).
