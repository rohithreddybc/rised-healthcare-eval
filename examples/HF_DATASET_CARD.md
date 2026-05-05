---
license: mit
language:
  - en
tags:
  - healthcare
  - clinical-decision-support
  - synthetic-data
  - algorithmic-fairness
  - risk-stratification
  - health-equity
  - medicare
  - medicaid
pretty_name: RISED Synthetic Patient Cohort (10K)
size_categories:
  - 1K<n<10K
task_categories:
  - tabular-classification
configs:
  - config_name: default
    data_files: synthetic_cohort_10k.csv
---

# RISED Synthetic Patient Cohort (10,000 patients)

A fully synthetic patient cohort designed to mirror the demographic and clinical
complexity of a Medicare/Medicaid-weighted U.S. healthcare population. Generated
deterministically (random_state = 42) by a Synthea-inspired computational model
implemented in the [`rised`](https://github.com/rohithreddybc/rised-healthcare-eval)
Python package. **No real patient records were used at any stage.**

This is the reference dataset used in the demonstration application of the
**RISED Framework** — a five-dimension pre-deployment evaluation framework for
clinical AI decision-support systems (Reliability, Inclusivity, Sensitivity,
Equity, Deployability).

## Dataset Summary

| Attribute | Value |
|-----------|-------|
| Patients (rows) | 10,000 |
| Features (cols) | 26 |
| Outcome prevalence | 30.0% high-need (3,000 patients) |
| License | MIT |
| Source code | [github.com/rohithreddybc/rised-healthcare-eval](https://github.com/rohithreddybc/rised-healthcare-eval) |
| Generator | Synthea-inspired, deterministic (seed = 42) |

## Demographic Composition

| Group | Full cohort | High-need (n=3,000) |
|-------|-------------|---------------------|
| Age 18–44 | 18.4% | 0.2% |
| Age 45–64 | 25.0% | 9.2% |
| Age 65–74 | 28.2% | 31.6% |
| Age 75+   | 28.4% | 59.0% |
| Female / Male | 55.5% / 44.5% | 55.8% / 44.2% |
| White | 63.8% | 63.3% |
| Black | 13.4% | 13.5% |
| Hispanic | 13.0% | 13.1% |
| Asian | 5.7% | 5.9% |
| Other | 4.1% | 4.3% |
| Medicare | 47.4% | 74.2% |
| Medicaid | 14.4% | 7.7% |
| Private | 29.8% | 13.8% |
| Uninsured | 8.4% | 4.4% |

Mean Charlson Comorbidity Index: 0.99 ± 1.20 (full); 1.86 ± 1.33 (high-need).

## Features

**Demographics:** `age`, `sex_male`, `age_group`, `sex`, `race`, `insurance`,
`ins_medicare`, `ins_medicaid`, `ins_private`

**Clinical:** `cci_score`, `has_hypertension`, `has_diabetes`, `has_chf`,
`has_ckd`, `has_copd`, `has_mi`, `has_cvd`, `has_dementia`, `has_cancer`,
`has_metastatic`, `prior_hosp_count`, `ed_visits_count`, `bmi`

**Social:** `adi_score` (Area Deprivation Index, 1–100)

**Outcome:** `high_need` (binary; 1 = top-30% predicted clinical risk)

## Outcome Definition

The `high_need` label is derived from a logistic risk score combining age,
diabetes, CHF, CKD, COPD, prior MI, CCI, prior hospitalizations, ED utilization,
and ADI, with additive Gaussian noise (σ=0.5) for realistic stochasticity.
Patients in the top 30% of predicted risk are labeled `high_need = 1`.

**Important:** Because the outcome is derived directly from the feature space,
this dataset is suitable for evaluation framework demonstrations but not for
benchmarking model accuracy on a real-world prediction task. Real EHR cohorts
introduce distribution shifts and access-barrier distortions absent from
synthetic data.

## Intended Use

- **Primary:** Demonstrating the RISED Framework for pre-deployment evaluation
  of healthcare AI decision-support systems.
- **Secondary:** Teaching, methodological development, and reproducibility
  benchmarking for fairness/calibration/sensitivity tooling.

**Not intended for:** training production clinical models, benchmarking
discrimination performance against real-world systems, or any deployed
clinical use.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("rohithreddybc/rised-synthetic-cohort-10k")
df = ds["train"].to_pandas()
print(df.shape)        # (10000, 26)
print(df["high_need"].mean())  # 0.30
```

Or load directly with pandas:

```python
import pandas as pd
df = pd.read_csv("hf://datasets/rohithreddybc/rised-synthetic-cohort-10k/synthetic_cohort_10k.csv")
```

To regenerate from source (deterministic):

```python
from rised.datasets import generate_synthea_cohort
df = generate_synthea_cohort(n=10000, random_state=42)
```

## Citation

If you use this dataset, please cite the accompanying paper:

```bibtex
@article{bellibatlu2025rised,
  author = {Bellibatlu, Rohith Reddy},
  title  = {Evaluating AI-Assisted Decision Support in High-Stakes Healthcare:
            A Framework for Reliability, Inclusivity, Sensitivity, Equity,
            and Deployability},
  year   = {2025},
  journal = {Artificial Intelligence in Medicine (under review)},
  url    = {https://github.com/rohithreddybc/rised-healthcare-eval}
}
```

## License

MIT. The dataset is fully synthetic and contains no information derived from
real patients; redistribution and derivative works are unrestricted under MIT
terms.

## Limitations

1. **Synthetic only.** Distributions reflect a generative model, not real-world
   epidemiology. Results obtained on this cohort do not generalize to real
   clinical populations without further validation.
2. **No access-barrier distortion.** Unlike real utilization-based outcomes,
   this cohort's `high_need` label is unconfounded by structural inequity,
   which makes it well-suited for need-based fairness demonstrations but
   absent the access-barrier failure mode documented in real-world systems
   (e.g., Obermeyer et al., *Science* 2019).
3. **Single random seed.** All values are deterministic at seed = 42; future
   versions may include alternative seeds.

## Contact

Rohith Reddy Bellibatlu — `rohithreddybc@gmail.com` — ORCID:
[0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)
