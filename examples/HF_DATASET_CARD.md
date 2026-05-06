---
license: mit
language:
  - en
tags:
  - healthcare
  - clinical-ai
  - synthetic-data
  - algorithmic-fairness
  - clinical-decision-support
  - model-evaluation
pretty_name: RISED Synthetic Clinical Cohort (10K)
size_categories:
  - 1K<n<10K
task_categories:
  - tabular-classification
configs:
  - config_name: default
    data_files: synthetic_cohort_10k.csv
---

# RISED Synthetic Clinical Cohort (10,000 patients)

A fully synthetic adult clinical cohort generated deterministically
(random_state = 42) by a Synthea-inspired computational model implemented in
the [`rised`](https://github.com/rohithreddybc/rised-healthcare-eval) Python
package. **No real patient records were used at any stage.** The cohort is
intended as a methodological testbed for the RISED Framework and is
demographically heterogeneous to support subgroup-level evaluation.

This is the reference dataset used in the demonstration application of the
**RISED Framework** — a five-dimension pre-deployment evaluation framework for
clinical AI decision-support systems (Reliability, Inclusivity, Sensitivity,
Equity, Deployability).

## Dataset Summary

| Attribute | Value |
|-----------|-------|
| Patients (rows) | 10,000 |
| Features (cols) | 26 |
| Outcome prevalence | 30.0% positive class (3,000 patients) |
| License | MIT |
| Source code | [github.com/rohithreddybc/rised-healthcare-eval](https://github.com/rohithreddybc/rised-healthcare-eval) |
| Generator | Synthea-inspired, deterministic (seed = 42) |

## Demographic Composition

| Group | Full cohort | Outcome=1 (n=3,000) |
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
| Insurance: Public-major | 47.4% | 74.2% |
| Insurance: Public-secondary | 14.4% | 7.7% |
| Insurance: Private | 29.8% | 13.8% |
| Insurance: Uninsured | 8.4% | 4.4% |

Mean Charlson Comorbidity Index: 0.99 ± 1.20 (full); 1.86 ± 1.33 (outcome=1).

## Features

**Demographics:** `age`, `sex_male`, `age_group`, `sex`, `race`, `insurance`,
`ins_medicare`, `ins_medicaid`, `ins_private` (insurance-type indicators
included as a demographic axis on which to evaluate subgroup performance)

**Clinical:** `cci_score`, `has_hypertension`, `has_diabetes`, `has_chf`,
`has_ckd`, `has_copd`, `has_mi`, `has_cvd`, `has_dementia`, `has_cancer`,
`has_metastatic`, `prior_hosp_count`, `ed_visits_count`, `bmi`

**Neighborhood:** `adi_score` (deprivation index, 1–100 scale)

**Outcome:** `high_need` (binary; 1 = top-30% derived clinical risk score).
The column is named `high_need` for backward compatibility with earlier
versions of the codebase; it represents a generic adverse-clinical-outcome
label and the cohort is **not** specific to any particular clinical use case
or deployed risk-stratification program.

## Outcome Definition

The binary outcome label is derived from a logistic transformation of age,
diabetes, congestive heart failure, chronic kidney disease, COPD, prior
myocardial infarction, CCI, prior hospitalization count, ED utilization
count, and the deprivation index, with additive Gaussian noise (σ=0.5).
Patients in the top 30% of the predicted score receive label = 1.

**Important:** Because the outcome is derived directly from the feature
space, this dataset is suitable for evaluation framework demonstrations
but *not* for benchmarking model accuracy on a real-world prediction task.
Real EHR cohorts introduce distribution shifts and access-barrier
distortions absent from synthetic data.

## Intended Use

- **Primary:** Demonstrating the RISED Framework for pre-deployment
  evaluation of clinical AI decision-support systems.
- **Secondary:** Teaching, methodological development, and reproducibility
  benchmarking for fairness, calibration, and sensitivity tooling.

**Not intended for:** training production clinical models, benchmarking
discrimination performance against real-world systems, or any deployed
clinical use.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("Rohithreddybc/rised-synthetic-cohort-10k")
df = ds["train"].to_pandas()
print(df.shape)         # (10000, 26)
print(df["high_need"].mean())  # 0.30
```

Or load directly with pandas:

```python
import pandas as pd
df = pd.read_csv(
    "hf://datasets/Rohithreddybc/rised-synthetic-cohort-10k/synthetic_cohort_10k.csv"
)
```

To regenerate from source (deterministic):

```python
from rised.datasets import generate_synthea_cohort
df = generate_synthea_cohort(n=10000, random_state=42)
```

## Citation

If you use this dataset, please cite the accompanying paper:

```bibtex
@article{bellibatlu2026rised,
  author  = {Bellibatlu, Rohith Reddy},
  title   = {{RISED}: A Pre-Deployment Evaluation Framework for Clinical {AI}
             Decision-Support Systems Spanning Reliability, Inclusivity,
             Sensitivity, Equity, and Deployability},
  year    = {2026},
  journal = {Artificial Intelligence in Medicine (under review)},
  url     = {https://github.com/rohithreddybc/rised-healthcare-eval}
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
2. **Self-derived outcome.** The outcome label is a function of the feature
   space, so high accuracy is expected and does not indicate real predictive
   skill. Use for methodology evaluation only.
3. **Single random seed.** All values are deterministic at seed = 42; future
   versions may include alternative seeds.

## Contact

Rohith Reddy Bellibatlu — `rohithreddybc@gmail.com` — ORCID:
[0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364)
