# RISED Framework

**Reliability · Inclusivity · Sensitivity · Equity · Deployability**

[![Dataset on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-yellow)](https://huggingface.co/datasets/Rohithreddybc/rised-synthetic-cohort-10k)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)

A structured pre-deployment evaluation framework for AI-assisted decision-support
systems in healthcare. RISED operationalizes the testing requirements implied by
the FDA AI/ML Action Plan, the ONC HTI-1 rule, the EU AI Act, and TRIPOD+AI
through five dimensions, each with formally specified pass/fail criteria and
bootstrap confidence intervals.

> 📄 Paper: *Evaluating AI-Assisted Decision Support in High-Stakes Healthcare:
> A Framework for Reliability, Inclusivity, Sensitivity, Equity, and Deployability*
> — under review at *Artificial Intelligence in Medicine*
> (arXiv: ARXIV_ID_PLACEHOLDER)

## Quickstart

```bash
pip install -e .
```

```python
import rised
from rised.datasets import load_synthea_cohort, train_baseline_model
from sklearn.model_selection import train_test_split

X, y, demo = load_synthea_cohort()
X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
    X, y, demo, test_size=0.20, random_state=42, stratify=y)
model = train_baseline_model(X_tr, y_tr)

specs = [
    {"type": "gaussian_noise", "scale": 0.05, "random_state": 0, "label": "noise_5pct"},
    {"type": "gaussian_noise", "scale": 0.10, "random_state": 1, "label": "noise_10pct"},
    {"type": "unit_rescaling", "feature_index": 0, "factor": 1.05, "label": "age_+5pct"},
    {"type": "unit_rescaling", "feature_index": 0, "factor": 1.06, "label": "age_+6pct"},
]

report = rised.evaluate_all(
    model, X_te, y_te, d_te,
    perturbation_specs=specs,
    random_state=42, n_bootstrap=1000,
)

print(report.summary())  # {'reliability': False, 'inclusivity': False, ...}
```

## The Five Dimensions

| Dimension | Primary metric | Pass threshold |
|-----------|----------------|----------------|
| **R**eliability   | Judge Sensitivity Score (JSS) | < 0.05 |
| **I**nclusivity   | AUC parity gap Δ_AUC          | ≤ 0.05 |
| **S**ensitivity   | Max threshold flip rate       | ≤ 10%  |
| **E**quity        | Need-prediction ρ             | ≥ 0.70 |
| **D**eployability | Mean inference latency Λ      | ≤ 500 ms |

## Dataset

The 10,000-patient synthetic cohort used in the paper's demonstration is
available on HuggingFace:

🤗 **[Rohithreddybc/rised-synthetic-cohort-10k](https://huggingface.co/datasets/Rohithreddybc/rised-synthetic-cohort-10k)**

```python
from datasets import load_dataset
ds = load_dataset("Rohithreddybc/rised-synthetic-cohort-10k")
```

Or regenerate deterministically:

```python
from rised.datasets import generate_synthea_cohort
df = generate_synthea_cohort(n=10000, random_state=42)
```

## Reproducing the Paper's Results

The paper applies RISED to an XGBoost classifier (AUROC 0.961, Brier 0.073) on
the 2,000-patient held-out test split. Three of five dimensions fail:

| Dimension | Value | 95% CI | Status |
|-----------|------:|:------:|:------:|
| JSS              | 0.064  | [0.058, 0.070]   | FAIL |
| Δ_AUC            | 0.059  | [0.052, 0.097]   | FAIL |
| Max TFR          | 19.9%  | [18.3%, 21.7%]   | FAIL |
| ρ_need           | 0.732  | —                | PASS |
| Λ (per cohort)   | ~1 ms  | —                | PASS |

Bootstrap CIs from 1,000 iterations with `random_state=42`.

## Tests

```bash
pytest --tb=short -q
```

81 tests, ~93% coverage.

## Citation

If you use RISED, please cite:

```bibtex
@article{bellibatlu2025rised,
  author  = {Bellibatlu, Rohith Reddy},
  title   = {Evaluating AI-Assisted Decision Support in High-Stakes Healthcare:
             A Framework for Reliability, Inclusivity, Sensitivity, Equity,
             and Deployability},
  year    = {2025},
  journal = {Artificial Intelligence in Medicine (under review)},
  url     = {https://github.com/rohithreddybc/rised-healthcare-eval}
}
```

## License

MIT — see [LICENSE](LICENSE).
