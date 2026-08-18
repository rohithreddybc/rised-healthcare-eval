# RISED and the case-mix critique of subgroup AUROC-gap fairness tests

This repository has two things in it. `rised/` is a Python library that evaluates
clinical prediction models across five dimensions (Reliability, Inclusivity,
Sensitivity, Equity, Deployability) and produced the ten fitted models used
throughout the analysis below. `recompute/` is the analysis pipeline behind
the manuscript *Subgroup AUROC-gap tests are not evidence of unfairness in
clinical prediction models: case-mix differences make the equal-discrimination
null false* (submitted to *BMC Medical Research Methodology*). The question
the manuscript answers: when a fairness audit reduces subgroup AUROC to a
cut-point, a ratio rule, or a permutation test, is a flag evidence of an unfair
model, or can a perfectly fair model produce the same flag just because its
subgroups differ in case mix? The pipeline runs seven such procedures against
simulated data where the answer is known by construction, against 21
demographic partitions of five real clinical cohorts, and against a
calibration-based alternative, and reports what each procedure actually
measures.

If you are here to reproduce one table or figure from the manuscript, skip to
[Reproduction instructions](#reproduction-instructions).

## Directory map

| Directory | Contents |
|---|---|
| `recompute/` | The analysis pipeline: simulation, permutation-null and comparator code (`recompute/comparators/`), and the scripts that render `docs/*.md` from the result files |
| `recompute/results/` | Every CSV, JSON and cached simulation cell the pipeline produces; tracked in git so the committed numbers do not depend on re-running anything |
| `docs/` | Methods documentation and the generated evaluation reports (case-mix null, permutation-null specification, comparator evaluation, case-mix attribution, SD-ratio robustness, replacement metrics) |
| `rised/` | The RISED evaluation library: reliability, inclusivity, sensitivity, equity and deployability metrics with bootstrap confidence intervals, and the policy layer that turns them into advisory verdicts |
| `examples/` | Runnable per-cohort evaluation scripts, one per dataset, plus the synthetic-cohort generator and demo notebook |
| `tests/` | The pytest suite for `rised/` and `recompute/` |
| `verification/` | An independent numerical check of six mathematical propositions used elsewhere in the analysis (`verify_p1.py`-`verify_p6.py`); self-contained, not modified as part of this repository's other pipelines |
| `data/` | Local cache of the Folktables ACS-Income cohort. Not tracked in git; see [Data provenance](#data-provenance) |
| `brfss_cache/` | Local cache of the CDC BRFSS 2024 cohort. Not tracked in git |
| `nhanes_cache/` | Local cache of the NCHS NHANES 2021-2023 cohort. Not tracked in git |
| `nhis_cache/` | Local cache of the NCHS NHIS 2023/2024 cohorts. Not tracked in git |
| `mimic_ed_demo/` | Local cache of the public MIMIC-IV-ED demo subset. Not tracked in git |

## Environment and installation

Results in this repository were produced with:

| Package | Version |
|---|---|
| Python | 3.11.7 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| fairlearn | 0.13.0 |

numpy, scipy and pandas are built against OpenBLAS (`openblas64`, version
0.3.23.dev; full detail in `recompute/results/comparator_runtime_env.json`,
including the exact SIMD extensions available at build time). If your
conda-forge or pip wheel links a different BLAS, results that are exact by
construction (the case-mix quadrature, the closed-form Lum estimators) will
not move, and permutation p-values and bootstrap intervals should not move
outside their reported Monte Carlo error, but bit-identical reproduction is
only guaranteed with the same BLAS.

Two ways to install:

```bash
# exact reproduction (recommended for checking a specific number)
conda env create -f environment.yml && conda activate rised
# or, without conda:
python -m venv .venv && . .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements-lock.txt
pip install -e .

# general use (version floors, not exact pins)
pip install -r requirements.txt
pip install -e .
```

`requirements-lock.txt` pins every dependency to the exact version above.
`environment.yml` pins the same versions for conda. `requirements.txt` gives
version floors for anyone who wants a more current environment and does not
need bit-for-bit reproduction.

## Reproduction instructions

Each row is one display item in the manuscript. `results file` is what the
command writes; it is already in this repository, so you can inspect it
without running anything. Commands are given relative to the repository root.

| Display item | What it shows | Command | Results file |
|---|---|---|---|
| Table 1 (measured case-mix geometry, `tab:anchor`) | The per-level linear-predictor SD ratio measured in 21 real clinical partitions, and its stability under 24 refit specifications | `python -m recompute.comparators.sd_ratio_robustness && python -m recompute.comparators.sd_ratio_report` | `recompute/results/cohort_sd_ratios.csv`, `recompute/results/sd_ratio_robustness.csv`, rendered in `docs/sd_ratio_robustness.md` |
| Table 2 (flag rates under case mix, `tab:casemix`) and Figure 2 (`fig:casemix`) | False-flag rate of all seven procedures on a perfectly fair, Bayes-optimal model, as case-mix magnitude increases | `python -m recompute.comparators.type1 --sims 1000 --perm 999 --jobs 10 --force && python -m recompute.comparators.report` | `recompute/results/comparator_type1.csv` (case-mix rows), rendered in `docs/case_mix_null.md` and tables T14-T17 of `docs/comparator_evaluation_tables.md` |
| Table 3 (recommended alternatives, `tab:alternatives`) | Type I error and mechanism-discrimination of subgroup Cox calibration and model-based concordance on the same sweep and matched pairs | `python -m recompute.comparators.replacement_study --sims 1000 --perm 999 --jobs 6 --rules m30,ev10 && python -m recompute.comparators.replacement_report > docs/replacement_metrics_evaluation.tables.md` | `recompute/results/replacement_metrics.csv`, rendered in `docs/replacement_metrics_evaluation.md` |
| Table 4 (Type I error, `tab:type1`) | Type I error at nominal 0.05 under 17 equal-true-AUROC null geometries (simple and composite), all seven procedures | same run as Table 2 | `recompute/results/comparator_type1.csv`, tables T11-T13 of `docs/comparator_evaluation_tables.md` |
| Table 5 (verdict grid, `tab:verdicts`) | Which of the ten cohorts each procedure flags, at five subgroup-admissibility rules | `python -m recompute.comparators.run --stage cohorts --reps 10000 --jobs 5 && python -m recompute.comparators.report` | `recompute/results/comparator_comparison.csv`, tables T1-T2 of `docs/comparator_evaluation_tables.md` |
| Figure 1 (false-flag rate vs. case-mix magnitude, `fig:sweep`) | The same false-flag rate as Table 2, swept over 16 SD-ratio values instead of 6 fixed geometries | `python -m recompute.comparators.run_casemix_sweep --jobs 10` | `recompute/results/casemix_sweep.csv`, `recompute/results/casemix_positive_control.csv` |
| Figure 3 (admissibility silences minority strata, `fig:silencing`) | Which subgroup levels each admissibility rule excludes, by demographic axis and reason | `python -m recompute.run_all && python -m recompute.aggregate` | `recompute/results/excluded_subgroups.csv`, rendered in `docs/cohort_evaluation_results.md` |
| Supplementary Table S1 (software version manifest, `tab:versions`) | Exact package versions and hardware | no command; already recorded | `recompute/results/comparator_runtime_env.json`, `requirements-lock.txt` |
| Supplementary Table S2 (cohort and model provenance, `tab:cohorts`) | Estimator class, hyperparameters and split for each of the ten fitted models | `python -m recompute.model_provenance` | `recompute/results/model_provenance.csv`, `.json` |
| Permutation-null specification (referenced throughout, not separately numbered) | Joint vs. independent permutation, subgroup-inclusion-rule sensitivity, the combined clinical test | `python -m recompute.run_null_joint --jobs 5 --reps 10000 && python -m recompute.aggregate_null_joint && python -m recompute.report_null_joint` | `recompute/results/null_joint/`, `null_comparison_joint.csv`, `null_sweep_mmin.csv`, `null_joint_combined.csv`, rendered in `docs/permutation_null_specification.md` |
| Case-mix attribution on real cohorts (supports the Conclusions' "we could not decompose observed gaps") | How much of each cohort's observed subgroup AUROC gap model-based concordance attributes to case mix, and how much is left over | `python -m recompute.comparators.cohort_casemix --boot 2000 --seed 42` | `recompute/results/model_based_concordance.csv`, rendered in `docs/case_mix_attribution.md` |

A reviewer checking one number: pick a row, run its command, and diff the
result file it names against the copy already in the repository. Every
generated `.md` file in `docs/` carries a `<!-- generated by ... -->` comment
at the top if it is not meant to be hand-edited.

## Runtime expectations

All measured on: Intel Core, Family 6 Model 186, 8 physical / 12 logical
cores, 13.7 GB RAM, Windows 10.0.26200, Python 3.11.7 (`recompute/results/comparator_runtime_env.json`).

| Step | Runtime | Notes |
|---|---|---|
| `recompute.comparators.type1 --sims 1000 --perm 999 --jobs 10 --force` | about 107 minutes | 46 geometry/rule cells; checkpointed per cell in `recompute/results/type1_cells/`, so an interrupted run resumes |
| `recompute.comparators.run_casemix_sweep --jobs 10` | not separately benchmarked; budget on the order of the Type I run above | 32 SD-ratio-sweep cells plus 9 positive-control cells at the same per-cell cost; checkpointed in `recompute/results/casemix_sweep_cells/` |
| `recompute.comparators.replacement_study --jobs 6` | not separately benchmarked; budget more than the Type I run, fewer workers | same 23-geometry scope as the Type I study |
| `recompute.comparators.run --stage cohorts --reps 10000 --jobs 5` | under 2 minutes, all ten cohorts | dominated by the shared permutation draw; per-method detail in table T7 of `docs/comparator_evaluation_tables.md` |
| `recompute.run_null_joint --jobs 5 --reps 10000` | about 12 minutes total, all ten cohorts, run as 5 parallel subprocesses | per-cohort detail in `docs/permutation_null_specification.md`, section 6 |
| `recompute.run_all` then `recompute.aggregate` | about 87 minutes CPU, dominated by Diabetes 130 (about 40 minutes) | the BCa bootstrap's delete-one-unit jackknife is O(n_test) replicates of an O(n_test) statistic; quadratic in the test split size |
| `recompute.comparators.cohort_casemix --boot 2000 --seed 42` | about 21 minutes | 9 of 10 cohorts contribute rows; UCI Heart has no partition with two admissible levels |
| `recompute.comparators.sd_ratio_robustness` | about 6 minutes | 25 model-class/seed specifications across 10 cohorts |
| `recompute.model_provenance` | seconds | loads ten already-fitted models and reads their hyperparameters, no fitting |

## Data provenance

| Cohort | Source | Access | Cached at |
|---|---|---|---|
| Synthetic baseline | Generated in-process (`rised.datasets.generate_synthea_cohort`) | Public, no download | n/a, regenerated deterministically |
| UCI Heart Disease (Cleveland) | UCI ML Repository via `sklearn`'s OpenML fetcher | Public, downloads automatically | scikit-learn's OpenML cache |
| UCI Diabetes 130-US Hospitals | UCI ML Repository via OpenML | Public, downloads automatically | scikit-learn's OpenML cache |
| NCHS NHIS 2024, NHIS 2023 | CDC/NCHS public use files | Public, downloads automatically on first use | `nhis_cache/` |
| NCHS NHANES 2021-2023 | CDC/NCHS public use files | Public, downloads automatically | `nhanes_cache/` |
| CDC BRFSS 2024 | CDC public use file | Public, downloads automatically | `brfss_cache/` |
| Statlog German Credit, UCI Adult Income | UCI ML Repository via OpenML | Public, downloads automatically | scikit-learn's OpenML cache |
| Folktables ACS-Income (CA 2018) | US Census ACS PUMS via the `folktables` package | Public, downloads automatically | `data/2018/` |
| MIMIC-IV-ED | PhysioNet | **Credentialed.** The public demo subset (`mimic-iv-ed-demo-2.2`) is used in `examples/external_validation_mimic_ed.py` and runs offline; the full cohort requires a PhysioNet data use agreement and is not part of this repository's reproducible results | `mimic_ed_demo/` (demo only) |

Every table and figure listed under [Reproduction instructions](#reproduction-instructions)
reproduces fully offline once the caches above are populated: none of it
touches MIMIC-IV-ED. The caches themselves are not tracked in git (see
`.gitignore`); the first script that needs a cohort downloads and caches it,
and every later run of any script is offline. If you already have this
repository's caches present (as they are on the machine that produced these
results), nothing downloads at all.

## The test suite

```bash
pytest                         # full suite, 478 tests
pytest -m "not slow"           # 457 tests, skips cohort-level checks; ~2.5 minutes
```

The suite covers: every `rised/` metric function against closed-form and
simulated ground truth; the permutation-null seeding fix and its
reproducibility across process pools (`test_type1_reproducibility.py`, 88
tests); the case-mix quadrature against Monte Carlo; the model-based-
concordance bootstrap's estimand consistency (`test_mbc_bootstrap.py`); the
comparator implementations (DiCiccio, Lum, four-fifths) against fairlearn and
against each other's closed forms; and the reporting layer (p-value floor
rendering, cross-cohort multiplicity, rule-stability churn detection).

Tests marked `slow` load every real cohort and fit every model; they are what
makes the full run cohort-dependent rather than pure unit tests, and they are
the ones deselected by `-m "not slow"` for a fast local loop. CI
(`.github/workflows/tests.yml`) runs the full suite including `slow` on
Python 3.9, 3.10 and 3.11.

What the suite does not cover: it does not re-run the full-scale studies in
the reproduction table above (that would make CI take hours), so it checks
the *code paths* those studies use on small inputs, not the published numbers
themselves. Reproducing the published numbers means running the commands in
the table and diffing against the checked-in result files, not running
`pytest`. Two functions in `recompute/comparators/simulate.py`,
`verify_case_mix` and the Gauss-Hermite branch of `_apply_unfair`, have no
dedicated test; their outputs are used as diagnostics rather than reported
numbers, and this is called out explicitly in the module so it is not silent.
`verification/` is a separate, independently written check of six
propositions and is not run by `pytest`; run its six scripts directly if you
want that layer checked too.

## Licence and citation

MIT licence, see `LICENSE`.

Citation metadata is in `CITATION.cff`. Version 0.3.0 is archived on Zenodo at
[10.5281/zenodo.21918249](https://doi.org/10.5281/zenodo.21918249), which is the
DOI the manuscript cites. The concept DOI
[10.5281/zenodo.21918248](https://doi.org/10.5281/zenodo.21918248) always
resolves to the most recent published version.

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21918249-blue)](https://doi.org/10.5281/zenodo.21918249)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11.7](https://img.shields.io/badge/python-3.11.7-blue)](requirements-lock.txt)

```bibtex
@unpublished{bellibatlu2026subgroupauroc,
  title   = {Subgroup {AUROC}-gap tests are not evidence of unfairness in
             clinical prediction models: case-mix differences make the
             equal-discrimination null false},
  author  = {Bellibatlu, Rohith Reddy and Singh, Manpreet and
             Jajoo, Yash and Israni, Abhishek and Joshi, Rahul},
  note    = {Manuscript under review at BMC Medical Research Methodology.
             A preprint is posted via Springer Nature In Review on
             Research Square},
  year    = {2026}
}
```

If you use the RISED library independently of the case-mix manuscript, cite
the software entry in `CITATION.cff` instead; the dataset behind the
synthetic cohort has its own DOI, 10.57967/hf/8734, on
[HuggingFace](https://huggingface.co/datasets/Rohithreddybc/rised-healthcare-eval-dataset).

## Known limitations

**The permutation-null studies use a fixed subgroup-admissibility rule
(`m30`) as their published reference point, and the choice matters.**
`docs/permutation_null_specification.md` shows that switching from a
row-count rule to an events-based rule (at least 10 observations in each
outcome class) changes which cohorts the incumbent permutation test flags.
Row-count thresholds between 20 and 50 make almost no difference; the axis
that matters is rows versus events, and that axis was never varied before
this analysis.

**The minimum detectable effect at `m30` is large.** Four of five estimable
clinical cohorts cannot detect a subgroup AUROC gap below 0.23-0.37 at that
rule. A negative result from a procedure with that MDE is not evidence of
absence, and none of the reported "does not flag" results in this repository
should be read that way without checking the MDE for that specific cell.

**Model-based concordance cannot decompose most observed gaps precisely.**
`docs/case_mix_attribution.md` reports a median case-mix-attributable
fraction near 0.29 across the 27 of 64 real-cohort partitions where the
fraction is even estimable, with a median 95% interval 0.72 wide. The
recalibrated variant of the same estimator is close to a tautology (it
reproduces the observed AUROC gap by construction once predictions are
recalibrated within subgroup) and must not be read as a decomposition; this
is stated in the module docstring as well as here.

**The subgroup Cox calibration test is only shown valid where the model's
scores are the true conditional probability.** That holds in the simulated
case-mix geometries by construction. It has not been validated as a fairness
test on a fitted model whose scores are merely well calibrated in the
ordinary sense, and the manuscript does not claim it has.

**Two code paths have no dedicated regression test**, noted in the test
suite section above: `verify_case_mix` in `recompute/comparators/simulate.py`
and its Gauss-Hermite integration branch. Both are exercised indirectly
through the geometries that use them, but a targeted test asserting their
documented tolerances does not exist yet.

**Two of the case-mix geometries move slightly between the `m30` and `ev10`
admissibility rules** even though most case-mix geometries are rule-invariant
by construction: `casemix_location_3`'s smallest level sits near the event
threshold and is dropped by `ev10` in a small fraction of replicates. This is
documented in `docs/case_mix_null.md` and does not change the direction of
any finding, only the third decimal place of two cells.

**The external-cohort figures quoted in `docs/cohort_evaluation_results.md`'s
comparison against an earlier measurement pipeline (labelled 0.1.0) are shown
for methodological context, not as a claim that 0.1.0's numbers were ever
independently validated.** They come from the same code that produced the
corrected numbers, run with the older, less careful settings, so the
comparison isolates the measurement change; it is not a comparison against
an external ground truth.

## RISED library quickstart

```python
import rised
from rised.datasets import load_synthea_cohort, train_baseline_model
from sklearn.model_selection import train_test_split

X, y, demo = load_synthea_cohort()
X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
    X, y, demo, test_size=0.20, random_state=42, stratify=y)
model = train_baseline_model(X_tr, y_tr)

report = rised.evaluate_all(
    model, X_te, y_te, d_te,
    perturbation_specs=[{"type": "gaussian_noise", "scale": 0.05, "random_state": 0}],
    tau_ref=0.5, random_state=42, n_bootstrap=1000,
)
print(report.measurement_summary())
```

`rised.evaluate_all` returns metrics and confidence intervals only; it ships
no default cut-points and does not certify a model as safe to deploy. Full
API in `docs/api_reference.md`, the five dimensions' formal definitions in
`docs/framework.md`.

## Contact

Rohith Reddy Bellibatlu, [rbell084@fiu.edu](mailto:rbell084@fiu.edu),
ORCID [0009-0003-6083-0364](https://orcid.org/0009-0003-6083-0364).
