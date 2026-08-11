# Model-based-concordance case-mix attribution: diagnosis and fix

Scope: `recompute/comparators/cohort_casemix.py` (`mbc_rows`, `_bootstrap_fraction`)
and its output `recompute/results/model_based_concordance.csv`.

## 1. Diagnosis

### 1.1 The reported defect: an interval that cannot contain its point estimate

The cited failures are real and reproduce exactly:

| cohort | partition | rule | reported fraction | reported 95% CI |
|---|---|---|---|---|
| nhis2024 | age_group | m30 | 1.183 | [-0.38, 0.37] |
| nhanes2123 | age_group | m30 | 0.866 | [0.00, 0.51] |
| diabetes130 | race | m30 | 1.169 | [-0.51, 0.22] |

**Cause: the interval belongs to a different column than the point estimate.**

`mbc_rows` writes *two* point estimates per partition:

* `casemix_attributable_fraction` = `mbc_gap_aligned / observed_gap`, from raw
  model-based concordance;
* `casemix_attributable_fraction_recalibrated` =
  `mbc_gap_aligned_recalibrated / observed_gap`, from MBC computed on predictions
  after a within-level Cox recalibration `y ~ a + b logit(s)`.

`_bootstrap_fraction` computes **only the raw variant** — it calls
`model_based_concordance(sb)` on the resampled scores and never recalibrates.
The single resulting interval (`boot_fraction_lo95` / `hi95`) is then written
next to *both* point estimates.

Every one of the three cited numbers is a **recalibrated** point estimate paired
with a **raw-MBC** interval. In the current CSV: 0 of 64 partitions violate
containment for the raw fraction, and **26 of 64 violate it for the recalibrated
fraction**. That is the whole signature. Confirmed by re-running the bootstrap
with recalibration refit inside each replicate — containment is then restored in
every case:

| cohort / partition | recal. point | raw-MBC CI (as shipped) | recal. CI (refit per replicate) |
|---|---|---|---|
| nhis2024 age_group | +1.183 | [-0.383, +0.370] ✗ | [+0.576, +2.284] ✓ |
| diabetes130 race | +1.169 | [-0.505, +0.220] ✗ | [+0.461, +2.545] ✓ |
| nhanes2123 age_group | +0.866 | [+0.005, +0.510] ✗ | [+0.666, +1.215] ✓ |
| brfss2024 race | +0.956 | [-0.127, +0.379] ✗ | [+0.857, +1.044] ✓ |

### 1.2 Second defect: winner's-curse inflation of the denominator

`_bootstrap_fraction` re-selects `k_hi` / `k_lo` by **bootstrap** AUROC in every
replicate, while the point estimate fixes them from the observed AUROC. Strictly
this is a self-consistent plug-in bootstrap of the max-min functional, so it does
not by itself break containment — but the max-min of noisy AUROCs is biased
upward, so the denominator is systematically inflated and the ratio is dragged
toward zero:

| cohort / partition | observed gap | mean bootstrap gap, re-selected | mean bootstrap gap, levels fixed |
|---|---|---|---|
| diabetes130 race | 0.1236 | 0.1651 (+34%) | 0.1255 |
| nhis2024 age_group | 0.3279 | 0.3719 (+13%) | 0.3429 |
| brfss2024 race | 0.2237 | 0.2339 (+5%) | 0.2211 |

The `argmax`/`argmin` functional is also non-smooth, where the nonparametric
bootstrap has no consistency guarantee.

### 1.3 Third defect: the ratio has an unstable denominator

Observed gaps run from 0.0037 to 0.328; the lower decile is 0.0097. Two-level
partitions (`sex`, `insured`, `health_plan`) produce intervals such as
[-27.4, +16.2]. A ratio with a denominator indistinguishable from zero is not a
reportable quantity.

### 1.4 The finding that matters most: the 93% headline is close to a tautology

The manuscript's "case mix explains a median 93% of observed subgroup AUROC gaps"
is the median of `casemix_attributable_fraction_recalibrated` (m30 0.941,
ev10 0.940) — precisely the column whose interval was mismatched.

That column cannot support the claim, for a reason independent of the bootstrap.
A Cox recalibration `a + b logit(s)` with `b > 0` is a **monotone** transform, so
it leaves the observed AUROC of the level *exactly* unchanged, while forcing the
predictions to be calibrated in that level. Model-based concordance of calibrated
predictions then reproduces that level's observed AUROC almost exactly. Measured
over all 224 admissible levels in the current CSV:

| quantity | median abs. difference from observed AUROC | correlation |
|---|---|---|
| raw MBC | 0.0162 | 0.935 |
| recalibrated MBC | **0.0032** | **0.994** |

So `mbc_gap_aligned_recalibrated` ≈ `observed_gap` by construction, and the
"attributable fraction" is driven to ≈ 1 by the recalibration step rather than by
case mix. The recalibrated fraction measures almost nothing.

The informative quantity is the **raw** MBC fraction — MBC of the model's actual
predictions, which is what van Klaveren et al. (2016) defines. Its median is
0.235 (m30) / 0.344 (ev10), not 0.93.
