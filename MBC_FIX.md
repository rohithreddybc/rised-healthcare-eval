# Model-based-concordance case-mix attribution: diagnosis and fix

Scope: `recompute/comparators/cohort_casemix.py` (`mbc_rows`, `_bootstrap_fraction`, now `_bootstrap_partition`)
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

## 2. What changed

`recompute/comparators/cohort_casemix.py`; regression tests in
`tests/test_mbc_bootstrap.py`. `verification/` untouched.

### 2.1 Same estimand for the point estimate and every replicate

* **Recalibration is refit inside every replicate.** `_bootstrap_partition` now
  returns a raw *and* a recalibrated replicate vector, and each point estimate is
  paired with its own interval. This is the fix for the reported defect.
* **The extreme pair is selected once, on the observed data, and held fixed**
  across replicates. The shipped code re-picked `argmax`/`argmin` per replicate,
  which inflated the denominator by a median 16% and up to 6.2x. The interval is
  therefore *conditional on the observed selection* — it does not cover the
  uncertainty in which levels are extreme. The re-selecting max-min functional is
  retained as `fraction_selection_inclusive_*` for sensitivity only.
* **Replicates are no longer filtered on `gap > 0`.** Discarding replicates where
  the observed ordering reverses conditioned the distribution away from the
  point estimate.

### 2.2 A difference, not a ratio, is the primary summary

Observed gaps run 0.004-0.328 (lower decile 0.0097). Dividing by a denominator
indistinguishable from zero produced intervals such as [-27.4, +16.2]: 28 of 64
shipped intervals were wider than 2.0, one was 92.5 wide.

The primary quantity is now the **residual gap** -- `observed_gap -
mbc_gap_aligned`, the part of the observed gap that case mix does *not* explain,
in AUROC units. It has no denominator and is defined for all 64 partitions.
The fraction is still reported, but only where `fraction_reportable` is true:
the bootstrap 2.5th percentile of the gap must exceed zero. That holds for
**27 of 64** partitions.

### 2.3 Numerics and reproducibility

* Cox recalibration refit by guarded Newton (step-halving, coefficient bound,
  stable `expit`) rather than `scipy.optimize.minimize`; this removed the
  `overflow encountered in exp` warnings and made B=2000 affordable.
* Seeds pinned per `(cohort, rule, partition)` via `_partition_seed`. Previously
  one generator was threaded through the whole loop, so an interval depended on
  iteration order and on `--only`.
* **Bootstrap replicates raised 400 -> 2000** (not reduced). Monte Carlo error of
  the interval endpoints is now reported: median MC SE 0.007 (lower) and 0.025
  (upper) for the fraction, 0.0015 for the residual gap. The MC error is small
  relative to the interval widths below — the imprecision is statistical, not
  computational.

### 2.4 Regression test

`tests/test_mbc_bootstrap.py` asserts that **every reported 95% interval contains
its own point estimate**, on synthetic data and on the shipped CSV. Verified by
mutation: reintroducing the original defect (skipping recalibration inside
replicates) makes it fail with the exact shipped signature
(`fraction_recalibrated = 1.0023` against a raw CI of `[-0.126, 0.225]`).
Further tests cover the monotone/AUROC-preserving property of recalibration,
denominator inflation under re-selection, seed pinning, and determinism.

## 3. Before / after

| | before | after |
|---|---|---|
| bootstrap replicates | 400 | 2000 |
| partitions | 64 (9 cohorts) | 64 (9 cohorts) |
| **containment failures, raw fraction** | 0 / 64 | **0 / 27 reportable** |
| **containment failures, recalibrated fraction** | **26 / 64** | **0 / 27 reportable** |
| containment failures, residual gap | not reported | 0 / 64 |
| intervals wider than 2.0 | 28 / 64 | 0 / 27 reportable |
| widest interval | 92.5 | 3.34 |
| **median "attributable fraction"** | **0.941** (recalibrated) | **0.29** (raw, reportable only) |

The three cited failures, corrected:

| cohort / partition (m30) | before: point vs CI | after: fraction [95% CI] | after: residual gap [95% CI] |
|---|---|---|---|
| NHIS 2024 age_group | 1.183 vs [-0.38, 0.37] FAIL | -0.06 [-0.90, 0.21] | 0.348 [0.109, 0.634] |
| NHANES 21-23 age_group | 0.866 vs [0.00, 0.51] FAIL | 0.28 [0.09, 0.77] | 0.054 [0.007, 0.111] |
| Diabetes 130 race | 1.169 vs [-0.51, 0.22] FAIL | not reportable (gap CI includes 0) | 0.126 [-0.079, 0.316] |

All 15 reportable partitions under the published rule (m30):

| cohort | partition | obs gap | residual gap [95% CI] | case-mix fraction [95% CI] |
|---|---|---|---|---|
| NHIS 2024 | age_group | 0.328 | 0.348 [0.109, 0.634] | -0.06 [-0.90, 0.21] |
| BRFSS 2024 | race | 0.224 | 0.171 [0.107, 0.241] | 0.24 [0.05, 0.40] |
| Diabetes 130 | age_group | 0.216 | 0.014 [-0.078, 0.108] | 0.94 [0.65, 1.63] |
| NHIS 2023 | age_group | 0.128 | 0.047 [-0.072, 0.135] | 0.64 [0.33, 2.44] |
| NHIS 2024 | race | 0.109 | 0.077 [-0.002, 0.156] | 0.29 [-0.05, 0.97] |
| NHIS 2024 | income | 0.097 | 0.081 [0.005, 0.158] | 0.16 [-0.04, 0.67] |
| Adult Income | race | 0.081 | 0.106 [0.046, 0.169] | -0.31 [-2.94, 0.15] |
| NHANES 21-23 | age_group | 0.075 | 0.054 [0.007, 0.111] | 0.28 [0.09, 0.77] |
| Adult Income | age_group | 0.067 | 0.044 [0.031, 0.058] | 0.34 [0.26, 0.45] |
| Adult Income | sex | 0.058 | 0.047 [0.032, 0.062] | 0.18 [0.06, 0.29] |
| ACS-Income | age_group | 0.053 | 0.041 [0.018, 0.063] | 0.23 [0.07, 0.44] |
| BRFSS 2024 | income | 0.046 | 0.027 [-0.016, 0.070] | 0.41 [0.04, 1.98] |
| Synthetic baseline | insurance | 0.046 | 0.003 [-0.012, 0.018] | 0.94 [0.70, 1.38] |
| Synthetic baseline | age_group | 0.040 | 0.011 [-0.013, 0.035] | 0.73 [0.41, 1.91] |
| NHANES 21-23 | insured | 0.033 | 0.036 [0.016, 0.063] | -0.09 [-1.73, 0.27] |

## 4. What can now be claimed

**The "case mix explains a median 93% of observed subgroup AUROC gaps" claim
must be withdrawn.** It rested on the recalibrated fraction, which is
near-tautological (section 1.4) and was the column whose interval was mismatched.
It is not recoverable by fixing the bootstrap.

What the corrected analysis supports:

1. **Case mix explains a minority of the observed gap — a median of about 29%**
   (m30: 0.275; ev10: 0.346), across the 27 of 64 partitions whose denominator
   can carry a ratio. Roughly 60% of the typical observed gap is left unexplained
   (median residual gap 0.028 against a median observed gap 0.045).

2. **The estimate is imprecise, and the imprecision must be shown.** The median
   95% interval on the fraction is 0.72 wide; 11 of 27 are wider than 1.0. No
   single partition pins the fraction tightly except Adult Income `age_group`
   (0.34 [0.26, 0.45]) and `sex` (0.18 [0.06, 0.29]). The honest statement is a
   range, not a point: for most partitions the data are consistent with anything
   from "case mix explains almost none of it" to "case mix explains half".

3. **Some gaps are demonstrably not case mix.** For 18 of 64 partitions the
   residual-gap interval excludes zero — the model really does discriminate
   less well in some subgroups than its own predictions imply. The strongest is
   NHIS 2024 `age_group` (residual 0.348 [0.109, 0.634]), where the point
   estimate of the case-mix fraction is actually *negative*: case mix, if
   anything, predicts a gap in the opposite direction.

4. **Two partitions are consistent with case mix explaining the whole gap** --
   Diabetes 130 `age_group` (0.94 [0.65, 1.63]) and the synthetic baseline
   (by construction). These are the exception, not the median.

5. **No partition's interval contains both 0 and 1**, so the analysis is not
   vacuous — it does discriminate. But the direction of the finding is opposite
   to the withdrawn claim: case mix is a *partial* explanation, and for a
   substantial minority of partitions it is not the explanation at all.

The recalibrated columns remain in the CSV for transparency, flagged by
`max_abs_mbc_recalibrated_minus_obs_auc`, but must not be used as a headline.

## 5. Reproducing

```
python -m recompute.comparators.cohort_casemix --boot 2000 --seed 42
python -m pytest tests/test_mbc_bootstrap.py
```

Runtime about 21 minutes. `uci_heart` (n=61) has no partition with two
rule-admissible levels and contributes no rows, which is why 9 cohorts appear.
