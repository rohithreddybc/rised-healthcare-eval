# Model-based concordance: how much of the observed subgroup AUROC gap is case mix

Scope: `recompute/comparators/cohort_casemix.py` (`mbc_rows`, `_bootstrap_partition`), output `recompute/results/model_based_concordance.csv`.

## What this measures

Model-based concordance (MBC; van Klaveren et al. 2016) is the c-statistic a model's *own predictions* imply for the case mix it was applied to: treat each predicted risk as the truth and average pairwise concordance. Computed per subgroup, it answers the question the case-mix simulation (`docs/case_mix_null.md`) is a proxy for: of the observed max-min subgroup AUROC gap, how much would be there anyway with a model that discriminates equally well everywhere, purely because the subgroups' risk distributions differ?

The headline quantity is the **residual gap**,

    residual_gap = observed_gap - mbc_gap_aligned,

the part of the observed max-min subgroup AUROC gap that case mix does *not* explain, in AUROC units. It is a difference, so it is defined and stable for every partition. The attributable **fraction**, `mbc_gap_aligned / observed_gap`, is also reported, but only where the denominator can carry it: observed gaps in these ten cohorts run from 0.004 to 0.33, and a ratio whose denominator is indistinguishable from zero is arithmetic rather than evidence. `fraction_reportable` is true only when the bootstrap 2.5th percentile of the gap is above zero.

## Method

The pair of levels that produces the observed maximum and minimum is selected **once, on the observed data**, and held fixed inside every bootstrap replicate, so the point estimate and the replicates target the same estimand. The interval is therefore conditional on that selection; a sensitivity column re-selects the extreme pair per replicate for comparison, and its denominator is winner's-curse inflated (a re-selected max-min of noisy AUROCs is biased upward by a median 16% and up to 6.2x across these partitions).

Two variants of MBC are computed. The **raw** variant applies MBC directly to the model's fitted scores. The **recalibrated** variant first applies a within-level Cox recalibration, `y ~ a + b logit(s)`, and computes MBC on the recalibrated predictions; `_bootstrap_partition` recalibrates inside every bootstrap replicate, so each point estimate is paired with its own interval rather than sharing one computed only from the raw statistic. Cox recalibration is fit by guarded Newton (step-halving, coefficient bound, stable `expit`), which keeps B = 2000 replicates affordable and avoids overflow warnings.

Seeds are pinned per `(cohort, rule, partition)`, so an interval does not depend on iteration order. Replicates are not filtered on `gap > 0`; discarding replicates where the observed ordering reverses would condition the distribution away from the point estimate.

Monte Carlo error of the interval endpoints is reported: median MC SE 0.007 (lower) and 0.025 (upper) for the fraction, 0.0015 for the residual gap. The MC error is small relative to the interval widths below, so the imprecision in the results is statistical, not computational.

## Why recalibrated MBC cannot be the headline

A Cox recalibration `a + b logit(s)` with `b > 0` is a **monotone** transform, so it leaves the observed AUROC of the level *exactly* unchanged while forcing the predictions to be calibrated in that level. Model-based concordance of calibrated predictions then reproduces that level's observed AUROC almost exactly. Measured over all 224 admissible levels in the current CSV:

| quantity | median abs. difference from observed AUROC | correlation |
|---|---|---|
| raw MBC | 0.0162 | 0.935 |
| recalibrated MBC | 0.0032 | 0.994 |

So `mbc_gap_aligned_recalibrated` approximately equals `observed_gap` by construction, and an "attributable fraction" built from it is driven toward 1 by the recalibration step rather than by case mix. The recalibrated fraction is close to a tautology and is kept in the CSV for transparency (flagged by `max_abs_mbc_recalibrated_minus_obs_auc`), but it is not the quantity reported as the headline here. The informative quantity is the **raw** MBC fraction, which is what van Klaveren et al. (2016) defines.

## Results

27 of 64 partitions have a denominator that can carry a ratio (`fraction_reportable`). Median case-mix fraction across those, m30: 0.275; ev10: 0.346. Median residual gap 0.028 against a median observed gap 0.045, so roughly 60% of the typical observed gap is left unexplained by case mix.

Three worked examples, at the published rule (m30):

| cohort / partition | observed gap | residual gap [95% CI] | case-mix fraction [95% CI] |
|---|---|---|---|
| NHIS 2024, age_group | 0.328 | 0.348 [0.109, 0.634] | -0.06 [-0.90, 0.21] |
| NHANES 21-23, age_group | 0.075 | 0.054 [0.007, 0.111] | 0.28 [0.09, 0.77] |
| Diabetes 130, race | 0.124 | 0.126 [-0.079, 0.316] | not reportable (gap CI includes 0) |

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

## What the estimate supports

1. **Case mix explains a minority of the observed gap, a median of about 29%** (m30: 0.275; ev10: 0.346) across the 27 of 64 partitions whose denominator can carry a ratio.

2. **The estimate is imprecise, and the imprecision has to be shown alongside the point value.** The median 95% interval on the fraction is 0.72 wide; 11 of 27 are wider than 1.0. Only two partitions pin the fraction tightly: Adult Income `age_group` (0.34 [0.26, 0.45]) and `sex` (0.18 [0.06, 0.29]). For most partitions the data are consistent with anything from "case mix explains almost none of it" to "case mix explains half."

3. **Some gaps are demonstrably not case mix.** For 18 of 64 partitions the residual-gap interval excludes zero: the model discriminates less well in some subgroups than its own predictions imply. The clearest case is NHIS 2024 `age_group` (residual 0.348 [0.109, 0.634]), where the point estimate of the case-mix fraction is negative: case mix, if anything, predicts a gap in the opposite direction.

4. **Two partitions are consistent with case mix explaining the whole gap**: Diabetes 130 `age_group` (0.94 [0.65, 1.63]) and the synthetic baseline (by construction). These are the exception, not the median.

5. **No partition's interval contains both 0 and 1**, so the analysis is not vacuous. Case mix is a *partial* explanation of the observed gaps in this study, and for a substantial minority of partitions it explains none of the gap.

## Caveats

* Model-based concordance assumes the predictions are calibrated in the subgroup it is computed on. They frequently are not; that is exactly what the recalibrated variant controls for, and exactly why it cannot also serve as the headline (see above).
* Predictions are in-sample for the case mix (the same held-out rows the AUROC is computed on), so the residual gap and the observed AUROC gap are not statistically independent.
* The bootstrap resamples rows within (level, outcome). For Diabetes 130 a patient can contribute several rows to the test set even under the group split, so its interval is optimistic.

## Reproducing

```
python -m recompute.comparators.cohort_casemix --boot 2000 --seed 42
python -m pytest tests/test_mbc_bootstrap.py
```

Runtime about 21 minutes. `uci_heart` (n=61) has no partition with two rule-admissible levels and contributes no rows, which is why 9 of the 10 cohorts appear.
