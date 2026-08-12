# Is rho-hat robust to model specification?

`rho-hat` is the ratio of the largest to the smallest per-level standard
deviation of the linear predictor, measured across the 21 clinical
demographic partitions under the published `m30` inclusion rule. The
manuscript reports median **1.145**, min **1.022**, max **3.304**, all
from a single fitted model per cohort at seed 42.

This document refits every cohort under **4 model classes x 6 seeds = 24 specifications**,
varying the train/test split and the estimator initialisation together,
and recomputes rho-hat for every partition under every one.

## Verdict

**(a) The median.** Across the 24 refit specifications the
median rho-hat over the 21 partitions ranges from **1.132** to
**1.223**, with a median of **1.170**.
The published value is 1.145.

**(b) The age-versus-sex ordering.** Across all 120 (specification, cohort) pairs the age partition's
rho-hat exceeds the sex partition's in **116** (97%).
20 of 24 specifications reproduce the ordering in every cohort.

**(c) The maximum does NOT hold.** The largest rho-hat is
`diabetes130|age_group` in 23 of 24 specifications, so its *identity* is stable -- but its
*value* runs from **1.999** to **6.388**, a
factor of 3.2. The published 3.304 is one
draw from that range and is not a property of the cohort. This is the
one headline quantity the check does not support, and it is the one the
simulation's most extreme geometry was anchored to.

## Methods

### Model classes

| class | configuration | which cohorts published it |
|---|---|---|
| `xgb_published` | XGBoost, `n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8` -- the manuscript's own configuration, seed varied | the 6 clinical cohorts and `synthetic` |
| `logreg_l2` | `StandardScaler` + L2-penalised logistic regression, `C=1.0, max_iter=2000` | `adult_income`, `acs_income`, `german_credit` |
| `random_forest` | `n_estimators=300, min_samples_leaf=5, max_features='sqrt'` | none -- new |
| `hgb_deep` | `HistGradientBoostingClassifier, max_iter=300, learning_rate=0.10, max_leaf_nodes=63, l2_regularization=1.0` -- a deliberately different point in the boosting hyperparameter space | none -- new |

Every cohort gets all four, so every cohort has at least two classes
beyond its own. No configuration was tuned toward the published numbers;
the four are declared once in `recompute/refit.py` and applied unchanged.

### Seeds

`42, 43, 44, 45, 46, 47`. Each seed sets **both** the
train/test split's `random_state` **and** the estimator's own, so a seed
moves the resampling and the fit together. Diabetes 130 keeps its
`GroupShuffleSplit` on `patient_nbr` under every seed, so no refit
reintroduces the row-level leakage the group split exists to remove.

### Traceability

The published fit is carried as its own row (`model_class=published`) and
reproduces `recompute/results/cohort_sd_ratios.csv` to the last bit; the
test suite asserts it. The published split is **not** recoverable from
the loaders' outputs (`train_test_split` depends on row order and the
reconstruction reorders rows), so the seed-42 refit is a genuinely new
split rather than a re-run of the published one.

Total estimator fit time: **5.9 minutes** over 25 specifications x 10 cohorts.

## 1. Per-partition spread of rho-hat

Across the 24 refits, per partition. `published %ile` is where the
published value sits inside the refit distribution -- 50 means the
published fit was typical, 0 or 100 means it was an extreme.

| partition | specs | published | refit median | min | max | max/min | published %ile |
|---|---|---|---|---|---|---|---|
| `diabetes130|age_group` | 24 | 3.304 | 2.501 | 1.999 | 6.388 | 3.20 | 67 |
| `nhanes2123|age_group` | 24 | 1.602 | 1.774 | 1.132 | 2.409 | 2.13 | 38 |
| `brfss2024|age_group` | 24 | 1.285 | 1.556 | 1.168 | 1.922 | 1.65 | 25 |
| `nhis2023|age_group` | 24 | 1.348 | 1.442 | 1.220 | 1.808 | 1.48 | 42 |
| `nhis2024|age_group` | 24 | 1.362 | 1.416 | 1.023 | 2.296 | 2.25 | 50 |
| `nhanes2123|race` | 24 | 1.316 | 1.294 | 1.170 | 1.779 | 1.52 | 62 |
| `brfss2024|race` | 24 | 1.335 | 1.252 | 1.172 | 1.650 | 1.41 | 75 |
| `nhis2024|insurance` | 12 | 1.322 | 1.247 | 1.065 | 1.429 | 1.34 | 83 |
| `nhis2023|race` | 24 | 1.186 | 1.192 | 1.114 | 1.475 | 1.32 | 50 |
| `diabetes130|race` | 24 | 1.240 | 1.170 | 1.089 | 1.693 | 1.55 | 58 |
| `brfss2024|income` | 24 | 1.063 | 1.163 | 1.045 | 1.469 | 1.41 | 12 |
| `nhis2024|race` | 24 | 1.051 | 1.149 | 1.048 | 1.241 | 1.18 | 4 |
| `nhanes2123|insured` | 20 | 1.140 | 1.111 | 1.016 | 1.337 | 1.32 | 65 |
| `nhis2024|income` | 24 | 1.140 | 1.111 | 1.037 | 1.262 | 1.22 | 58 |
| `brfss2024|sex` | 24 | 1.113 | 1.098 | 1.043 | 1.197 | 1.15 | 71 |
| `nhis2023|insured` | 24 | 1.037 | 1.094 | 1.010 | 1.288 | 1.28 | 25 |
| `nhis2024|sex` | 24 | 1.145 | 1.078 | 1.015 | 1.190 | 1.17 | 88 |
| `nhanes2123|sex` | 24 | 1.090 | 1.077 | 1.003 | 1.244 | 1.24 | 58 |
| `brfss2024|health_plan` | 24 | 1.029 | 1.063 | 1.001 | 1.246 | 1.24 | 25 |
| `nhis2023|sex` | 24 | 1.022 | 1.051 | 1.001 | 1.147 | 1.15 | 33 |
| `diabetes130|gender` | 24 | 1.022 | 1.030 | 1.006 | 1.191 | 1.18 | 33 |

2 partition(s) have fewer than 24 specifications: the `m30` inclusion rule drops a level when a refit's split pushes a small stratum below the threshold, so the partition ceases to be evaluable. That is itself instability in the anchor -- the *set* of 21 partitions is not fixed across specifications either.

### The headline summary, one row per specification

| model class | seed | n partitions | median | min | max |
|---|---|---|---|---|---|
| `hgb_deep` | 42 | 20 | 1.188 | 1.019 | 2.450 |
| `hgb_deep` | 43 | 21 | 1.170 | 1.006 | 1.999 |
| `hgb_deep` | 44 | 20 | 1.202 | 1.023 | 2.474 |
| `hgb_deep` | 45 | 21 | 1.184 | 1.029 | 2.308 |
| `hgb_deep` | 46 | 20 | 1.196 | 1.014 | 2.126 |
| `hgb_deep` | 47 | 20 | 1.162 | 1.015 | 2.502 |
| `logreg_l2` | 42 | 20 | 1.217 | 1.026 | 3.386 |
| `logreg_l2` | 43 | 21 | 1.223 | 1.010 | 6.388 |
| `logreg_l2` | 44 | 20 | 1.170 | 1.018 | 5.359 |
| `logreg_l2` | 45 | 21 | 1.197 | 1.040 | 6.248 |
| `logreg_l2` | 46 | 20 | 1.165 | 1.001 | 3.341 |
| `logreg_l2` | 47 | 20 | 1.189 | 1.006 | 3.339 |
| `random_forest` | 42 | 20 | 1.191 | 1.010 | 2.213 |
| `random_forest` | 43 | 21 | 1.141 | 1.003 | 2.296 |
| `random_forest` | 44 | 20 | 1.180 | 1.011 | 2.501 |
| `random_forest` | 45 | 21 | 1.197 | 1.048 | 2.152 |
| `random_forest` | 46 | 20 | 1.160 | 1.015 | 2.139 |
| `random_forest` | 47 | 20 | 1.138 | 1.014 | 2.235 |
| `xgb_published` | 42 | 20 | 1.156 | 1.001 | 3.310 |
| `xgb_published` | 43 | 21 | 1.156 | 1.010 | 2.493 |
| `xgb_published` | 44 | 20 | 1.155 | 1.010 | 3.392 |
| `xgb_published` | 45 | 21 | 1.135 | 1.001 | 3.007 |
| `xgb_published` | 46 | 20 | 1.145 | 1.016 | 2.961 |
| `xgb_published` | 47 | 20 | 1.132 | 1.015 | 3.073 |
| **published** | 42 | 21 | 1.145 | 1.022 | 3.304 |

Splitting that by what moved: within a single class, changing only the
seed moves the median by
* `hgb_deep`: 1.162 to 1.202 (sd over 6 seeds 0.0153)
* `logreg_l2`: 1.165 to 1.223 (sd over 6 seeds 0.0236)
* `random_forest`: 1.138 to 1.197 (sd over 6 seeds 0.0252)
* `xgb_published`: 1.132 to 1.156 (sd over 6 seeds 0.0109)

and switching class moves the *class* medians across 1.150 to 1.193. Seed and class contribute
comparably; neither alone accounts for the spread.

### Does the maximum, 3.304, hold?

No. The partition that carries the maximum is nearly always the same one, but the number is not.

| model class | min | median | max |
|---|---|---|---|
| `hgb_deep` | 1.999 | 2.379 | 2.502 |
| `logreg_l2` | 3.339 | 4.372 | 6.388 |
| `random_forest` | 2.105 | 2.183 | 2.501 |
| `xgb_published` | 2.493 | 3.040 | 3.392 |
| **published** | 3.304 | 3.304 | 3.304 |

The penalised logistic regression puts this partition between 3.339 and 6.388; the random forest puts it between 2.105 and 2.501. Those two ranges do not overlap. Whatever else rho-hat is at this partition, it is not a measurement of the cohort.

8 of 488 refit rows have a rho-hat beyond the sweep grid's last node (3.167); their induced flag rate is clamped to the endpoint and flagged in the CSV. Every one of them is already at or near a flag rate of 1.0, so the clamp does not change any conclusion -- but the sweep carries no information out there and the manuscript's most extreme anchor now sits outside it.

## 2. Does the ordinal age-versus-sex pattern survive?

2 partition(s) are not admissible under every specification and are excluded from the rank analysis: `nhanes2123|insured`, `nhis2024|insurance`.

### Overall concordance of the partition ordering

* Kendall's W over 24 specifications ranking 19 partitions: **0.692**
  (independent random rankings give mean 0.041, 95th percentile 0.065 over 2000 draws).
* Pairwise Spearman between specifications (276 pairs): median **0.701**, 5th percentile 0.417, min 0.274;
  14% of pairs below 0.5, 49% below 0.7. Worst pair: `logreg_l2|s47 vs random_forest|s42`.
* Spearman of each refit against the **published** ordering: median **0.771**, min 0.618, max 0.916.

### The claim itself: age above sex, within cohort

The claim is ordinal and paired, so the direct test is: in each cohort
that has both partitions, is rho-hat(age) > rho-hat(sex)?

| cohort | specs | age > sex | frac | median log(age/sex) | min | max |
|---|---|---|---|---|---|---|
| `brfss2024` | 24 | 24 | 1.00 | 0.290 | 0.068 | 0.567 |
| `diabetes130` | 24 | 24 | 1.00 | 0.903 | 0.687 | 1.761 |
| `nhanes2123` | 24 | 24 | 1.00 | 0.437 | 0.121 | 0.821 |
| `nhis2023` | 24 | 24 | 1.00 | 0.320 | 0.159 | 0.471 |
| `nhis2024` | 24 | 20 | 0.83 | 0.274 | -0.152 | 0.787 |

Published fit: 5 of 5 cohorts.
Across refits: **116 of 120** (specification, cohort) pairs, 97%.
Per-specification paired sign test over cohorts: 20 of 24 specifications reach p <= 0.05.

### What survives and what does not

**The age-versus-sex contrast survives and does not need withdrawing.**
It holds in 97% of 120 (specification, cohort) pairs, in 4 of 5 cohorts without a single exception, and the
effect is large on the log scale in every cohort. It is the one claim
here that the refits strengthen rather than weaken.

**The finer ordering does not.** Kendall's W of 0.692 is far above chance but far below reproducible: 49% of specification pairs rank the
partitions at Spearman below 0.7 and 14% below 0.5. Any claim that reads off
the ordering *between* the middle partitions -- which race partition
sits above which income partition, say -- is not supported. Only the
coarse age-high / sex-low contrast is.

## 3. The induced false-flag-rate distribution

Each rho-hat is mapped onto the existing `casemix_sweep.csv` curve by
linear interpolation in the SD ratio. `permutation_null` is the
incumbent -- the manuscript's own procedure. Nominal level is 0.05, so
the manuscript's "median roughly double nominal" claim is a median
near 0.10.

| method | published median | refit: median of spec medians | min | max | specs with median > 0.10 | sweep MC SE |
|---|---|---|---|---|---|---|
| `permutation_null` | 0.094 | 0.109 | 0.086 | 0.151 | 19 / 24 | 0.0090 |
| `diciccio2020` | 0.093 | 0.107 | 0.086 | 0.148 | 16 / 24 | 0.0090 |
| `lum2022` | 0.117 | 0.135 | 0.107 | 0.186 | 24 / 24 | 0.0098 |
| `fixed_threshold_005` | 0.378 | 0.414 | 0.359 | 0.491 | 24 / 24 | 0.0146 |
| `four_fifths` | 0.000 | 0.000 | 0.000 | 0.000 | 0 / 24 | 0.0000 |

The sweep itself has Monte-Carlo error: 1,000 simulations per grid node,
so a flag rate near 0.10 carries an SE near 0.009. Differences between
specifications smaller than about 0.02 are not resolvable by this curve.

## 4. Cohort or model? A variance decomposition

Balanced three-way crossed random-effects ANOVA of `log rho-hat` over
(19 partitions) x (4 model classes) x (6 seeds), one observation per cell. The residual is the
three-way interaction and is not separately identified from it.

| component | variance | share of positive total |
|---|---|---|
| partition (the cohort side) | 0.05301 | 71.4% |
| model class | -0.00021 | 0.0% |
| seed / split | -0.00010 | 0.0% |
| partition x class | 0.01503 | 20.2% |
| partition x seed | 0.00174 | 2.3% |
| class x seed | 0.00013 | 0.2% |
| residual (3-way interaction) | 0.00435 | 5.9% |

* Cohort side (partition main effect): **71.4%** of the positive variance.
* Model side (everything else): **28.6%**.
* One-way check: a model that knows only *which partition it is* explains **76.4%** of the total sum of squares in `log rho-hat`.
* Negative component estimates summing to -0.00032 were set to zero for the share column only; they are reported as estimated in the variance column.

The partition main effect is the largest single component, so the
editor's framing is half right: rho-hat *is* substantially a property of
the cohort-partition. But the model side is not a rounding error, and
almost all of it is the **partition x class interaction** (20.2%), not a class main effect
(0.0%). That is the worst shape this
could have taken. A class main effect would mean every model class
rescales rho-hat by roughly the same factor, and the *ordering* -- the
part the manuscript actually uses -- would be untouched. An interaction
means the class changes rho-hat by different amounts at different
partitions, which is precisely what reorders them and precisely what
makes a per-partition value like 3.304 non-transferable.

## Does the anchor survive?

**Partly, and the parts must be separated.**

| claim | verdict |
|---|---|
| median rho-hat ~ 1.145 | **survives**: 1.132-1.223 across 24 specifications. Report it as a range, not a point. |
| min rho-hat ~ 1.022 | **survives**: the floor is near 1.0 under every specification, as it must be -- a ratio of standard deviations cannot go below 1. It was never an informative number. |
| max rho-hat = 3.304 | **does not survive**: 1.999-6.388 across specifications, and non-overlapping between model classes. |
| age partitions high, sex partitions low | **survives**, strongly. |
| the full 21-partition ordering | **does not survive** beyond the coarse age/sex contrast. |
| induced false-flag median ~ double nominal | **survives** as a statement about the median; the range across specifications is wider than the sweep's own Monte-Carlo error, so it is a range too. |
| rho-hat is a property of the cohort | **mostly true but not safe to assume**: 71% cohort, 29% model, and the model share is concentrated in the interaction that moves individual partitions. |

The honest summary is that the *distributional* claims about rho-hat --
its median, its ordinal age/sex contrast, the induced flag rate near
twice nominal -- hold up under refitting, while the *per-partition*
claims do not. The manuscript should stop quoting 3.304, quote the
median as a range over specifications, and state that the anchor was
measured under a specification panel rather than a single fit.

---

Generated by `python -m recompute.comparators.sd_ratio_report`. Inputs:
`recompute/results/sd_ratio_robustness.csv` (produced by
`python -m recompute.comparators.sd_ratio_robustness`) and
`recompute/results/casemix_sweep.csv`.
