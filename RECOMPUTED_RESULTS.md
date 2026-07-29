# Recomputed RISED results: 0.1.0 → 0.2.0

Every cohort re-run under the corrected measurement pipeline, against the same data, split, seed and fitted model as the 0.1.0 run, so each difference is attributable to the measurement change alone.

Reproduce with `python -m recompute.run_all` then `python -m recompute.aggregate`. Seeds fixed at 42; B = 1000 bootstrap replicates with a full delete-one-unit jackknife for the BCa acceleration (delete-one-*patient* on Diabetes 130).

> **On the word "failure".** 0.2.0 withdrew the PASS/FAIL gate, so nothing here is a library verdict. "Failure" means only "exceeds the illustrative cut-point the paper's claim was stated against": JSS ≥ 0.05, max per-partition ΔAUC > 0.05, max TFR > 10%. None of those cut-points has been calibrated against deployment outcomes.

## Bottom line

**The corrections do not leave the empirical case standing in the form the paper states it.**

Under the corrected pipeline, on ten cohorts:

* **Reliability fails nowhere.** Every 0.1.0 Reliability failure was driven by the multiplicative age rescalings now reclassified as covariate shift, plus Gaussian noise applied to binary and categorical columns. With semantics-preserving perturbations only and a typed schema, the highest JSS across all ten cohorts is 0.0328, against a 0.05 cut-point. The dimension does not discriminate between these models at all.
* **Inclusivity fails in name on 7 cohorts, but only 1 survives its own equality null.** That one is Cross-domain: Folktables ACS-Income (CA 2018) — a non-clinical cross-domain demo, at ΔAUC 0.053 against a 0.05 cut-point. **No clinical cohort produces an Inclusivity gap distinguishable from chance.** Two cohorts measure a gap *smaller* than what pure selection over their subgroups produces.
* **Sensitivity still fails on several cohorts**, but max TFR never reads `y_true`, so it cannot be read as a performance failure at all, and a constant predictor scores a perfect 0 on it. It is the weakest of the three as evidence.
* **Equity is effectively unmeasured.** Nine of the ten proxies are model input features and one is the outcome's own diagnostic criterion. The single genuinely independent proxy in the whole study — German Credit's savings status — gives ρ = −0.019, i.e. nothing.

The synthetic cohort's collapse (JSS 0.064→0.011, TFR 19.9%→7.9%, ΔAUC 0.059→0.046) was not special to the synthetic cohort. The same three corrections, plus the equality null, remove all of the clinical evidence too. The Diabetes 130 result the README calls "decisive" — ΔAUC 0.262, max TFR 49.1% — becomes ΔAUC 0.216 (p = 0.15 against its own null) and max TFR 1.4% once the patient identity is restored, the band is narrowed and the exclusion rule is applied consistently.

What survives is narrower and worth stating precisely: **the metrics do measure quantities that AUROC cannot see** — TFR and JSS are functionals that never touch the labels. What the data no longer supports is that those quantities **detect deployment-relevant failures** in these cohorts. The instrument reads something real; the readings on this evidence base are not distinguishable from what an equal-performance model would produce.

## The three questions, answered

### 1. Which failures persist under correct measurement, and which were artifacts?

| Dimension | Failed under 0.1.0 and still fails | Failed under 0.1.0, passes under 0.2.0 (artifact) | Failed under 0.1.0, no longer evaluable |
|---|---|---|---|
| Reliability (JSS ≥ 0.05) | *none* | Synthetic baseline (Synthea-inspired, 10k), UCI Heart Disease (Cleveland) | — |
| Inclusivity (ΔAUC > 0.05) | UCI Diabetes 130-US Hospitals (grouped on patient_nbr), NCHS NHIS 2024 (Sample Adult, CHD/MI), NCHS NHIS 2023 (Sample Adult, diabetes), NCHS NHANES 2021-2023 (diabetes, with lab HbA1c), CDC BRFSS 2024 (CHD/MI), Cross-domain: UCI Adult Income, Cross-domain: Folktables ACS-Income (CA 2018) | Synthetic baseline (Synthea-inspired, 10k), Cross-domain: Statlog German Credit | UCI Heart Disease (Cleveland) |
| Sensitivity (max TFR > 10%) | UCI Heart Disease (Cleveland), CDC BRFSS 2024 (CHD/MI), Cross-domain: UCI Adult Income, Cross-domain: Folktables ACS-Income (CA 2018), Cross-domain: Statlog German Credit | Synthetic baseline (Synthea-inspired, 10k), UCI Diabetes 130-US Hospitals (grouped on patient_nbr), NCHS NHIS 2024 (Sample Adult, CHD/MI), NCHS NHIS 2023 (Sample Adult, diabetes) | — |

"No longer evaluable" is not a pass. It means the n ≥ 30 rule, now applied in the point estimate as well as the intervals, leaves fewer than two estimable subgroups in every partition, so the parity gap has no value at all. The 0.1.0 figure for such a cohort was computed over subgroups the same release already knew were too small to trust.

### 2. For persistent failures, is the effect large relative to the null?

The generic reference from `verification/results/p2_summary.json` is mean **0.0889** and p95 **0.1304** at 10 subgroups of 500 under exact equality. Each cohort is additionally measured against a null built from its *own* partition geometry, which is the fairer comparison: a cohort of two-level partitions has a much smaller null than that grid cell, and a cohort with seven small race levels has a larger one.

* Inclusivity gaps that clear both the 0.05 cut-point **and** their own cohort null (above p95, one-sided p < 0.05): **Cross-domain: Folktables ACS-Income (CA 2018)**.
* Inclusivity gaps above 0.05 but **not** separable from the equality null: UCI Diabetes 130-US Hospitals (grouped on patient_nbr), NCHS NHIS 2024 (Sample Adult, CHD/MI), NCHS NHIS 2023 (Sample Adult, diabetes), NCHS NHANES 2021-2023 (diabetes, with lab HbA1c), CDC BRFSS 2024 (CHD/MI), Cross-domain: UCI Adult Income. For these the number is consistent with selection bias over subgroups and is not evidence of disparity.
* Cohorts whose measured gap is *below* its own null mean — i.e. smaller than what pure selection over subgroups produces at that geometry: NCHS NHIS 2023 (Sample Adult, diabetes), Cross-domain: Statlog German Credit.

### 3. Does any cohort still show a failure that aggregate AUROC would miss?

Taking "AUROC would miss it" to mean the model looks good on the aggregate number (AUROC ≥ 0.80) while some dimension still exceeds its cut-point — and, for Inclusivity, survives its own equality null:

* **Inclusivity**, evidential and AUROC-invisible: Cross-domain: Folktables ACS-Income (CA 2018)
* **Sensitivity** (max TFR > 10% on the narrow band): UCI Heart Disease (Cleveland), Cross-domain: UCI Adult Income, Cross-domain: Folktables ACS-Income (CA 2018)
* **Reliability** (JSS ≥ 0.05 on semantics-preserving perturbations only): *none*

Threshold flip rate and JSS never read `y_true` at all — TFR is a functional of the score CDF alone — so a failure on either is *by construction* invisible to a discrimination metric. That is the strongest form the claim can take, but it cuts both ways: the same independence makes TFR gameable, since a constant predictor scores a perfect 0 while being useless. TFR must be read next to AUROC, never instead of it, and a TFR finding is weaker evidence for the paper's thesis than an Inclusivity finding would be.

## Headline table, all cohorts

| Cohort | n (test) | AUROC | JSS old→new | ΔAUC pooled (old) | ΔAUC per-partition (new) | null mean / p95 | p | TFR wide (old) → narrow (new) | Equity |
|---|---:|---:|---|---:|---|---|---:|---|---|
| Synthetic baseline (Synthea-inspired, 10k) | 10,000 (2,000) | 0.961 | 0.0644 → 0.0107 | 0.0588 | 0.0456 [0.030, 0.056] | 0.035 / 0.057 | 0.157 | 19.9% → 7.9% | ρ 0.599 (cci_proxy) |
| UCI Heart Disease (Cleveland) | 303 (61) | 0.867 | 0.0779 → 0.0328 | 0.1183 | — — | — / — | — | 34.4% → 11.5% | ρ -0.383 (chol_proxy) |
| UCI Diabetes 130-US Hospitals (grouped on patient_nbr) | 99,492 (19,890) | 0.639 | 0.0008 → 0.0008 | 0.2160 | 0.2160 [0.129, 0.298] | 0.155 / 0.261 | 0.153 | 48.4% → 1.4% | ρ 0.757 (n_inpatient_proxy) |
| NCHS NHIS 2024 (Sample Adult, CHD/MI) | 9,747 (1,950) | 0.836 | 0.0113 → 0.0046 | 0.3279 | 0.3279 [0.252, 0.596] | 0.211 / 0.357 | 0.087 | 22.5% → 5.2% | ρ 0.505 (genhlth_proxy) |
| NCHS NHIS 2023 (Sample Adult, diabetes) | 27,114 (5,423) | 0.839 | 0.0169 → 0.0040 | 0.1834 | 0.1290 [0.065, 0.174] | 0.187 / 0.352 | 0.741 | 32.5% → 9.3% | ρ 0.724 (genhlth_proxy) |
| NCHS NHANES 2021-2023 (diabetes, with lab HbA1c) | 4,096 (820) | 0.964 | 0.0265 → 0.0104 | 0.0748 | 0.0748 [0.038, 0.143] | 0.069 / 0.118 | 0.312 | 9.8% → 2.9% | ρ 0.826 (hba1c_proxy) |
| CDC BRFSS 2024 (CHD/MI) | 44,888 (8,978) | 0.767 | 0.0359 → 0.0099 | 0.2334 | 0.2237 [0.143, 0.279] | 0.147 / 0.234 | 0.066 | 64.2% → 18.3% | ρ 0.409 (physhlth_proxy) |
| Cross-domain: UCI Adult Income | 45,222 (9,045) | 0.890 | 0.0119 → 0.0185 | 0.1081 | 0.0808 [0.055, 0.112] | 0.072 / 0.136 | 0.346 | 33.9% → 13.1% | ρ 0.520 (education_proxy) |
| Cross-domain: Folktables ACS-Income (CA 2018) | 20,000 (4,000) | 0.866 | 0.0207 → 0.0186 | 0.0527 | 0.0527 [0.027, 0.076] | 0.016 / 0.028 | 0.000 | 39.4% → 17.6% | ρ 0.637 (schl_proxy) |
| Cross-domain: Statlog German Credit | 1,000 (200) | 0.655 | 0.0088 → 0.0025 | 0.0587 | 0.0291 [0.001, 0.042] | 0.105 / 0.207 | 0.942 | 88.5% → 37.5% | ρ -0.019 (savings_proxy) |

## What happened to each headline failure

| Cohort | Reliability (JSS ≥ 0.05) | Inclusivity (ΔAUC > 0.05) | Sensitivity (max TFR > 10%) |
|---|---|---|---|
| Synthetic baseline (Synthea-inspired, 10k) | artifact (was failing, now passing) | artifact (was failing, now passing) | artifact (was failing, now passing) |
| UCI Heart Disease (Cleveland) | artifact (was failing, now passing) | not evaluable | persists |
| UCI Diabetes 130-US Hospitals (grouped on patient_nbr) | passed before and after | persists | artifact (was failing, now passing) |
| NCHS NHIS 2024 (Sample Adult, CHD/MI) | passed before and after | persists | artifact (was failing, now passing) |
| NCHS NHIS 2023 (Sample Adult, diabetes) | passed before and after | persists | artifact (was failing, now passing) |
| NCHS NHANES 2021-2023 (diabetes, with lab HbA1c) | passed before and after | persists | passed before and after |
| CDC BRFSS 2024 (CHD/MI) | passed before and after | persists | persists |
| Cross-domain: UCI Adult Income | passed before and after | persists | persists |
| Cross-domain: Folktables ACS-Income (CA 2018) | passed before and after | persists | persists |
| Cross-domain: Statlog German Credit | passed before and after | artifact (was failing, now passing) | persists |

## Inclusivity gaps against the equality null

`verification/results/p2_summary.json` establishes that the max−min AUC range has a strictly positive expectation when every subgroup shares the same true AUC: at 10 subgroups of 500 the mean is **0.0889** with p95 **0.1304**. That grid is generic, so each cohort is also given its own null, computed by permuting subgroup labels within outcome classes — which preserves every subgroup's size and prevalence while forcing equal true AUC — at the cohort's real partition geometry, 2000 replicates.

| Cohort | included subgroups | observed ΔAUC (per-partition) | cohort null mean | cohort null p95 | excess | one-sided p | above null p95? |
|---|---:|---:|---:|---:|---:|---:|---|
| synthetic | 14 | 0.0456 | 0.0345 | 0.0572 | +0.0110 | 0.157 | no |
| uci_heart | 1 | — | — | — | — | — | not evaluable |
| diabetes130 | 16 | 0.2160 | 0.1547 | 0.2611 | +0.0613 | 0.153 | no |
| nhis2024 | 19 | 0.3279 | 0.2107 | 0.3569 | +0.1172 | 0.087 | no |
| nhis2023 | 15 | 0.1290 | 0.1867 | 0.3517 | -0.0577 | 0.741 | no |
| nhanes2123 | 13 | 0.0748 | 0.0685 | 0.1177 | +0.0063 | 0.312 | no |
| brfss2024 | 24 | 0.2237 | 0.1470 | 0.2337 | +0.0767 | 0.066 | no |
| adult_income | 9 | 0.0808 | 0.0717 | 0.1360 | +0.0091 | 0.346 | no |
| acs_income | 6 | 0.0527 | 0.0155 | 0.0277 | +0.0372 | 0.000 | **yes** |
| german_credit | 4 | 0.0291 | 0.1046 | 0.2070 | -0.0754 | 0.942 | no |

Read the p-value column first. Only one cohort is below 0.05. Six measure a gap above the 0.05 cut-point that a model with *identical* subgroup performance would produce at least 6.6% of the time at that cohort's own geometry, and two measure a gap smaller than the null average.

## Every subgroup excluded by the n ≥ 30 / estimability rule

0.1.0 applied this rule inconsistently: subgroups with n < 30 were *flagged* but still entered the point estimate, while the bootstrap and jackknife dropped them — so the interval targeted a different parameter from the estimate it was attached to. 0.2.0 applies one rule in all three places. The last column shows which of these subgroups were silently inflating the 0.1.0 pooled gap.

| Cohort | Subgroup | n | positives | negatives | Rule | Was in the 0.1.0 point estimate? |
|---|---|---:|---:|---:|---|---|
| synthetic | `age_group=18-44` | 371 | 1 | 370 | degenerate labels | no |
| uci_heart | `age_group=51-60` | 28 | 15 | 13 | n < 30 | **yes** |
| uci_heart | `age_group=<=50` | 18 | 12 | 6 | n < 30 | **yes** |
| uci_heart | `age_group=>60` | 15 | 6 | 9 | n < 30 | **yes** |
| uci_heart | `sex=F` | 19 | 16 | 3 | n < 30 | **yes** |
| diabetes130 | `age_group=[0-10)` | 31 | 0 | 31 | degenerate labels | no |
| nhis2024 | `race=NH-AIAN` | 23 | 0 | 23 | n < 30 | no |
| nhis2024 | `race=NH-AIAN+other` | 17 | 1 | 16 | n < 30 | no |
| nhis2024 | `race=NH-Other/Multi` | 18 | 0 | 18 | n < 30 | no |
| brfss2024 | `age_group=18-24` | 44 | 0 | 44 | degenerate labels | no |

## Equity: what a valid proxy costs

0.2.0 refuses `y_true` as the need proxy, because with a binary outcome proxy ρ = √(12p(1−p))·(n/√(n²−1))·(AUROC−0.5) exactly, so the statistic is an affine reparameterisation of discrimination and cannot fail independently of it. Every 0.1.0 equity number in the study was computed that way.

Replacing it requires a proxy that is genuinely independent. Of the ten cohorts, one has one.

| Cohort | Proxy | Class | ρ (0.1.0, `y_true`) | ρ (0.2.0, proxy) | Note |
|---|---|---|---:|---:|---|
| Synthetic baseline (Synthea-inspired, 10k) | `cci_proxy (Charlson comorbidity index)` | **model_input** | 0.732 | 0.599 | CCI is a model input feature and enters the label's data-generating process; rho is partly mechanical. |
| UCI Heart Disease (Cleveland) | `chol_proxy (serum cholesterol)` | **model_input** | 0.633 | -0.383 | A model input feature. Not outcome-derived, but not independent of the score either. |
| UCI Diabetes 130-US Hospitals (grouped on patient_nbr) | `n_inpatient_proxy (prior inpatient visits)` | **model_input** | 0.154 | 0.757 | A model input feature and the dominant predictor of readmission; rho largely re-expresses that dependence. |
| NCHS NHIS 2024 (Sample Adult, CHD/MI) | `genhlth_proxy (self-rated general health)` | **model_input** | 0.307 | 0.505 | A model input feature. |
| NCHS NHIS 2023 (Sample Adult, diabetes) | `genhlth_proxy (self-rated general health)` | **model_input** | 0.370 | 0.724 | A model input feature. |
| NCHS NHANES 2021-2023 (diabetes, with lab HbA1c) | `hba1c_proxy (HbA1c)` | **outcome_defining** | 0.541 | 0.826 | HbA1c >= 6.5% is the diagnostic criterion for the outcome (diabetes) and is also a model input. The proxy is not outcome-INDEPENDENT in the sense F8 requires, even though the library's structural guard does not reject it. Equity should be treated as not evaluable on this cohort. |
| CDC BRFSS 2024 (CHD/MI) | `physhlth_proxy (days of poor physical health)` | **model_input** | 0.378 | 0.409 | A model input feature. |
| Cross-domain: UCI Adult Income | `education_proxy (education-num)` | **model_input** | 0.584 | 0.520 | A model input feature. |
| Cross-domain: Folktables ACS-Income (CA 2018) | `schl_proxy (educational attainment)` | **model_input** | 0.623 | 0.637 | A model input feature. |
| Cross-domain: Statlog German Credit | `savings_proxy (savings-status ordinal)` | **independent** | 0.245 | -0.019 | Carried alongside the split and never given to the model. The only genuinely independent proxy in the study. |

`model_input` means the proxy is a legitimate measurement but also one of the model's own predictors, so ρ partly measures the model against its own input rather than against need. `outcome_defining` means the proxy is part of the diagnostic criterion for the outcome; the library's structural guard does not reject it, but it is not outcome-independent in the sense F8 requires, and Equity should be treated as **not evaluable** there. Only `independent` supports the dimension as specified.

## Method, and what would change the answer

* **Both columns come from the same fitted model on the same split.** Data preparation, split, seed and hyperparameters are transcribed from the `examples/` scripts unchanged, so AUROC and Brier are identical in both columns by construction and every difference is a measurement difference.
* **Diabetes 130 is the one deliberate departure.** `patient_nbr` is retained and the split is a `GroupShuffleSplit` on it, with the clustered bootstrap and delete-one-patient jackknife. Both the 0.1.0 and the 0.2.0 columns are computed on that group split so the comparison still isolates the measurement change; the published 0.1.0 figures, which came from a row-level split that leaks 42.1% of test rows, are reported separately in the per-cohort section.
* **τ₀ = 0.5 is used for the headline** so the before/after difference is attributable to the band change alone. Several cohorts have prevalence near 0.1, where 0.5 is not an operating point anyone would deploy; the prevalence-matched alternative is reported per cohort as a point estimate.
* **The equality null conditions on the observed scores.** It permutes subgroup labels within outcome classes, so it holds every subgroup's size and prevalence fixed and asks only whether the measured spread exceeds what pure selection over that many groups of those sizes produces. It reproduces the published p2 headline cell to within 0.001 (`python -m recompute.null_reference`).
* **No bootstrap replicates were reduced.** B = 1000 everywhere, with the full delete-one-unit jackknife the BCa acceleration requires. Nothing here is a shortened run.

### Offline availability and runtime

All 10 cohorts ran offline from caches already in the working tree: the sklearn OpenML cache (UCI Heart, Diabetes 130, Adult, German Credit), `examples/adult24.csv`, `nhis_cache/adult23.csv`, `nhanes_cache/*.xpt`, `brfss_cache/LLCP2024.XPT` and `data/2018/1-Year/psam_p06.csv`. No cohort required network access.

MIMIC-IV-ED is absent by necessity, not oversight: the full cohort needs PhysioNet credentials, and only the public demo is present.

| Cohort | test rows | wall clock |
|---|---:|---:|
| Synthetic baseline (Synthea-inspired, 10k) | 2,000 | 2.4 min |
| UCI Heart Disease (Cleveland) | 61 | 0.2 min |
| UCI Diabetes 130-US Hospitals (grouped on patient_nbr) | 19,890 | 39.9 min |
| NCHS NHIS 2024 (Sample Adult, CHD/MI) | 1,950 | 4.2 min |
| NCHS NHIS 2023 (Sample Adult, diabetes) | 5,423 | 7.9 min |
| NCHS NHANES 2021-2023 (diabetes, with lab HbA1c) | 820 | 1.8 min |
| CDC BRFSS 2024 (CHD/MI) | 8,978 | 17.8 min |
| Cross-domain: UCI Adult Income | 9,045 | 8.9 min |
| Cross-domain: Folktables ACS-Income (CA 2018) | 4,000 | 3.2 min |
| Cross-domain: Statlog German Credit | 200 | 0.5 min |
| **total CPU** | | **87 min** |

Runtime is dominated by the BCa jackknife, which is delete-one-unit and so costs O(n_test) replicates of an O(n_test) statistic — quadratic in the test split. Run with `--jobs 5` the wall clock is roughly the longest single cohort.

## Per-cohort detail

### Synthetic baseline (Synthea-inspired, 10k)

n = 10,000 (test 2,000); prevalence 0.300; 4 demographic partitions (`age_group`, `sex`, `race`, `insurance`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.9610 | 0.9610 | — | unchanged (same model, same split) |
| Brier | 0.0732 | 0.0732 | — | unchanged |
| JSS / PSS | 0.0644 | **0.0107** | [0.0074, 0.0145] | artifact (was failing, now passing) |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.0588 | 0.0588 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.0456** | [0.0302, 0.0562] | widest partition `insurance` |
| Max TFR — wide band [0.10, 0.90] | 19.9% | 19.9% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **7.9%** | [6.8%, 9.1%] | artifact (was failing, now passing) |
| Equity ρ | 0.7317 (proxy = `y_true`) | 0.5994 (proxy = `cci_proxy`) | — | 8-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 1.21 ms | 1.29 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.376 ms** | — | new in 0.2.0 |

Per-partition gaps: `insurance` 0.0456, `age_group` 0.0398, `race` 0.0309, `sex` 0.0083.

**Against the equality null for this cohort's own partition geometry:** null mean 0.0345, p95 0.0572; observed 0.0456, one-sided p = 0.157, excess over the null mean +0.0110.

*Equity proxy*: cci_proxy (Charlson comorbidity index) — **model_input**. CCI is a model input feature and enters the label's data-generating process; rho is partly mechanical.

Subgroups excluded by the n≥30 / estimability rule:

| Subgroup | n | positives | negatives | Reason | In the 0.1.0 point estimate? |
|---|---:|---:|---:|---|---|
| `age_group=18-44` | 371 | 1 | 370 | degenerate labels (n_pos=1, n_neg=370); AUC undefined | no |

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.4640 (rather than the 0.5 convention) the narrow-band max TFR is 7.6% and the wide-band max TFR is 18.6%. Point estimates only.


### UCI Heart Disease (Cleveland)

n = 303 (test 61); prevalence 0.541; 2 demographic partitions (`sex`, `age_group`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.8669 | 0.8669 | — | unchanged (same model, same split) |
| Brier | 0.1503 | 0.1503 | — | unchanged |
| JSS / PSS | 0.0779 | **0.0328** | [0.0082, 0.0902] | artifact (was failing, now passing) |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.1183 | — | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **—** | — | widest partition `None` |
| Max TFR — wide band [0.10, 0.90] | 34.4% | 34.4% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **11.5%** | [3.3%, 19.7%] | persists |
| Equity ρ | 0.6334 (proxy = `y_true`) | -0.3833 (proxy = `chol_proxy`) | — | 56-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 0.61 ms | 0.64 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.429 ms** | — | new in 0.2.0 |

*Equity proxy*: chol_proxy (serum cholesterol) — **model_input**. A model input feature. Not outcome-derived, but not independent of the score either.

Subgroups excluded by the n≥30 / estimability rule:

| Subgroup | n | positives | negatives | Reason | In the 0.1.0 point estimate? |
|---|---:|---:|---:|---|---|
| `age_group=51-60` | 28 | 15 | 13 | n=28 < min_subgroup_n=30 | yes |
| `age_group=<=50` | 18 | 12 | 6 | n=18 < min_subgroup_n=30 | yes |
| `age_group=>60` | 15 | 6 | 9 | n=15 < min_subgroup_n=30 | yes |
| `sex=F` | 19 | 16 | 3 | n=19 < min_subgroup_n=30 | yes |

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.8091 (rather than the 0.5 convention) the narrow-band max TFR is 18.0% and the wide-band max TFR is 26.2%. Point estimates only.


### UCI Diabetes 130-US Hospitals (grouped on patient_nbr)

n = 99,492 (test 19,890, 69,667 unique patients, 13,934 in test, test-row leakage 0.0%); prevalence 0.116; 3 demographic partitions (`race`, `gender`, `age_group`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.6392 | 0.6392 | — | unchanged (same model, same split) |
| Brier | 0.0994 | 0.0994 | — | unchanged |
| JSS / PSS | 0.0008 | **0.0008** | [0.0004, 0.0018] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.2160 | 0.2160 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.2160** | [0.1286, 0.2981] | widest partition `age_group` |
| Max TFR — wide band [0.10, 0.90] | 48.4% | 48.4% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **1.4%** | [1.2%, 1.9%] | artifact (was failing, now passing) |
| Equity ρ | 0.1543 (proxy = `y_true`) | 0.7574 (proxy = `n_inpatient_proxy`) | — | 15-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 7.48 ms | 5.41 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.468 ms** | — | new in 0.2.0 |

Per-partition gaps: `age_group` 0.2160, `race` 0.1236, `gender` 0.0135.

**Against the equality null for this cohort's own partition geometry:** null mean 0.1547, p95 0.2611; observed 0.2160, one-sided p = 0.153, excess over the null mean +0.0613.

*Equity proxy*: n_inpatient_proxy (prior inpatient visits) — **model_input**. A model input feature and the dominant predictor of readmission; rho largely re-expresses that dependence.

Subgroups excluded by the n≥30 / estimability rule:

| Subgroup | n | positives | negatives | Reason | In the 0.1.0 point estimate? |
|---|---:|---:|---:|---|---|
| `age_group=[0-10)` | 31 | 0 | 31 | degenerate labels (n_pos=0, n_neg=31); AUC undefined | no |

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.1686 (rather than the 0.5 convention) the narrow-band max TFR is 11.6% and the wide-band max TFR is 37.0%. Point estimates only.

*Split note*: the published 0.1.0 figures came from a row-level split in which 42.1% of test rows belong to a patient also seen in training. Under that split the 0.1.0 pipeline gives AUROC 0.6362, pooled ΔAUC 0.2617, max TFR 49.1%, JSS 0.0004. The table above uses the group split for *both* columns so the comparison isolates the measurement change.


### NCHS NHIS 2024 (Sample Adult, CHD/MI)

n = 9,747 (test 1,950); prevalence 0.075; 5 demographic partitions (`age_group`, `sex`, `race`, `income`, `insurance`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.8363 | 0.8363 | — | unchanged (same model, same split) |
| Brier | 0.0622 | 0.0622 | — | unchanged |
| JSS / PSS | 0.0113 | **0.0046** | [0.0023, 0.0072] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.3279 | 0.3279 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.3279** | [0.2521, 0.5956] | widest partition `age_group` |
| Max TFR — wide band [0.10, 0.90] | 22.5% | 22.5% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **5.2%** | [4.2%, 6.2%] | artifact (was failing, now passing) |
| Equity ρ | 0.3066 (proxy = `y_true`) | 0.5047 (proxy = `genhlth_proxy`) | — | 5-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 1.55 ms | 1.65 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.439 ms** | — | new in 0.2.0 |

Per-partition gaps: `age_group` 0.3279, `race` 0.1090, `income` 0.0968, `insurance` 0.0439, `sex` 0.0037.

**Against the equality null for this cohort's own partition geometry:** null mean 0.2107, p95 0.3569; observed 0.3279, one-sided p = 0.087, excess over the null mean +0.1172.

*Equity proxy*: genhlth_proxy (self-rated general health) — **model_input**. A model input feature.

Subgroups excluded by the n≥30 / estimability rule:

| Subgroup | n | positives | negatives | Reason | In the 0.1.0 point estimate? |
|---|---:|---:|---:|---|---|
| `race=NH-AIAN` | 23 | 0 | 23 | n=23 < min_subgroup_n=30 | no |
| `race=NH-AIAN+other` | 17 | 1 | 16 | n=17 < min_subgroup_n=30 | no |
| `race=NH-Other/Multi` | 18 | 0 | 18 | n=18 < min_subgroup_n=30 | no |

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.2799 (rather than the 0.5 convention) the narrow-band max TFR is 7.3% and the wide-band max TFR is 16.4%. Point estimates only.


### NCHS NHIS 2023 (Sample Adult, diabetes)

n = 27,114 (test 5,423); prevalence 0.112; 4 demographic partitions (`age_group`, `sex`, `race`, `insured`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.8387 | 0.8387 | — | unchanged (same model, same split) |
| Brier | 0.0814 | 0.0814 | — | unchanged |
| JSS / PSS | 0.0169 | **0.0040** | [0.0027, 0.0055] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.1834 | 0.1834 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.1290** | [0.0653, 0.1738] | widest partition `race` |
| Max TFR — wide band [0.10, 0.90] | 32.5% | 32.5% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **9.3%** | [8.5%, 10.0%] | artifact (was failing, now passing) |
| Equity ρ | 0.3697 (proxy = `y_true`) | 0.7243 (proxy = `genhlth_proxy`) | — | 5-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 2.94 ms | 2.87 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.505 ms** | — | new in 0.2.0 |

Per-partition gaps: `race` 0.1290, `age_group` 0.1280, `sex` 0.0188, `insured` 0.0183.

**Against the equality null for this cohort's own partition geometry:** null mean 0.1867, p95 0.3517; observed 0.1290, one-sided p = 0.741, excess over the null mean -0.0577.

*Equity proxy*: genhlth_proxy (self-rated general health) — **model_input**. A model input feature.

No subgroup was excluded by the n≥30 rule on this cohort.

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.3233 (rather than the 0.5 convention) the narrow-band max TFR is 11.2% and the wide-band max TFR is 24.6%. Point estimates only.


### NCHS NHANES 2021-2023 (diabetes, with lab HbA1c)

n = 4,096 (test 820); prevalence 0.130; 4 demographic partitions (`age_group`, `sex`, `race`, `insured`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.9639 | 0.9639 | — | unchanged (same model, same split) |
| Brier | 0.0398 | 0.0398 | — | unchanged |
| JSS / PSS | 0.0265 | **0.0104** | [0.0061, 0.0159] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.0748 | 0.0748 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.0748** | [0.0377, 0.1431] | widest partition `age_group` |
| Max TFR — wide band [0.10, 0.90] | 9.8% | 9.8% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **2.9%** | [2.0%, 4.0%] | passed before and after |
| Equity ρ | 0.5413 (proxy = `y_true`) | 0.8257 (proxy = `hba1c_proxy`) | — | 62-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 0.88 ms | 0.82 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.442 ms** | — | new in 0.2.0 |

Per-partition gaps: `age_group` 0.0748, `race` 0.0465, `insured` 0.0328, `sex` 0.0203.

**Against the equality null for this cohort's own partition geometry:** null mean 0.0685, p95 0.1177; observed 0.0748, one-sided p = 0.312, excess over the null mean +0.0063.

*Equity proxy*: hba1c_proxy (HbA1c) — **outcome_defining**. HbA1c >= 6.5% is the diagnostic criterion for the outcome (diabetes) and is also a model input. The proxy is not outcome-INDEPENDENT in the sense F8 requires, even though the library's structural guard does not reject it. Equity should be treated as not evaluable on this cohort.

No subgroup was excluded by the n≥30 rule on this cohort.

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.4506 (rather than the 0.5 convention) the narrow-band max TFR is 2.9% and the wide-band max TFR is 9.0%. Point estimates only.


### CDC BRFSS 2024 (CHD/MI)

n = 44,888 (test 8,978); prevalence 0.211; 5 demographic partitions (`age_group`, `sex`, `race`, `income`, `health_plan`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.7674 | 0.7674 | — | unchanged (same model, same split) |
| Brier | 0.1400 | 0.1400 | — | unchanged |
| JSS / PSS | 0.0359 | **0.0099** | [0.0084, 0.0117] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.2334 | 0.2334 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.2237** | [0.1425, 0.2787] | widest partition `race` |
| Max TFR — wide band [0.10, 0.90] | 64.2% | 64.2% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **18.3%** | [17.6%, 19.1%] | persists |
| Equity ρ | 0.3778 (proxy = `y_true`) | 0.4094 (proxy = `physhlth_proxy`) | — | 30-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 3.88 ms | 3.77 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.457 ms** | — | new in 0.2.0 |

Per-partition gaps: `race` 0.2237, `age_group` 0.0684, `income` 0.0462, `sex` 0.0136, `health_plan` 0.0109.

**Against the equality null for this cohort's own partition geometry:** null mean 0.1470, p95 0.2337; observed 0.2237, one-sided p = 0.066, excess over the null mean +0.0767.

*Equity proxy*: physhlth_proxy (days of poor physical health) — **model_input**. A model input feature.

Subgroups excluded by the n≥30 / estimability rule:

| Subgroup | n | positives | negatives | Reason | In the 0.1.0 point estimate? |
|---|---:|---:|---:|---|---|
| `age_group=18-24` | 44 | 0 | 44 | degenerate labels (n_pos=0, n_neg=44); AUC undefined | no |

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.3294 (rather than the 0.5 convention) the narrow-band max TFR is 20.3% and the wide-band max TFR is 49.6%. Point estimates only.


### Cross-domain: UCI Adult Income

n = 45,222 (test 9,045); prevalence 0.248; 3 demographic partitions (`sex`, `race`, `age_group`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.8902 | 0.8902 | — | unchanged (same model, same split) |
| Brier | 0.1119 | 0.1119 | — | unchanged |
| JSS / PSS | 0.0119 | **0.0185** | [0.0164, 0.0210] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.1081 | 0.1081 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.0808** | [0.0548, 0.1120] | widest partition `race` |
| Max TFR — wide band [0.10, 0.90] | 33.9% | 33.9% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **13.1%** | [12.4%, 13.8%] | persists |
| Equity ρ | 0.5836 (proxy = `y_true`) | 0.5196 (proxy = `education_proxy`) | — | 16-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 1.64 ms | 1.90 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.158 ms** | — | new in 0.2.0 |

Per-partition gaps: `race` 0.0808, `age_group` 0.0673, `sex` 0.0577.

**Against the equality null for this cohort's own partition geometry:** null mean 0.0717, p95 0.1360; observed 0.0808, one-sided p = 0.346, excess over the null mean +0.0091.

*Equity proxy*: education_proxy (education-num) — **model_input**. A model input feature.

No subgroup was excluded by the n≥30 rule on this cohort.

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.3905 (rather than the 0.5 convention) the narrow-band max TFR is 15.0% and the wide-band max TFR is 28.3%. Point estimates only.


### Cross-domain: Folktables ACS-Income (CA 2018)

n = 20,000 (test 4,000); prevalence 0.410; 3 demographic partitions (`sex`, `race`, `age_group`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.8656 | 0.8656 | — | unchanged (same model, same split) |
| Brier | 0.1457 | 0.1457 | — | unchanged |
| JSS / PSS | 0.0207 | **0.0186** | [0.0155, 0.0222] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.0527 | 0.0527 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.0527** | [0.0270, 0.0761] | widest partition `age_group` |
| Max TFR — wide band [0.10, 0.90] | 39.4% | 39.4% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **17.6%** | [16.4%, 18.8%] | persists |
| Equity ρ | 0.6230 (proxy = `y_true`) | 0.6373 (proxy = `schl_proxy`) | — | 22-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 1.05 ms | 1.06 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.335 ms** | — | new in 0.2.0 |

Per-partition gaps: `age_group` 0.0527, `race` 0.0097, `sex` 0.0040.

**Against the equality null for this cohort's own partition geometry:** null mean 0.0155, p95 0.0277; observed 0.0527, one-sided p = 0.000, excess over the null mean +0.0372.

*Equity proxy*: schl_proxy (educational attainment) — **model_input**. A model input feature.

No subgroup was excluded by the n≥30 rule on this cohort.

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.4776 (rather than the 0.5 convention) the narrow-band max TFR is 17.6% and the wide-band max TFR is 37.5%. Point estimates only.


### Cross-domain: Statlog German Credit

n = 1,000 (test 200); prevalence 0.700; 2 demographic partitions (`sex`, `age_group`).

| Measurement | 0.1.0 | 0.2.0 | 95% BCa CI (0.2.0) | Change |
|---|---:|---:|:--:|---|
| AUROC | 0.6545 | 0.6545 | — | unchanged (same model, same split) |
| Brier | 0.1968 | 0.1968 | — | unchanged |
| JSS / PSS | 0.0088 | **0.0025** | [0.0000, 0.0075] | passed before and after |
| ΔAUC — pooled cross-partition (old headline, now diagnostic) | 0.0587 | 0.0587 | — | retained as diagnostic only |
| ΔAUC — max per-partition (new headline) | — | **0.0291** | [0.0014, 0.0424] | widest partition `age_group` |
| Max TFR — wide band [0.10, 0.90] | 88.5% | 88.5% | — | secondary in 0.2.0 |
| Max TFR — narrow band [0.30, 0.70] (new primary) | — | **37.5%** | [31.0%, 44.1%] | persists |
| Equity ρ | 0.2453 (proxy = `y_true`) | -0.0194 (proxy = `savings_proxy`) | — | 5-level proxy, no binary ceiling |
| Deployability — batch scoring (whole cohort) | 0.20 ms | 0.26 ms | — | renamed, not a latency |
| Deployability — single-row latency | *not measured* | **0.132 ms** | — | new in 0.2.0 |

Per-partition gaps: `age_group` 0.0291, `sex` 0.0280.

**Against the equality null for this cohort's own partition geometry:** null mean 0.1046, p95 0.2070; observed 0.0291, one-sided p = 0.942, excess over the null mean -0.0754.

*Equity proxy*: savings_proxy (savings-status ordinal) — **independent**. Carried alongside the split and never given to the model. The only genuinely independent proxy in the study.

No subgroup was excluded by the n≥30 rule on this cohort.

*τ₀ sensitivity*: at the prevalence-matched threshold τ₀ = 0.6324 (rather than the 0.5 convention) the narrow-band max TFR is 29.5% and the wide-band max TFR is 65.5%. Point estimates only.

