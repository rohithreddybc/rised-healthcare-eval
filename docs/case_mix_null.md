# The case-mix null: why equal subgroup AUROC is the wrong fairness test for case-mix heterogeneity

Scope: `recompute/comparators/simulate.py` (the case-mix geometries), `recompute/comparators/type1.py`, results in `docs/comparator_evaluation_tables.md` (tables T10, T14-T17) and `recompute/results/comparator_type1.csv`.

## The design

Six geometries. In each, a **single shared model** is applied to subgroups whose covariate distributions differ:

```
lp    = b0 + x,   x ~ N(loc_g, scale_g^2)      # ONE coefficient vector for all
y     ~ Bernoulli(expit(lp))
score = expit(lp)                              # the score IS the true probability
```

`b0` is solved by quadrature so the cohort prevalence is exactly on target. The subgroup label is drawn independently of everything and only then determines which `(loc_g, scale_g)` the covariate comes from.

This is the strongest possible statement of fairness for a risk model. The model is not merely "fair" under some criterion, it is the exact data-generating probability. It is correctly specified, perfectly calibrated in every subgroup and in every decile of predicted risk within every subgroup, and it applies one coefficient vector to everybody. Nothing in it treats any subgroup differently, and no modelling choice could improve it, because it is already the Bayes optimum.

And yet the subgroups' **true** AUROCs differ, by construction, and by an amount `true_subgroup_auc()` computes exactly by quadrature rather than by simulation:

| geometry | predictor SD by level | true subgroup AUROC | true gap |
|---|---|---|---|
| `casemix_location_3` | 1.0 / 1.0 / 1.0 (means differ) | 0.757 / 0.748 / 0.740 | 0.017 |
| `casemix_mild_3` | 0.9 / 1.0 / 1.1 | 0.726 / 0.745 / 0.762 | 0.036 |
| `casemix_moderate_3` | 0.7 / 1.0 / 1.4 | 0.684 / 0.745 / 0.807 | 0.123 |
| `casemix_strong_4` | 0.6 / 0.9 / 1.3 / 1.9 | 0.661 / 0.726 / 0.794 / 0.860 | 0.199 |
| `casemix_moderate_3_n10000` | 0.7 / 1.0 / 1.4, n = 10,000 | as moderate | 0.123 |
| `casemix_moderate_3part` | as moderate, plus 2 pure-noise partitions | as moderate | 0.123 |

The mean AUROC is 0.74-0.76 throughout, matching the 0.75 the rest of the study uses, and the gaps span the range seen in the real cohorts (`docs/sd_ratio_robustness.md`). `casemix_mild_3` sits below the 0.05 "clinically meaningful" convention that a naive fixed-threshold screen uses; `casemix_location_3` isolates the effect of differing predictor **location** from differing **spread**, and shows location alone produces only about a third as much gap.

This is not a contrived construction. It is the standard result on case-mix dependence of discrimination (Vergouwe et al. 2010; van Klaveren et al. 2016): AUROC measures how well a model rank-orders the patients it is shown, and a subgroup whose members are more homogeneous in true risk is intrinsically harder to rank-order. A low subgroup AUROC there is a property of that subgroup's case mix, not a defect of the model, and it is not remediable by any modelling choice, because the Bayes-optimal model already attains it.

## Result

Flag rate at nominal 0.05, 1000 simulations per cell, B = 999, rule m30. Every number in this table is a false alarm about fairness: the data-generating model is Bayes-optimal and treats no subgroup differently.

| geometry | true AUROC gap | n | Perm. null (incumbent) | DiCiccio (studentized) | Lum z | Lum Q | Lum boot CI | Four-fifths | Fixed 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| `casemix_location_3` | 0.017 | 2,000 | 0.226 | 0.044 | 0.113 | 0.091 | 0.000 | 0.031 | 0.642 |
| `casemix_mild_3` | 0.036 | 2,000 | 0.162 | 0.164 | 0.208 | 0.170 | 0.000 | 0.000 | 0.506 |
| `casemix_moderate_3part` | 0.123 | 2,000 | 0.852 | 0.864 | 0.918 | 0.860 | 0.000 | 0.102 | 0.993 |
| `casemix_moderate_3` | 0.123 | 2,000 | 0.920 | 0.919 | 0.946 | 0.926 | 0.013 | 0.119 | 0.991 |
| `casemix_strong_4` | 0.199 | 2,000 | 0.999 | 0.999 | 0.999 | 0.999 | 0.757 | 0.791 | 1.000 |
| `casemix_moderate_3_n10000` | 0.123 | 10,000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.972 | 0.006 | 1.000 |

Monte-Carlo SEs are at most 0.016 throughout and are in `recompute/results/comparator_type1.csv`. The `ev10` cells are identical to `m30` for five of the six geometries; `casemix_location_3` differs slightly because its levels have genuinely different event rates (see the note at the end of this document).

**At a realistic case-mix difference, every inferential procedure fires almost always.** A true AUROC gap of 0.123, three subgroups at 0.68 / 0.75 / 0.81, well inside the range these cohorts actually show, is flagged by the incumbent permutation test 92.0% of the time and by the studentized test 91.9%. At n = 10,000 both reach 100.0%. This is not tail behaviour or a small-sample artefact; it is the procedures working as designed on a model that cannot be improved.

**Even a case-mix difference below the field's own materiality threshold fires one time in six.** `casemix_mild_3` has a true gap of 0.036, below the 0.05 "clinically meaningful AUROC difference" convention that the fixed-threshold screen uses, and both permutation procedures flag it 16% of the time, three times the nominal 5%.

**Studentization helps against location shifts and not at all against spread.** `casemix_location_3` is the one cell where the incumbent and the studentized test come apart: the incumbent flags 22.6%, the studentized test 4.4%. That geometry's subgroups are shifted in location but their true AUROCs are nearly equal (gap 0.017), which is precisely the situation studentization was built for, and it delivers there. But once the case-mix difference is one of *spread* rather than location, and the true AUROCs genuinely separate, studentization buys nothing: 0.919 against 0.920. Studentization fixes the composite-null problem addressed elsewhere in this study; it does not make subgroup-AUROC parity testing usable on clinical prediction models where case mix drives the gap.

## What it means

None of the five procedures is malfunctioning. Subgroup membership genuinely is associated with the score given the outcome here (a subgroup with wider predictor spread has more dispersed scores among its cases and among its controls alike), so the null the permutation test literally tests, "every subgroup has the same true AUROC," is genuinely false. The procedures are correctly rejecting it.

The problem is that this null is not the question a fairness audit is asking. The audit asks whether the model treats subgroups inequitably. Equal subgroup AUROC is used as a proxy for that, and the case-mix geometries show the proxy fails in the most damaging direction available: it fires, at close to certainty, on a model that is provably beyond reproach. No amount of Type I calibration protects against this, because it is not a Type I error against the null it is measured against. A procedure with exactly the right size against the equal-AUROC null will still flag here, and the better calibrated and more powerful it is, the more reliably it will.

The Type I calibration study elsewhere in this repository (`docs/comparator_evaluation.md`) establishes that the incumbent and the studentized test control their error rate against the null they test. This document establishes that that null is the wrong one for clinical prediction models, the paper's own application domain.

A subgroup-AUROC parity test that is valid under case mix needs a case-mix correction: comparing each subgroup's observed discrimination against the discrimination expected *for that subgroup's case mix* under a shared model, rather than against the other subgroups' observed values. That is what the prognostic-model validation literature does (van Klaveren et al. 2016); `docs/case_mix_attribution.md` implements and evaluates one such correction, model-based concordance, against these same ten fitted cohorts.

Note also that the two rules `m30` and `ev10` give identical numbers on five of the six case-mix geometries, as they must: both admit every level there, so with the same data every method returns the same value. `casemix_location_3` is the exception, because its level 0 has true prevalence 0.0223 (about 15 expected events in 668 rows), which `ev10` drops in a small fraction of replicates; that level carries the highest true AUROC, so the inclusion rule interacts with the geometry built specifically to isolate location from spread (see `recompute/comparators/casemix_sweep_sim.py`, `n_admissible_p0`). Its ev10 numbers are 0.214 / 0.040 / 0.104 / 0.085 / 0.000 / 0.022 / 0.589 in the same column order as the table above.

## Reproducing

```
python -m recompute.comparators.type1 --sims 1000 --perm 999 --jobs 10 --force
python -m recompute.comparators.report
```

Both commands are part of the full comparator-evaluation reproduction in `docs/comparator_evaluation.md`; this document reports only the case-mix subset of that run's output (tables T10, T14-T17 in `docs/comparator_evaluation_tables.md`).
