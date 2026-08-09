# Round-2 fixes

Response to the second-round review. Every reported defect is addressed below in
the reviewer's numbering. Where a published number moved, the old and new values
are given side by side. Where a conclusion inverted, it is said so plainly.

Branch `jbi-revision`. Nothing has been pushed.

**Two things a reader should take away before anything else.**

1. **The Type I table was not reproducible and has been recomputed in full.**
   The seed depended on Python's salted string hash, so every worker process
   drew different data and no run could be repeated. See §1 for whether the
   numbers moved.
2. **The case-mix geometry is the most important result in this document.**
   A single shared, correctly specified, perfectly calibrated model applied to
   subgroups with different predictor spread — no unfairness of any kind — is
   flagged as inequitable by every procedure that claims a nominal 5% level,
   at rates of 90% and above. See §5.

## Summary of outcomes

| # | issue | outcome |
|---|---|---|
| 1 | Type I simulation not reproducible | **Fixed and fully re-run.** 95 of 112 shared cells moved; **no method's calibration verdict changed**. The old table's own internal inconsistency (8/35 vs 133/133) proves the defect was real. |
| 2 | Methods/code scheme mismatch | **Settled with a verified table.** Every comparator artefact is `joint`; the pre-joint per-cohort analysis is `independent`. Both schemes now run; they change **no verdict anywhere**. `scheme` in `null_joint_combined.csv` is the permutation scheme, **not** a Stouffer assumption — both rows combine under independence. |
| 3 | Per-cohort verdict concordance | **The claim was false and is withdrawn.** The honest ranking **inverts** the previous conclusion: the incumbent and the studentized test are the *least* stable of the seven procedures. |
| 4 | Composite null too thin | **Expanded from 2 to 11 geometries** — 1/3/5 partitions, n 500–10,000, prevalence 0.05–0.50, three transform families. Neither permutation procedure broke. |
| 5 | **NEW: case-mix geometry** | **Six geometries added. The result is severe.** At a realistic true AUROC gap of 0.123 with a provably fair model, the incumbent flags 92.0% and the studentized test 91.9%; both reach 100.0% at n = 10,000. |
| 6 | Equal footing for comparators | **Fixed.** The Lum bootstrap CI ran at 0.025 (cohort path) and α/2P (Type I path); both now run at one-sided α/P, and the level is emitted in the output. |
| 7 | Reporting gaps | **All seven addressed.** MC SE surfaced with a reference band for maxima; Holm/BH surfaced; `VAR_FLOOR` drops counted (**zero everywhere**); floored p-values rendered as inequalities; runtime re-benchmarked with hardware stated; hyperparameters recorded, and the gradient-boosted-tree claim shown **false for 3 of 10 cohorts**. |

---

## 1. BLOCKING — the Type I simulation was not reproducible

### The defect, demonstrated

`simulate.py:133` seeded with
`np.random.default_rng([seed, rep, abs(hash(geom.name)) % 2**31])`. CPython salts
`str.__hash__` per interpreter process unless `PYTHONHASHSEED` is set before
start, and the study dispatches its cells across a `ProcessPoolExecutor`. Five
fresh interpreters, same geometry, same replicate, same seed:

```
old, abs(hash(name)):   seed word 1923471066 -> 414 events, next draw 0.255272330192
                        seed word  876900740 -> 391 events, next draw 0.949977044026
                        seed word 1096977917 -> 406 events, next draw 0.449339073800
                        seed word  979143533 -> 434 events, next draw 0.027198770448
                        seed word 1709975777 -> 398 events, next draw 0.383948319032

new, crc32(name):       seed word 1121839185 -> 419 events, next draw 0.326073262477   (x5, identical)
```

The docstring's claim that `make_dataset` is "a pure function of `(geometry,
replicate index, seed)`" was false.

**The published table contains its own proof of this.** On the geometries where
the `m30` and `ev10` rules provably admit *exactly the same subgroups in every
replicate*, the two cells must return identical numbers for every method if they
were run on the same data. In the old table they agreed in **8 of 35**
(geometry, method) pairs. In the new table: **133 of 133.** The five geometries
that disagreed were `balanced_3x1000`, `balanced_5x200`, `many_10`,
`multi_partition` and `composite_shift_4` — including the two most balanced,
where no other explanation is available.

### The fix

- `geometry_seed_word(name) = zlib.crc32(name.encode()) & 0x7fffffff` — a
  specified function of the bytes, with no process or interpreter state.
- `PYTHONHASHSEED` pinned to 0 for the whole process tree; `type1.py` relaunches
  itself once if it was not set before interpreter start, and records the value
  in every output row alongside the geometry's seed word.
- `tests/test_type1_reproducibility.py` (61 tests) asserts bit-identical draws
  across separate interpreters started with `PYTHONHASHSEED` in {0, 1, 999},
  and through the `ProcessPoolExecutor` the study actually uses. These fail
  against the old code.

**End-to-end check.** Two published cells — `casemix_mild_3`/m30 and
`composite_pwl_4`/m30 — were recomputed from scratch in a fresh process pool with
a *different worker count* (2 instead of 10). The re-written checkpoint files
differ from the published ones in exactly two fields — `geometry_wall_s` and
`mean_runtime_per_dataset_s`, both wall-clock timings. Every statistical field,
for all seven methods, is **bit-identical**. Under the old seeding this was not
possible even in principle.

### Did the published table change? Yes — every cell, but no conclusion

All 46 cells were recomputed from scratch (`--force`, 1000 sims, B = 999,
seed 42, 107 min on 10 workers). Of the 112 cells the old table shared with the
new one, **17 are identical and 95 moved**; mean absolute change 0.0083, largest
0.0280.

That is exactly the size of change two independent 1000-simulation runs should
show. Treating the two tables as independent Monte-Carlo runs, the standard
error of each difference is ~0.010, and 8 of 112 cells exceed 2 SE with 3
exceeding 3 SE — against 5.6 and 0.3 expected by chance. So the movement is
consistent with re-randomisation and shows no systematic bias, which is the
expected outcome: the old seeding was *wrong*, not *biased*.

**No method's calibration verdict changed.** Worst cell over the equal-AUROC
geometries, old versus new:

| method | worst old | worst new | verdict |
|---|---|---|---|
| Permutation null (incumbent) | 0.058 | **0.058** | unchanged — calibrated |
| DiCiccio 2020 (studentized) | 0.060 | **0.069** | unchanged — calibrated |
| Lum 2022 (z-test) | 0.191 | **0.168** | unchanged — anti-conservative |
| Lum 2022, Cochran Q | 0.184 | **0.207** | unchanged — anti-conservative |
| Lum 2022, bootstrap CI | 0.003 | **0.017** | unchanged — no power |
| Four-fifths | 0.447 | **0.466** | unchanged — no nominal level |
| Fixed threshold 0.05 | 1.000 | **0.999** | unchanged — no nominal level |

The largest single movements were `skewed_5`/m30 `diciccio2020` 0.040 -> 0.068,
`balanced_5x200`/ev10 fixed threshold 0.930 -> 0.904, and
`multi_partition`/ev10 `permutation_null` 0.058 -> 0.035.

The Lum bootstrap-CI row also moved for a second, independent reason: its
nominal level was corrected (§6). Its old and new numbers are therefore not
comparable as a re-randomisation, and it is the one row in the table where the
change is partly systematic.

### The full recomputed table, expanded

The new study is 23 geometries x 2 rules = 46 cells, of which 34 are
equal-true-AUROC cells (simple + composite) and 12 are case-mix cells reported
separately (§5). Worst and median over the 34 equal-AUROC cells:

| method | simple: median | simple: worst | composite: median | composite: worst | overall worst (MC SE) |
|---|---|---|---|---|---|
| Permutation null (incumbent) | 0.054 | 0.058 | 0.027 | 0.050 | **0.058** (0.0074) |
| DiCiccio 2020 (studentized) | 0.044 | 0.069 | 0.048 | 0.059 | **0.069** (0.0080) |
| Lum 2022 (z-test) | 0.078 | 0.167 | 0.074 | 0.168 | **0.168** (0.0118) |
| Lum 2022, Cochran Q | 0.066 | 0.207 | 0.057 | 0.091 | **0.207** (0.0128) |
| Lum 2022, bootstrap CI | 0.000 | 0.017 | 0.000 | 0.008 | **0.017** (0.0041) |
| Four-fifths | 0.079 | 0.466 | 0.002 | 0.143 | **0.466** (0.0158) |
| Fixed threshold 0.05 | 0.917 | 0.999 | 0.805 | 0.960 | **0.999** (0.0010) |

Both maxima that matter must be read against the reference band in §7a: over 34
cells at 1000 simulations, an exactly-sized procedure produces a worst cell with
median 0.065 and 95th percentile **0.072**. The incumbent's 0.058 and the
studentized test's 0.069 are both inside it. Neither is evidence of
anticonservatism; the honest statement is that both control their size, and the
expanded composite null — three transform families, 1/3/5 partitions,
n from 500 to 10,000, prevalence 0.05 to 0.50 — did not break either of them.

---

## 2. Which permutation scheme produced which number

New artefact: `recompute/results/scheme_provenance.csv`, generated by
`python -m recompute.scheme_provenance`. It is derived from the stored artefacts,
and where an artefact carries enough information the scheme is **verified**
rather than read off the source.

### The definitive table

| result file | scheme used | B | rules | evidence |
|---|---|---|---|---|
| `recompute/results/<cohort>.json` (`null_reference` block) | **independent** | 2,000 | m30 | verified from the artefact |
| `recompute/results/summary.csv`, `summary.json`, `null_comparison.csv`, `findings.json` | **independent** | 2,000 | m30 | same |
| `RECOMPUTED_RESULTS.md` | **independent** | 2,000 | m30 | same |
| `recompute/results/null_joint/<cohort>.json` | **both**, stored separately | 10,000 | all five | every entry carries `permutation_scheme` |
| `null_sweep_mmin.csv`, `null_joint_combined.csv`, `null_joint_sign_tests.csv` | **both** (`scheme` column) | 10,000 | all five | same |
| `null_comparison_joint.csv` | **both** (`old_*` = independent, `new_*` = joint) | 10,000 | m30 | column prefixes |
| `RECOMPUTED_NULL_JOINT.md` | **both**, side by side | 10,000 | all five | same |
| `comparator_comparison.csv`, `method=permutation_null` | **joint** | 10,000 | all five | verified: all 44 estimable p-values match the joint block |
| `comparator_comparison.csv`, `method=diciccio2020` | **joint** | 10,000 | all five | code path (`core.py:360` default) |
| `comparator_comparison.csv`, Lum / four-fifths / fixed threshold | n/a — no permutation | — | all five | closed-form or deterministic |
| `comparator_type1.csv` | **joint** | 999 | m30, ev10 | code path |
| `comparator_runtime.csv` | **joint** | 10,000 | all five | code path |
| `COMPARATOR_EVALUATION.md` / `.tables.md` | **joint** | 10,000 / 999 | all five | derived from the three CSVs above |

The verification for the incumbent rows is worth stating in full, because it is
the load-bearing one: of the 44 estimable `permutation_null` p-values in
`comparator_comparison.csv`, **44 match the joint block** of the stored null runs
to within 1e-12 (largest absolute difference 1.1e-16, one float ULP from the CSV
round trip) and **0 match only the independent block**. Five match both, because
on ACS-Income the two schemes agree.

### What this means for the manuscript

The manuscript sentence — permutation carried out "independently for each
partition" — describes the **older analysis** (`RECOMPUTED_RESULTS.md`, B = 2,000,
m30 only), not the tables it now sits beside. Every comparator table, and every
number in `COMPARATOR_EVALUATION.md`, is **joint**. The sentence must be
corrected, and the two analyses distinguished, because both appear in the paper.

The scheme also now travels in the data: `comparator_comparison.csv` carries a
`permutation_scheme` column per row, and the cohort sweep can be run under either
scheme or both (`--schemes joint,independent`).

### `null_joint_combined.csv`: what the `scheme` column means

Plainly: **`scheme` is the demographic-column permutation scheme used to compute
the per-cohort p-values, and nothing else.**

- `independent` — a fresh within-outcome-class permutation for *each demographic
  column separately*, which destroys the association between age, sex, race,
  insurance and income.
- `joint` — *one* within-outcome-class permutation of the row indices per
  replicate, carried across every column, preserving the joint contingency table.

It is **not** a dependence assumption in the Stouffer combination. The reviewer's
reading of `aggregate_null_joint.py:72-76` is correct:
`stouffer()` computes `z = sum(z_i) / sqrt(k)` with equal weights in **both**
rows, which is Stouffer's method **under independence of the k cohort p-values**.
`fisher()` likewise. Neither row uses a dependence-corrected combination. That
independence assumption is about the ten cohorts being separate datasets — which
they are — and has nothing to do with the permutation scheme. The two rows differ
only in how their *inputs* were generated, never in how they were combined.

This is now stated in the module docstring and carried in the CSV itself: each
row has a `scheme_means` column and a `combination_assumes` column reading
`"independent p-values across cohorts"`.

### Both schemes, reported

The cohort sweep now runs under either scheme or both
(`--schemes joint,independent`). It was cheap, so it was done, and the question
of whether the discrepancy matters is settled with data rather than argued
(table T0 of the report):

| method | cells compared | verdicts that differ | max &#124;p_joint − p_independent&#124; | median &#124;Δp&#124; |
|---|---|---|---|---|
| Permutation null (incumbent) | 50 | **0** | 0.0180 | 0.0031 |
| DiCiccio 2020 (studentized) | 50 | **0** | 0.0084 | 0.0011 |

The two schemes move p-values by a few thousandths and change **no verdict
anywhere**, at any rule, on any cohort. They also produce identical rule-stability
profiles. The joint scheme remains the correct one — the independent scheme
generates demographic assignments whose joint contingency structure does not
exist in the data — but on these ten cohorts the choice is immaterial. The
manuscript should say exactly that, rather than leave a text/table discrepancy
unexplained.

---

## 3. Per-cohort verdict concordance — the claim was wrong, and the ranking inverts

The reviewer's reading is confirmed exactly. `diciccio2020` flags **3** clinical
cohorts under all five rules — and a *different* three:

| rule | flagged clinical cohorts |
|---|---|
| m20 / m30 / m50 | brfss2024, diabetes130, nhis2024 |
| m100 | diabetes130, nhis2023, nhis2024 |
| ev10 | brfss2024, diabetes130, nhis2023 |

Only `diabetes130` is flagged under every rule. Three of the five clinical
cohorts with a verdict — brfss2024, nhis2023, nhis2024 — receive a *different*
verdict depending on which admissibility rule the analyst happened to pick. A
count is invariant to any permutation of the flagged set, so it could never have
detected this, and it should not have been used as a stability measure. The claim
that "the studentized test returns the same verdict under every admissibility
rule" is **false and is withdrawn**.

### The metric

`recompute/results/rule_stability.csv` — one row per (method x scheme x cohort)
with the full verdict trajectory `conclusion_m20 ... conclusion_ev10`, the
p-value trajectory `p_m20 ... p_ev10`, `p_range`, `p_straddles_alpha`, and both
readings of concordance (`verdict_constant`, which ignores non-evaluable rules,
and `verdict_constant_strict`, which does not).
`recompute/results/rule_stability_by_method.csv` — the per-method summary with
flagged sets and Jaccard overlaps.

### The honest ranking (clinical cohorts; 5 have a verdict)

| rank | method | cohorts changing verdict | which | flag count constant? | flagged set constant? | mean Jaccard vs m30 | min Jaccard | flags @m30 |
|---|---|---|---|---|---|---|---|---|
| 1 | Lum 2022, bootstrap CI | **0** | — | yes | yes | 1.000 | 1.000 | 0 *(vacuous)* |
| 2 | Fixed threshold 0.05 | 1 | nhanes2123 | no | no | 0.950 | 0.800 | 5 |
| 3 | Four-fifths | 1 | brfss2024 | no | no | 0.917 | 0.667 | 3 |
| 4 | Lum 2022, Cochran Q | 2 | brfss2024, nhanes2123 | no | no | 0.850 | 0.600 | 5 |
| 5 | **Permutation null (incumbent)** | 2 | nhis2023, nhis2024 | no | no | **0.500** | **0.000** | 0 |
| 6 | Lum 2022 (z-test) | 3 | brfss2024, nhanes2123, nhis2024 | no | no | 0.708 | 0.333 | 3 |
| 7 | **DiCiccio 2020 (studentized)** | **3** | brfss2024, nhis2023, nhis2024 | **yes** | no | 0.750 | 0.500 | 3 |

Over all ten cohorts the ordering is the same at the top; the incumbent falls to
last (3 cohorts changing verdict, mean Jaccard 0.667, min Jaccard 0.333).

**This inverts the previous conclusion, and it should be reported that way.**
The two procedures that claim a nominal level — the incumbent and the studentized
test — are the *least* stable of the seven under a change of admissibility rule.
The studentized test is the single worst on the clinical cohorts, and its
constant flag count of 3 was concealing complete churn in *which* health system
would be told its model is inequitable.

**Two caveats, both of which cut against over-reading the ranking.**

- Rank 1 is vacuous. After the level correction of §6 the Lum bootstrap CI flags
  nothing anywhere, so it is trivially stable. `jaccard_vacuous` marks it.
- Ranks 2 and 3 are near-saturated rather than accurate. The fixed 0.05 threshold
  flags 5 of 6 clinical cohorts and the four-fifths rule 3 of 6; neither has much
  room to churn, and neither is being credited here with getting the answer
  right — only with insensitivity to the admissibility rule. Their Type I
  behaviour (0.999 and 0.466 worst-case, §1) is why they are not candidates.

The defensible statement is the narrow one: *inclusion-rule sensitivity is not a
defect peculiar to the incumbent, but nor is the incumbent or its studentized
competitor immune to it, and the studentized test is not more stable than the
incumbent by any measure computed here.*

### Every cohort that changes verdict, with its p-value trajectory

`F` = flag, `.` = no flag; rules in order m20 / m30 / m50 / m100 / ev10.

| method | cohort | verdict | p-value trajectory | p range |
|---|---|---|---|---|
| Permutation null | nhis2023 | `. . . F F` | 0.742 0.742 0.224 0.018 0.018 | 0.724 |
| Permutation null | nhis2024 | `. . . . F` | 0.087 0.087 0.087 0.063 0.024 | 0.063 |
| Permutation null | adult_income | `. . . F .` | 0.341 0.341 0.341 0.008 0.121 | 0.332 |
| DiCiccio 2020 | brfss2024 | `F F F . F` | 0.005 0.005 0.005 0.192 0.002 | 0.190 |
| DiCiccio 2020 | nhis2023 | `. . . F F` | 0.203 0.203 0.057 0.018 0.018 | 0.184 |
| DiCiccio 2020 | nhis2024 | `F F F F .` | 0.038 0.038 0.038 0.013 **0.058** | 0.045 |
| Lum 2022 (z) | brfss2024 | `F F F . F` | ~0 ~0 ~0 1.000 ~0 | 1.000 |
| Lum 2022 (z) | nhis2024 | `. . . . F` | 0.599 0.599 0.599 0.599 ~0 | 0.599 |
| Lum 2022 (z) | nhanes2123 | `F F F . .` | ~0 ~0 ~0 0.195 0.933 | 0.933 |
| Lum 2022, Q | brfss2024 | `F F F . F` | ~0 ~0 ~0 0.076 ~0 | 0.076 |
| Lum 2022, Q | nhanes2123 | `F F F . .` | 0.020 0.020 0.020 0.551 0.154 | 0.531 |
| Four-fifths | brfss2024 | `F F F . F` | (no p-value) | — |
| Fixed threshold | nhanes2123 | `F F F F .` | (no p-value) | — |

Every one of these straddles 0.05, and several sweep most of the unit interval:
the incumbent's nhis2023 goes from 0.742 to 0.018 on nothing but the
admissibility rule, and Lum's brfss2024 from ~0 to 1.000. These are not
borderline p-values jittering around a threshold; the evidence itself changes.

### A constructive finding: multiplicity control removes most of the instability

Once the cross-cohort Holm adjustment of §7b is applied, **the incumbent flags
zero clinical cohorts under every one of the five rules** — the m100 and ev10
raw flags on nhis2023 and nhis2024, which were the entire source of its
instability, do not survive correction for having tested ten cohorts. The
studentized test's clinical instability drops from three cohorts to one
(brfss2024). Reporting the multiplicity-adjusted decision as the headline, rather
than the raw per-cohort decision, is the cheapest available fix for most of the
rule sensitivity, and it is a fix that is independently justified.

---

## 4. The composite null is no longer thin

Before: two geometries, both single-partition, both n = 2,000 at prevalence 0.20,
both using `s -> s^a`. It never exercised the maximum-over-partitions coupling
the mechanism claim depends on, never varied the sample size or the event count,
and could not distinguish a property of the composite null from a property of
the power family.

Now: eleven composite geometries.

| geometry | partitions | n | prevalence | transform |
|---|---|---|---|---|
| `composite_shift_4` (existing) | 1 | 2,000 | 0.20 | power |
| `composite_shift_skewed` (existing) | 1 | 2,000 | 0.20 | power |
| `composite_3part` | **3** | 2,000 | 0.20 | power |
| `composite_5part` | **5** | 2,000 | 0.20 | power |
| `composite_n500` | 1 | **500** | 0.20 | power |
| `composite_n10000` | 1 | **10,000** | 0.20 | power |
| `composite_prev005` | 1 | 2,000 | **0.05** | power |
| `composite_prev050` | 1 | 2,000 | **0.50** | power |
| `composite_logit_4` | 1 | 2,000 | 0.20 | **logit-scaling** |
| `composite_pwl_4` | 1 | 2,000 | 0.20 | **piecewise-linear** |
| `composite_logit_5part` | **5** | 2,000 | 0.20 | **logit-scaling** |

The two new transform families are `s -> expit(a * logit(s))` and a two-segment
piecewise-linear map on (0, 1) with a knot at 0.5 — the second deliberately
non-smooth. Both are strictly increasing, so both leave every subgroup's true
AUROC exactly unchanged, which is the only property the composite null needs.

Three guards were added because a "monotone" map that is not monotone *in
float64* would silently move the true AUROC and turn a null cell into a power
calculation:

- `make_dataset` counts distinct scores before and after the transform and raises
  if any were lost. `logit_scale` saturates once `a * |logit(s)|` passes about
  36.7, where `expit` rounds to exactly 1.0; the geometries use `a <= 3.0` on
  scores with `|logit(s)| < 6`, a factor of two of headroom, and the guard checks
  it on every dataset rather than trusting the argument.
- `tests/test_type1_reproducibility.py` asserts strict monotonicity and exact
  AUROC invariance for each family across the parameter range in use, and pins
  where the saturation bound is.
- `verify_null` confirms on 200,000-row draws that the maximum studentized
  subgroup difference stays under 3 in every one of these geometries.

Only the *first* partition carries the transform. That is deliberate: if two
partitions each applied their own per-level map, the map would no longer be
constant within a level of either, and the true AUROC would move. With one
transformed partition and up to four untransformed ones, the maximum still runs
over transformed and untransformed columns together, which is exactly the
coupling the mechanism claim rests on.

---

## 5. The case-mix null — the most important result in this document

### The design

Six new geometries. In each, a **single shared model** is applied to subgroups
whose covariate distributions differ:

```
lp   = b0 + x,    x ~ N(loc_g, scale_g^2)      # ONE coefficient vector for all
y    ~ Bernoulli(expit(lp))
score = expit(lp)                              # the score IS the true probability
```

`b0` is solved by quadrature so the cohort prevalence is exactly on target. The
subgroup label is drawn independently of everything and only then determines
which `(loc_g, scale_g)` the covariate comes from.

This is the strongest possible statement of fairness for a risk model. The model
is not merely "fair" under some criterion — it is the exact data-generating
probability. It is correctly specified, it is perfectly calibrated in every
subgroup and in every decile of predicted risk within every subgroup
(asserted in `tests/test_type1_reproducibility.py`), and it applies one
coefficient vector to everybody. Nothing in it treats any subgroup differently,
and no modelling choice could improve it, because it is already the Bayes
optimum.

And yet the subgroups' **true** AUROCs differ — by construction, and by an amount
`true_subgroup_auc()` computes exactly by quadrature rather than by simulation:

| geometry | predictor SD by level | true subgroup AUROC | true gap |
|---|---|---|---|
| `casemix_location_3` | 1.0 / 1.0 / 1.0 (means differ) | 0.757 / 0.748 / 0.740 | **0.017** |
| `casemix_mild_3` | 0.9 / 1.0 / 1.1 | 0.726 / 0.745 / 0.762 | **0.036** |
| `casemix_moderate_3` | 0.7 / 1.0 / 1.4 | 0.684 / 0.745 / 0.807 | **0.123** |
| `casemix_strong_4` | 0.6 / 0.9 / 1.3 / 1.9 | 0.661 / 0.726 / 0.794 / 0.860 | **0.199** |
| `casemix_moderate_3_n10000` | 0.7 / 1.0 / 1.4, n = 10,000 | as moderate | 0.123 |
| `casemix_moderate_3part` | as moderate, plus 2 pure-noise partitions | as moderate | 0.123 |

The mean AUROC is 0.74–0.76 throughout, matching the 0.75 the rest of the study
uses, and the gaps span the range seen in the real cohorts. `casemix_mild_3` sits
*below* the 0.05 "clinically meaningful" convention; `casemix_location_3`
isolates the effect of differing predictor **location** from differing
**spread**, and shows location alone produces only a third as much.

This is not a contrived construction. It is the standard result on case-mix
dependence of discrimination (Vergouwe et al. 2010; van Klaveren et al. 2016):
AUROC measures how well a model rank-orders the patients it is shown, and a
subgroup whose members are more homogeneous in true risk is intrinsically harder
to rank-order. A low subgroup AUROC there is a property of that subgroup's case
mix, not a defect of the model, and **it is not remediable by any modelling
choice**, because the Bayes-optimal model already attains it.

### The result

Flag rate at nominal 0.05, 1000 simulations per cell, B = 999, rule m30. **Every
number in this table is a false alarm about fairness.**

| geometry | true AUROC gap | n | Perm. null (incumbent) | DiCiccio (studentized) | Lum z | Lum Q | Lum boot CI | Four-fifths | Fixed 0.05 |
|---|---|---|---|---|---|---|---|---|---|
| `casemix_location_3` | 0.017 | 2,000 | **0.226** | 0.044 | 0.113 | 0.091 | 0.000 | 0.031 | 0.642 |
| `casemix_mild_3` | 0.036 | 2,000 | **0.162** | **0.164** | 0.208 | 0.170 | 0.000 | 0.000 | 0.506 |
| `casemix_moderate_3part` | 0.123 | 2,000 | **0.852** | **0.864** | 0.918 | 0.860 | 0.000 | 0.102 | 0.993 |
| `casemix_moderate_3` | 0.123 | 2,000 | **0.920** | **0.919** | 0.946 | 0.926 | 0.013 | 0.119 | 0.991 |
| `casemix_strong_4` | 0.199 | 2,000 | **0.999** | **0.999** | 0.999 | 0.999 | 0.757 | 0.791 | 1.000 |
| `casemix_moderate_3_n10000` | 0.123 | 10,000 | **1.000** | **1.000** | 1.000 | 1.000 | 0.972 | 0.006 | 1.000 |

Monte-Carlo SEs are at most 0.016 throughout and are in the CSV. The `ev10` cells
are identical to `m30` for five of the six geometries (see the note at the end of
this section); `casemix_location_3` differs slightly because its levels have
genuinely different event rates, and its ev10 numbers are 0.214 / 0.040 / 0.104 /
0.085 / 0.000 / 0.022 / 0.589 in the same column order.

Three readings.

**At a realistic case-mix difference, every inferential procedure fires almost
always.** A true AUROC gap of 0.123 — three subgroups at 0.68 / 0.75 / 0.81, well
inside the range these cohorts actually show — is flagged by the incumbent 92.0%
of the time and by the studentized test 91.9%. At n = 10,000 both reach
**100.0%**. This is not a tail behaviour or a small-sample artefact; it is the
procedures working as designed on a model that cannot be improved.

**Even a case-mix difference below the field's own materiality threshold fires
one time in six.** `casemix_mild_3` has a true gap of 0.036, *below* the 0.05
"clinically meaningful AUROC difference" convention that the naive baseline uses
as its threshold, and both permutation procedures flag it 16% of the time —
three times the nominal 5%.

**Studentization helps against location shifts and not at all against spread.**
`casemix_location_3` is the one cell where the two permutation procedures come
apart: the incumbent flags 22.6%, the studentized test 4.4%. That geometry
differs from the composite null only slightly — the subgroups' score
distributions are shifted but their true AUROCs are nearly equal (gap 0.017) —
which is precisely the situation DiCiccio's studentization was built for, and it
delivers, holding almost exactly to nominal while the unstudentized incumbent
inflates fourfold. But the moment the case-mix difference is one of *spread*
rather than location, and the true AUROCs genuinely separate, studentization buys
nothing: 0.919 against 0.920. The paper can legitimately claim that
studentization fixes the composite-null problem. It cannot claim that fixing the
composite-null problem makes the test usable on clinical prediction models.

### What it means

The five procedures are not malfunctioning. Subgroup membership genuinely is
associated with the score given the outcome here — a subgroup with wider
predictor spread has more dispersed scores among its cases and among its
controls alike — so the null that the permutation test literally tests, "every
subgroup has the same true AUROC", is genuinely **false**. The procedures are
correctly rejecting it.

The problem is that this null is not the question the audit is asking. The audit
asks whether the model treats subgroups inequitably. Equal subgroup AUROC is
being used as a proxy for that, and the case-mix geometries show the proxy fails
in the most damaging direction available: it fires, at essentially certainty,
on a model that is provably beyond reproach. No amount of Type I calibration
protects against this, because it is not a Type I error. A procedure with
exactly the right size against the equal-AUROC null will still flag here, and
the better calibrated and more powerful it is, the more reliably it will.

The consequence for the manuscript is direct and it is not small. The Type I
study establishes that the incumbent and the studentized test control their error
rate against the null they claim to test. The case-mix study establishes that
that null is the wrong one for clinical prediction models, which is the paper's
own application domain. Both must be reported, and the second is the one a
clinical reader needs.

The constructive reading is that a subgroup-AUROC parity test needs a case-mix
correction — comparing each subgroup's observed discrimination against the
discrimination expected *for that subgroup's case mix* under a shared model,
rather than against the other subgroups' observed values. That is what the
prognostic-model validation literature does, and it is outside the scope of this
revision, but the paper cannot claim a fairness test for clinical models without
either implementing it or scoping the claim down to the null actually tested.

Note also that the two rules `m30` and `ev10` give **identical** numbers on the
case-mix geometries, as they must: both admit every level there, so with the same
data every method must return the same value. That identity is itself a check
that the seeding fix worked — it was impossible before, when the two cells were
run on different data (see §1).

---

## 6. Equal footing for the comparators

The reviewer is right, and the discrepancy was worse than reported: the two call
paths did not even agree with each other.

| path | before | one-sided level | after |
|---|---|---|---|
| `lum.run_cohort` (cohort table) | `conf` defaulted to 0.95, no multiplicity adjustment at all | 0.025 | one-sided `alpha/P` |
| `lum.decide` (Type I study) | `conf = 1 - alpha/P` | `alpha/(2P)` | one-sided `alpha/P` |
| `dc_ztest`, `cochran_q` | Holm across partitions at `alpha` | family-wise `alpha` | unchanged |

The `flag iff ci_lo > 0` rule uses only the lower bound and is therefore
one-sided; a two-sided `conf` places that bound at `(1 - conf)/2`, i.e. half the
level the name suggests. The module docstring claimed `1 - alpha/P` for the
cohort path, which the code never did.

`bootstrap_ci` now takes an explicit `one_sided_alpha`, both paths pass
`alpha/P` (Bonferroni over the `P` partitions, giving family-wise level `alpha`
to match the Holm-adjusted z-test and Q), and the function returns `level_lo` —
the one-sided level actually used. That level is written into the diagnostics
JSON and into the `nominal_level_note` column of `comparator_type1.csv`, so the
comparison is auditable from the output rather than from the source.

The four-fifths rule and the fixed 0.05 threshold are deterministic and have no
nominal level to equalise; that is stated in the same column rather than left
implicit.

---

## 7. Reporting gaps

### 7a. Monte-Carlo standard error

`type1.py:165` computed it and only the CSV carried it. Every Type I table in
`COMPARATOR_EVALUATION.tables.md` now prints the rate with its MC SE in
parentheses, and the calibration summary adds the SE at the worst cell and
`(worst - 0.05) / SE`.

The reviewer's point about maxima is right and is now quantified rather than
asserted. `report.max_cell_reference_band` simulates the distribution of the
largest of *n* independent Binomial(1000, 0.05)/1000 proportions:

| cells | median of the max | 90th pct | 95th pct | 99th pct |
|---|---|---|---|---|
| 12 | 0.061 | 0.067 | **0.069** | 0.073 |
| 34 (the expanded study) | 0.065 | 0.070 | **0.072** | 0.075 |

So a worst cell at or below 0.069 across twelve cells — or 0.072 across
thirty-four — is exactly what a procedure of the correct size produces, and
quoting it as evidence of anticonservatism is a selection effect. Independence
across cells is assumed; the one real dependence (m30 and ev10 share datasets
within a geometry) is positive and makes the true maximum *smaller*, so the band
errs toward calling a method calibrated, which is the cautious direction here.

### 7b. Cross-cohort multiplicity

`null_sweep_mmin.csv` carried `p_holm` and `p_bh` and nothing surfaced them.
`comparator_comparison.csv` now carries `p_holm_across_cohorts`,
`p_bh_across_cohorts`, `flag_raw`, `flag_holm`, `flag_bh` and
`multiplicity_applies`, adjusted within each `(method, rule, scheme)` cell over
the cohorts estimable there — which is the family the "k of ten cohorts" sentence
is drawn from. Tables T5b and T5c of the report show raw vs Holm vs BH flag
counts per method, and name the cohorts lost to Holm.

**The result is substantive.** At `ev10` the incumbent's three raw flags become
**one** after Holm (only ACS-Income, a cross-domain cohort, survives); nhis2023
and nhis2024 do not. Restricted to the clinical cohorts, the incumbent flags
**zero** under every rule after Holm — which, as noted in §3, removes the entire
source of its inclusion-rule instability. The studentized test loses two of six
raw flags at m30 (nhis2024, synthetic) and its clinical instability falls from
three cohorts to one. Reporting the multiplicity-adjusted decision rather than
the raw per-cohort decision is therefore both independently correct and the
cheapest available fix for most of the rule sensitivity the review identified.

### 7c. Pairs dropped by `VAR_FLOOR`

`diciccio.studentized` returns `nan` when `Var_a + Var_b <= 1e-12`, which removes
the pair from the max-T family and so changes the family the family-wise error
rate is controlled over. That is now counted per rule — observed pairs and,
separately, pair-evaluations lost across the null replicates — and reported in
the `detail` string, in the diagnostics JSON, and in table T5d.

**Result: no pair was dropped anywhere.** Across all 50 (cohort x rule) cells of
the studentized test, `n_pairs_dropped_var_floor` is **0** in every one, and no
pair evaluation was lost in the null replicates either. The floor is not
silently shrinking the max-T family in this study, so the family-wise error rate
is being controlled over the family the report says it is. This is a clean
negative result, but it had to be measured rather than assumed.

### 7d. p-values at the `1/(B+1)` floor

`core.p_report` and `MethodResult.p_value_report` render a floored p-value as
`<= 1.00e-04` at B = 10,000, never as a number a downstream table could round to
`0.000`. The floor is taken from the row's own `n_perm` rather than hard-coded,
and `p_floor` is emitted as its own column. The rendering matches
`null_reference.mc_pvalue`'s existing format exactly, so the two halves of the
codebase spell the same bound the same way (the bound is rounded *up* to three
significant figures, which keeps the inequality true). ACS-Income is the case
that mattered: its incumbent and studentized p-values are both at the floor
under every rule.

### 7e. Runtime

The reviewer is right that `188 s vs 193 s` cannot carry the claim. It was a
single un-repeated timing on a machine that was never described, and the 5 s gap
is well inside the run-to-run spread of a three-minute numerical loop. It also
sat in the same table as a 295 s shipped-kernel figure, which invited the reader
to difference the wrong pair — the gap between 295 s and 188 s is the vectorised
kernel, not the statistic.

`bench.py` was rewritten: each timing is repeated (`--repeats`, default 3), the
cohorts run strictly sequentially so no measurement competes for a core, and
min / median / max / SD are reported along with the pooled repeat noise and a
flag for whether the overhead is even resolvable. The machine, core counts, RAM,
OS, Python, numpy, scipy and the BLAS thread caps are written to
`recompute/results/comparator_runtime_env.json`. Where the overhead is inside the
noise, the report prints "no material runtime cost" **with no number attached**.

**First, where the quoted numbers came from.** `188 s vs 193 s` and `295 s` are
not per-cohort figures: they are the **totals over all ten cohorts** of the old
single-shot table (295.3 / 192.9 / 187.8 s for shipped kernel / same kernel /
studentized). Note the direction — the studentized test came out *faster* than
the incumbent's own statistic, which is not what a "cost of studentization"
reading would predict.

**The repeated benchmark.** Machine: Intel Core (Family 6 Model 186), 8 physical
/ 12 logical cores, 13.7 GB RAM, Windows 10.0.26200, Python 3.11.7, numpy 1.26.4,
scipy 1.15.3, BLAS thread caps unset. B = 10,000, 3 repeats, strictly sequential.
Totals over the ten cohorts, median of three repeats:

| | shipped kernel | same kernel | studentized | Lum | four-fifths | fixed 0.05 |
|---|---|---|---|---|---|---|
| total, 10 cohorts | 295.3 s | **105.8 s** | **103.7 s** | 0.14 s | 8.87 s | 0.05 s |

Per cohort the studentized test is between 0.1% and 14.3% **faster** than the
incumbent's statistic on the same kernel, in all ten cohorts, and in five of them
the gap exceeds the run-to-run standard deviation. So the difference is real and
it is in the opposite direction to "studentization costs something".

**It should still not be quoted as a number.** The consistent sign is an artefact
of how the two are organised in the comparator package, not of the statistic:
`incumbent.recompute_null` re-applies the admissibility filter once per rule
*inside* the permutation loop (five passes per replicate), while
`diciccio.run_cohort` computes the pair statistics once per replicate and masks
by rule afterwards. Both produce all five rules; one simply hoists work out of
the loop. Attributing that to studentization would be as wrong as the original
claim in the other direction.

**Reported conclusion:** studentization carries **no material runtime cost**. No
number is attached to that statement, per the reviewer's second option. Both
procedures are dominated by the shared permutation draw, the whole ten-cohort
sweep is under two minutes at B = 10,000, and the full repeated measurements —
min, median, max, SD, and every individual timing — are in
`comparator_runtime.csv` and `comparator_runtime_raw.csv` for anyone who wants
them. The 295 s shipped-kernel figure is kept in its own column and must not be
differenced against the studentized column: the gap between 295 s and 106 s is
the vectorised kernel, not the statistic.

### 7f. Model hyperparameters, and the gradient-boosted-tree claim

New artefact: `recompute/results/model_provenance.csv` and `.json`, generated by
`python -m recompute.model_provenance`. It loads every cohort through the loader
the published numbers came from and interrogates the **fitted estimator object**,
not the source.

**The claim that every cohort model is a gradient-boosted tree is false: 7 of 10.**

| cohort | builder | estimator | GBT? | n features | n train | test AUROC |
|---|---|---|---|---|---|---|
| Synthetic (Synthea) | `rised.datasets.train_baseline_model` | XGBClassifier | yes | 20 | 8,000 | 0.9610 |
| UCI Heart | `cohorts._xgb` | XGBClassifier | yes | 13 | 242 | 0.8669 |
| Diabetes 130 | `cohorts._xgb` | XGBClassifier | yes | 14 | 79,602 | 0.6392 |
| NHIS 2024 | `cohorts._xgb` | XGBClassifier | yes | 19 | 7,797 | 0.8363 |
| NHIS 2023 | `cohorts._xgb` | XGBClassifier | yes | 14 | 21,691 | 0.8387 |
| NHANES 21-23 | `cohorts._xgb` | XGBClassifier | yes | 14 | 3,276 | 0.9639 |
| BRFSS 2024 | `cohorts._xgb` | XGBClassifier | yes | 19 | 35,910 | 0.7674 |
| Adult Income | `cohorts._logreg` | **LogisticRegression** | **no** | 10 | 36,177 | 0.8902 |
| ACS-Income | `cohorts._logreg` | **LogisticRegression** | **no** | 10 | 16,000 | 0.8656 |
| German Credit | `cohorts._logreg` | **LogisticRegression** | **no** | 11 | 800 | 0.6545 |

The claim holds if and only if it is restricted to the six clinical cohorts. The
three cross-domain cohorts use `Pipeline(StandardScaler,
LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42))`.

Hyperparameters for the manuscript to cite:

- **XGBClassifier** (seven cohorts): `n_estimators=200`, `max_depth=4`,
  `learning_rate=0.05`, `subsample=0.80`, `colsample_bytree=0.80`,
  `objective='binary:logistic'`, `random_state=42`. Six of the seven
  (`cohorts._xgb`) additionally pin `eval_metric='logloss'` and `seed=42`.
- **LogisticRegression** (three cohorts): `max_iter=1000`, `C=1.0`,
  `solver='lbfgs'`, `random_state=42`, standardised inputs.
- **Split**, all ten: `train_test_split(test_size=0.20, random_state=42,
  stratify=y)`; Diabetes 130 uses a `GroupShuffleSplit` on `patient_nbr`
  instead.

And the reviewer's suspicion about Synthea is confirmed: `load_synthetic` does go
through `rised.datasets.train_baseline_model` rather than `cohorts._xgb`. Both
give an `XGBClassifier` with the same six core hyperparameters, so the model is
the same family with the same settings; the differences are that `_xgb` pins
`eval_metric` and `seed`, and that `train_baseline_model` carries a **silent
fallback to `HistGradientBoostingClassifier`** when xgboost is not importable. In
an environment without xgboost, Synthea would quietly become a different model
class while the other six raised an ImportError. `xgboost_importable` and
`xgboost_version` are recorded in the output for the run that produced these
numbers (xgboost 3.2.0, importable).

---

## Files changed

Code:

- `recompute/comparators/simulate.py` — crc32 seed word; 9 new composite
  geometries; 6 case-mix geometries; three monotone transform families; exact
  true-subgroup-AUROC quadrature; tie guard on every dataset.
- `recompute/comparators/type1.py` — `PYTHONHASHSEED` pinning; `null_family`,
  `true_auc_*`, `flag_means`, `nominal_level_note`, `geometry_seed_word` and
  `pythonhashseed` columns.
- `recompute/comparators/lum.py` — explicit one-sided bootstrap level.
- `recompute/comparators/diciccio.py` — `VAR_FLOOR` accounting.
- `recompute/comparators/core.py` — `p_report`, `MethodResult.p_value_report`,
  `p_floor`.
- `recompute/comparators/incumbent.py` — carries `n_perm` and the stored
  `p_report`.
- `recompute/comparators/run.py` — cross-cohort Holm/BH; `--schemes`;
  `permutation_scheme` column.
- `recompute/comparators/bench.py` — repeated, sequential, hardware-recorded.
- `recompute/comparators/rule_stability.py` — **new**.
- `recompute/comparators/report.py` — MC SE, multiplicity, `VAR_FLOOR`,
  stability, case-mix tables, max-cell reference band.
- `recompute/scheme_provenance.py` — **new**.
- `recompute/model_provenance.py` — **new**.
- `recompute/aggregate_null_joint.py` — documents what `scheme` means.

Tests:

- `tests/test_type1_reproducibility.py` — **new**, 61 tests.
- `tests/test_round2_fixes.py` — **new**, 15 tests.
- Full suite: **345 passing**.
- `tests/test_comparators.py` — case-mix geometries excluded from the
  equal-AUROC null assertion; the artefact-consistency test made scheme-aware
  and extended to the floor rendering and the multiplicity columns.

Results:

- `recompute/results/comparator_type1.csv` — fully recomputed.
- `recompute/results/type1_cells/` — 46 cells.
- `recompute/results/rule_stability.csv`, `rule_stability_by_method.csv` — new.
- `recompute/results/scheme_provenance.csv` — new.
- `recompute/results/model_provenance.csv`, `.json` — new.
- `recompute/results/comparator_runtime_raw.csv`, `comparator_runtime_env.json`
  — new.
- `recompute/results/comparator_comparison.csv` — re-run under both schemes.
- `COMPARATOR_EVALUATION.tables.md` — regenerated.

## Reproducing

```
PYTHONHASHSEED=0 python -m recompute.comparators.type1 --sims 1000 --perm 999 --jobs 10 --force
python -m recompute.comparators.run --stage cohorts --schemes joint,independent --jobs 8
python -m recompute.comparators.rule_stability
python -m recompute.comparators.bench --repeats 3
python -m recompute.scheme_provenance
python -m recompute.model_provenance
python -m recompute.comparators.report
```

Everything is seeded at 42. `type1.py` pins `PYTHONHASHSEED=0` itself and
relaunches once if it was not set before interpreter start.
