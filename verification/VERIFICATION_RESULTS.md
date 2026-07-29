# RISED — Numerical Verification of Six Mathematical Propositions

Independent verification of six propositions about the RISED package
(`Project_healthAI_decision_sup/rised/` and `examples/`).

**No existing file was modified.** Everything here is new: six scripts
`verify_p1.py` … `verify_p6.py` and their machine-readable output in `results/`.
All scripts import the *installed* `rised` modules and exercise the real
functions; where a fast reimplementation was needed for a Monte-Carlo study, it
is validated against the real function inside the same script.

- Environment: Python 3.11.7, numpy 1.26.4, scipy 1.15.3, scikit-learn 1.8.0,
  pandas 3.0.2, xgboost 3.2.0, shap 0.44.1
- `random_state = 42` throughout.
- All six ran fully offline (NHIS/NHANES/BRFSS caches present; Diabetes130US
  served from the local `scikit_learn_data` OpenML cache). Nothing was skipped
  and nothing is estimated or extrapolated.

| # | Proposition | Verdict |
|---|---|---|
| P1 | Equity reduces to AUROC | **VERIFIED** (exact when scores are untied) |
| P2 | Pooled-range selection bias | **VERIFIED** |
| P3 | Estimand mismatch invalidates the BCa interval | **VERIFIED** |
| P4 | TFR is an outcome-free CDF functional and is gameable | **VERIFIED** |
| P5 | D2 is degenerate | **VERIFIED** |
| P6 | Diabetes 130 patient leakage | **PARTIAL** — leakage real and large; headline metrics essentially unaffected |

---

## P1 — Equity reduces to AUROC — **VERIFIED**

**Claim.** With a binary need proxy, `ρ = √(12p(1−p)) · (n/√(n²−1)) · (AUC − 0.5)`,
and the perfect-ranker ceiling is `ρ_max ≈ √(3p(1−p))`.

**Computed.** Both sides evaluated directly. `ρ` uses `scipy.stats.spearmanr`,
i.e. exactly the call in `rised/metrics.py::rank_correlation`.

| Cohort | n | p | AUROC | ρ empirical | ρ predicted | abs dev |
|---|---|---|---|---|---|---|
| NHIS 2023 diabetes | 5,441 | 0.1117 | 0.8432 | 0.374600 | 0.374600 | 2.3e-08 |
| NHIS 2023 hypertension | 5,443 | 0.3785 | 0.8368 | 0.565900 | 0.565900 | 5.5e-08 |
| Diabetes130US 30-day readmit | 19,899 | 0.1123 | 0.6362 | 0.148996 | 0.148996 | 4.3e-12 |

Simulated sweep: 304 cells (n ∈ {200, 1e3, 5e3, 2e4} × 19 prevalences from 0.01
to 0.99 × 4 signal levels), continuous scores.
**Max absolute deviation = 4.44e-16; mean = 7.0e-17** — machine precision.

**Max absolute deviation across real + tie-free simulated: 5.5e-08.**

**Ceiling.** Perfect ranker across prevalences: max deviation from the exact
finite-*n* form 3.3e-16; from the asymptotic `√(3p(1−p))` 1.1e-05 overall and
4.3e-09 for n ≥ 10⁴.

**Prevalence region where a 0.70 threshold is unattainable.**
`√(3p(1−p)) ≥ 0.70` ⟺ `3p² − 3p + 0.49 ≤ 0`, giving attainability only on
**p ∈ [0.2056, 0.7944]**. A 0.70 Equity criterion is therefore **mathematically
unreachable for p < 0.2056 or p > 0.7944**, at any model quality. Numeric sweep
at n = 10⁵ on a 0.01 grid confirms [0.21, 0.79]. The global maximum is
ρ = 0.8660 at p = 0.5. At the prevalence of the package's own Diabetes130
cohort (p = 0.112) the ceiling is **ρ_max = 0.546**.

**Condition found (stated honestly).** The identity is exact **only when the
score vector has no ties**. Ties shrink `Var(midranks)` below `(n²−1)/12`, which
the stated formula assumes. Discretising scores onto a k-level grid:

| score grid | 2 | 5 | 20 | 100 | 1000 | continuous |
|---|---|---|---|---|---|---|
| naive dev | 6.3e-02 | 1.1e-02 | 7.4e-04 | 3.1e-05 | 4.4e-07 | 1.1e-16 |
| tie-corrected dev | 5.6e-17 | 0 | 1.1e-16 | 5.6e-17 | 2.2e-16 | 5.6e-17 |

The real cohorts have high *tie fractions* (0.40, 0.50) yet deviations of only
~1e-08, because their ties occur in very small clusters. The tie-corrected form
(replacing `(n²−1)/12` with the realised midrank variance) is exact in every
case. **The proposition holds as stated for any practical continuous-score
model; it is not exact for coarsely discretised scores.**

Files: `results/p1_identity_real_cohorts.csv`, `p1_identity_simulation.csv`,
`p1_tie_sensitivity.csv`, `p1_ceiling.csv`, `p1_ceiling_reference_table.csv`,
`p1_summary.json`.

---

## P2 — Pooled-range selection bias — **VERIFIED**

**Claim.** `max−min` AUC over many subgroups has positive expectation under the
equality null, growing with group count and shrinking with group size.

**Computed.** Exact equality null: scores/labels drawn from one common binormal
model (true AUC 0.70, prevalence 0.20), then group membership assigned
*independently* of (score, label) — so every subgroup has identical true AUC and
the true parity gap is exactly 0. 2,000 replications per cell.
(`fast_auc` agrees with `sklearn.roc_auc_score` to 1.1e-16.)

Mean observed `Δ_AUC` under the null, disjoint groups:

| #groups \ size | 30 | 50 | 100 | 200 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|---|---|
| 2 | 0.145 | 0.109 | 0.075 | 0.053 | 0.032 | 0.024 | 0.016 |
| 5 | 0.300 | 0.225 | 0.153 | 0.107 | 0.067 | 0.048 | 0.034 |
| 10 | 0.397 | 0.294 | 0.201 | 0.141 | 0.089 | 0.063 | 0.045 |
| 20 | 0.486 | 0.358 | 0.248 | 0.172 | 0.108 | 0.076 | 0.054 |
| 30 | 0.529 | 0.396 | 0.270 | 0.189 | 0.119 | 0.084 | 0.059 |

Selected null distributions (mean / 95th percentile):

| G, m | mean | p95 | P(Δ>0.05) | P(Δ>0.10) |
|---|---|---|---|---|
| 10, 50 | 0.2938 | 0.4337 | 1.000 | 1.000 |
| 10, 500 | 0.0889 | 0.1304 | 0.967 | 0.284 |
| 20, 500 | 0.1081 | 0.1455 | 1.000 | 0.633 |
| 10, 2000 | 0.0448 | 0.0657 | 0.308 | 0.000 |

- **Positive in every one of the 56 cells** (range of cell means 0.016 → 0.529).
- **Strictly increasing in group count** for every group size (8/8 checks pass).
- **Strictly decreasing in group size** for every group count (7/7 pass).
- Log–log slope vs group size = **−0.517** (mean over G), matching the
  theoretical `m^(−1/2)` decay.

The overlapping design (C columns × k levels, each patient in C groups — what
`evaluate_inclusivity` actually pools over) reproduces the same behaviour: e.g.
32 pooled groups at m = 50 gives mean null range 0.380.

Files: `p2_null_range_disjoint.csv`, `p2_null_range_overlapping.csv`,
`p2_grid_disjoint_{mean_range,p95_range}.csv`, `p2_scaling_fit.csv`,
`p2_headline_cells.csv`, `p2_summary.json`.

---

## P3 — Estimand mismatch invalidates the BCa interval — **VERIFIED**

**Claim.** `rised/inclusivity.py` includes sub-30 subgroups in the point estimate
(~line 66) but drops them inside bootstrap replicates (~lines 90–107).

**Source confirmed.** Line 66 `if n_grp < 30:` is followed by
`small_groups.append(label)` — **no `continue`**, so the small group's AUC is
still recorded at line 72. Lines 98 and 120 both read `if mask_b.sum() < 30:
continue`. The same replicate function is used for the jackknife (line 136).
Two estimands:

- point estimate → max−min over all groups with ≥2 pos and ≥2 neg
- interval → max−min over groups with **n ≥ 30 only**

**Constructed cohort** (n = 1,000): five groups of 196 with identical true
AUC 0.75, plus one group of **n = 20** with true AUC 0.15. The fast
reimplementation used for the coverage study reproduces the real function
exactly (point estimate dev 0.0, CI endpoints dev 1.1e-16).

| quantity | value |
|---|---|
| point estimate `auc_parity_gap` | **0.7407** (includes `grp=SMALL`, AUC 0.0625) |
| BCa 95% CI | **(0.7724, 0.8236)** |
| bootstrap distribution mean / median | 0.1475 / 0.1353 |
| divergence (point − bootstrap mean) | **+0.5932** |
| point estimate's percentile in the bootstrap distribution | 98.7% |
| **CI contains its own point estimate?** | **No** |
| P(small group reaches n≥30 in a replicate) | 0.0224 |

The small group is flagged (`small_group_flags = ['grp=SMALL']`) yet still drives
the point estimate; it is dropped from 97.8% of replicates. BCa's bias
correction then pushes the interval *upward* (z₀ from 98.7% below), landing it
entirely above the bootstrap mass and still missing the point estimate.

**Empirical coverage** (200 independent cohorts, B = 400, nominal 95%):

| target | coverage |
|---|---|
| θ_true = 0.60 (the **point estimate's** estimand) | **14.5%** |
| θ_true = 0.00 (the **bootstrap's** estimand) | **0.0%** |
| the interval's **own point estimate** | **6.5%** |

A correctly specified interval would cover its own point estimate ~100% of the
time. Mean point estimate 0.6465 vs mean CI (0.6886, 0.7627).

Files: `p3_subgroup_aucs.csv`, `p3_bootstrap_distribution.csv`,
`p3_coverage.csv`, `p3_summary.json`.

---

## P4 — TFR is an outcome-free CDF functional and is gameable — **VERIFIED**

**(a) Outcome-free.** `y_true` appears **0 times** in the body of
`evaluate_sensitivity` (post-docstring); the docstring itself says it is "not
used in computation." Across **50 random permutations of `y_true`**, the maximum
change in *any* Sensitivity output (all 17 TFRs, rank-stability score, boundary
width) is **exactly 0.0** — while AUROC moves to 0.468–0.516, confirming the
labels were genuinely scrambled. TFR matches `|F_n(τ) − F_n(τ₀)|` to **1.1e-16**.

**(b) Constant model games it.** Every patient scored 0.05:

| | max TFR | rank_stability_score | AUROC |
|---|---|---|---|
| constant 0.05 | **0.0000** | **1.0000** (perfect) | **0.5000** (useless) |

Identical for constants 0.01, 0.95, 0.99 — any constant scores perfectly.

**(c) Concentrated model fails it despite discrimination.** Rank-preserving
squeeze of an informative score into a narrow band around 0.30:

| spread | score range | max TFR | rank_stability | AUROC |
|---|---|---|---|---|
| ±0.001 | [0.2990, 0.3010] | **1.0000** | 0.7353 | **0.8005** |
| ±0.050 | [0.2500, 0.3500] | **1.0000** | 0.7353 | **0.8005** |

Side by side on identical labels: the useless constant model **passes
perfectly** (TFR 0.0) and the AUROC-0.80 model **fails maximally** (TFR 1.0).
The metric is anti-correlated with usefulness here.

**Sub-claim S2.** `W_δ(τ₀) = TFR(τ₀−δ, τ₀) + TFR(τ₀+δ, τ₀)`, verified on 200
random continuous-score trials (n, prevalence, δ all randomised):
**max residual 1.1e-16 — exact.**

The endpoint atom behaves exactly as predicted, with one worthwhile floating-point
caveat found during verification: with δ = 0.05, `τ₀+δ` evaluates to a double
marginally greater than 0.5 + 0.05, so `|s−τ₀| ≤ δ` is *false* for `s = τ₀+δ` and
the atom cancels from both sides (residual 0). With binary-exact δ (0.125, 0.25)
the atom is counted by `W_δ` only and the residual equals its mass exactly:

| δ | atoms | residual | predicted | diff |
|---|---|---|---|---|
| 0.125 | 100/1000 | 0.100000 | 0.100000 | 2.8e-17 |
| 0.250 | 20/1000 | 0.020000 | 0.020000 | 1.7e-17 |

Files: `p4a_permutation_invariance.csv`, `p4b_constant_model.csv`,
`p4c_concentrated_model.csv`, `p4_s2_identity_continuous.csv`,
`p4_s2_identity_atoms.csv`, `p4_summary.json`.

---

## P5 — D2 is degenerate — **VERIFIED**

**Claim.** `rised/deployability.py` derives global and local feature sets from the
same first 50 rows, and `F_top3 ≡ 1` whenever d ≤ 3.

**Circularity confirmed by source.** Line 78 `X_bg = X_arr[:n_bg]` (n_bg =
min(50, len(X))); line 103 `shap_raw = explainer.shap_values(X_bg)`; line 114
`global_importance = sv.mean(axis=0)`; line 120 `local_top1 = np.argmax(sv,
axis=1)`. The "global" reference is the column mean of exactly the rows being
scored against it.

**Degeneracy confirmed by execution** of the real `evaluate_deployability`,
10 seeds × 2 data-generating processes (informative and pure-noise labels):

| DGP | d | F_top3 mean | min | stability mean | chance 3/d |
|---|---|---|---|---|---|
| informative | 2 | **1.000** | 1.00 | **1.000** | 1.00 |
| informative | 3 | **1.000** | 1.00 | **1.000** | 1.00 |
| informative | 4 | 0.992 | 0.94 | 0.966 | 0.75 |
| informative | 5 | 0.974 | 0.94 | 0.902 | 0.60 |
| informative | 10 | 0.860 | 0.68 | 0.764 | 0.30 |
| pure noise | 2 | **1.000** | 1.00 | **1.000** | 1.00 |
| pure noise | 3 | **1.000** | 1.00 | **1.000** | 1.00 |
| pure noise | 10 | 0.816 | 0.74 | 0.720 | 0.30 |

**All 40 runs with d ≤ 3 returned `explanation_faithfulness == 1.0` and
`top_feature_stability == 1.0` exactly**, including under labels independent of
the features. The identity is structural: for d ≤ 3 the global top-3 *is* the
full feature set, so the local top-1 belongs to it by construction (and
symmetrically the local top-3 always contains the global top-1).

**Chance level.** Analytically `min(1, 3/d)`. Under an exchangeable permutation
null (2,000 draws per d) the empirical null sits *above* 3/d — by +0.05 at d = 4
rising to +0.094 at d = 10 — precisely because the global set is estimated from
the same 50 rows, so circularity inflates the metric even with no signal:

| d | 4 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| analytic 3/d | 0.750 | 0.600 | 0.300 | 0.150 | 0.060 |
| empirical null | 0.802 | 0.674 | 0.394 | 0.233 | 0.118 |

So a reported F_top3 of 0.86 at d = 10 must be read against a null of ~0.39, not 0.

Files: `p5_deployability_by_d.csv`, `p5_summary_by_d.csv`,
`p5_chance_level.csv`, `p5_summary.json`.

---

## P6 — Diabetes 130 patient leakage — **PARTIAL**

**Claim.** The row-level split puts the same patients in train and test.

**Structural claim: VERIFIED, and the leakage is large.** Reproducing the exact
cleaning of `examples/external_validation_diabetes130.py` while retaining
`patient_nbr` (which the current loader drops):

| quantity | value |
|---|---|
| cohort rows (encounters) | 99,492 |
| **unique patients** | **69,667** |
| mean / max rows per patient | 1.428 / 40 |
| patients with >1 encounter | 16,483 (23.7%) |
| rows belonging to repeat patients | 46,308 (46.5%) |
| **patients appearing in BOTH splits** | **6,833** (38.2% of test patients) |
| **% of test rows whose patient is in training** | **42.12%** |

Under `GroupShuffleSplit` on `patient_nbr`: patients in both splits = 0.

**Headline-metric consequence: NOT supported.** Same model, same
hyperparameters, `test_size=0.2`:

| metric | row-level (leaky) | group-level | Δ |
|---|---|---|---|
| AUROC | 0.6362 | 0.6392 | +0.0029 |
| Brier | 0.0965 | 0.0994 | +0.0029 |
| Average precision | 0.1995 | 0.2002 | +0.0007 |
| Equity ρ (y_true) | 0.1490 | 0.1543 | +0.0053 |

Over **20 seeds**, mean AUROC change (group − row) = **−0.0022**, sd 0.0082,
**95% CI [−0.0058, +0.0014]** — indistinguishable from zero, and smaller than
split-to-split noise. Leaked test rows were 41.4–42.6% in every seed. The sign
of the per-seed difference flips (10 positive, 10 negative).

**A more sensitive test does find a real effect.** Comparing, *within* one leaky
split, AUROC on leaked test rows vs clean ones, and subtracting the same
repeat-vs-single-encounter contrast measured under the leak-free group split
(difference-in-differences, 10 seeds):

- raw gap (leaked − clean): −0.1498
- confounding gap (repeat − single, no leakage): −0.1743
- **difference-in-differences: +0.0245, se 0.0066, 95% CI [+0.0116, +0.0373]**

So leakage *does* give the leaked stratum a genuine ~0.025 AUROC advantage. It
does not propagate to the pooled headline number because the leaked stratum is
intrinsically much harder (AUROC ~0.51 vs ~0.67 for clean rows) and pooled AUROC
is dominated by that between-strata structure.

*Caveat, stated explicitly:* the two DiD strata are not perfectly matched —
"leaked" under the row split is a subset of repeat-encounter patients (those with
at least one other encounter in train), while "repeat" under the group split is
all patients with >1 encounter in the cohort. The adjustment is therefore
approximate.

**Conclusion.** The leakage is real, large, and should be fixed (it is a
methodological defect regardless of its measured effect). But the claim that it
inflates the reported result is **not supported on this cohort**: the published
AUROC ≈ 0.64 survives a leak-free split unchanged. The likely reason is that the
feature set is purely encounter-level with no patient identifier, and a
depth-4 boosted model on 15 features has no capacity to memorise individuals.
Do not report a corrected AUROC as if it differed materially.

Bootstrap CIs were not recomputed here: the RISED BCa path runs an O(n)
jackknife, infeasible at n ≈ 20,000 test rows. Metrics reported are those the
example itself prints.

Files: `p6_leakage_and_metrics.csv`, `p6_multiseed.csv`,
`p6_leaked_vs_clean_did.csv`, `p6_summary.json`.

---

## Reproducing

```bash
cd Project_healthAI_decision_sup/verification
python verify_p1.py   # ~1 min
python verify_p2.py   # ~7 min
python verify_p3.py   # ~3 min
python verify_p4.py   # ~1 min
python verify_p5.py   # ~1 min
python verify_p6.py   # ~12 min (needs the Diabetes130US OpenML cache or network)
```

Each script writes its own CSV/JSON into `results/` and prints the same figures
quoted above.
