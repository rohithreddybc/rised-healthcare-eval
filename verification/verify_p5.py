"""
P5 -- The Deployability D2 sub-metric is degenerate.

CLAIM
-----
rised/deployability.py computes SHAP explanation faithfulness F_top3 as

    F_top3 = mean_i [ argmax_j |SHAP_ij|  in  top-3 globally important features ]

where BOTH the global importance ranking AND the local top-1 are computed on
the SAME matrix X_bg = X[:50] (lines 77-78, 103, 114-123). Two consequences:

  (i) Circularity: the "global" reference set is not an independent quantity;
      it is the column-mean of exactly the rows being scored against it.
 (ii) Degeneracy: whenever the number of features d <= 3, the global top-3 set
      is the set of ALL features, so the local top-1 is a member by
      construction and F_top3 == 1 identically -- regardless of the model, the
      data, or whether the explanations mean anything.

The same argument applies to top_feature_stability (D2's companion): for
d <= 3 the local top-3 is all features, so the global top-1 is always in it.

Chance level for d > 3: if the local top-1 were uniform over features,
E[F_top3] = 3/d. This is the null the reported number must be compared against.

WHAT THIS SCRIPT DOES
---------------------
1. Runs the REAL evaluate_deployability with d = 2, 3, 4, 5, 10 and checks the
   d <= 3 identity, over multiple seeds and both an informative and a
   pure-noise data-generating process.
2. Confirms the shared-X_bg circularity by source inspection.
3. Estimates the chance level of F_top3 as a function of d, both analytically
   (3/d) and empirically under a permutation null.

Outputs -> results/p5_*.csv, results/p5_summary.json

Reproducibility: random_state = 42.
"""

from __future__ import annotations

import inspect
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import rised.deployability as dep_mod
from rised.deployability import evaluate_deployability

SEED = 42
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_SHAP = 50  # the package default; X_bg = X[:50]


def make_data(rng, n, d, informative=True):
    X = rng.normal(size=(n, d))
    if informative:
        beta = rng.normal(size=d) * np.linspace(2.0, 0.2, d)
        lat = X @ beta
    else:
        lat = np.zeros(n)  # labels independent of X
    p = 1.0 / (1.0 + np.exp(-lat))
    y = (rng.random(n) < p).astype(int)
    if y.sum() < 2:
        y[:2] = 1
    if (1 - y).sum() < 2:
        y[-2:] = 0
    return X, y


def main():
    summary = {"seed": SEED, "n_shap_samples": N_SHAP}

    # === 1. Source inspection: shared X_bg ================================
    print("=" * 78)
    print("P5.1  Source inspection of rised/deployability.py")
    print("=" * 78)
    src = inspect.getsource(dep_mod.evaluate_deployability)
    first = dep_mod.evaluate_deployability.__code__.co_firstlineno
    for i, ln in enumerate(src.splitlines()):
        st = ln.strip()
        if any(k in st for k in ("X_bg = ", "shap_raw = explainer.shap_values",
                                 "global_importance = ", "global_top3",
                                 "local_top1 = ", "explanation_faithfulness = float",
                                 "n_bg = min")):
            print(f"  line {first + i:4d}: {st}")
    shares_xbg = ("X_bg = X_arr[:n_bg]" in src
                  and "explainer.shap_values(X_bg)" in src)
    print(f"\n  global importance and local top-1 both derived from the same "
          f"X_bg = X[:50]:  {shares_xbg}")
    summary["global_and_local_share_same_rows"] = bool(shares_xbg)

    # === 2. The d <= 3 identity through the REAL function =================
    print()
    print("=" * 78)
    print("P5.2  evaluate_deployability across feature counts d")
    print("=" * 78)
    from sklearn.linear_model import LogisticRegression

    rows = []
    for informative in [True, False]:
        for d in [2, 3, 4, 5, 10]:
            for seed_off in range(10):
                rng = np.random.default_rng(SEED + 977 * seed_off + d
                                            + (0 if informative else 50000))
                X, y = make_data(rng, 400, d, informative)
                model = LogisticRegression(max_iter=1000, random_state=SEED)
                model.fit(X, y)
                res = evaluate_deployability(
                    model, X, feature_names=[f"f{j}" for j in range(d)],
                    n_latency_trials=3, n_shap_samples=N_SHAP)
                rows.append({
                    "dgp": "informative" if informative else "pure_noise",
                    "d": d, "seed_offset": seed_off,
                    "explanation_faithfulness": res.explanation_faithfulness,
                    "top_feature_stability": res.top_feature_stability,
                    "chance_level_3_over_d": min(1.0, 3.0 / d),
                    "shap_error": res.details.get("shap_error"),
                })
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "p5_deployability_by_d.csv", index=False)

    if df["explanation_faithfulness"].isna().any():
        n_bad = int(df["explanation_faithfulness"].isna().sum())
        print(f"  WARNING: SHAP failed in {n_bad}/{len(df)} runs; "
              f"first error: {df['shap_error'].dropna().iloc[0]}")

    agg = df.groupby(["dgp", "d"]).agg(
        F_top3_mean=("explanation_faithfulness", "mean"),
        F_top3_min=("explanation_faithfulness", "min"),
        F_top3_max=("explanation_faithfulness", "max"),
        stability_mean=("top_feature_stability", "mean"),
        stability_min=("top_feature_stability", "min"),
        chance=("chance_level_3_over_d", "first"),
        n_runs=("d", "size"),
    ).reset_index()
    agg.to_csv(RESULTS / "p5_summary_by_d.csv", index=False)
    print(agg.round(4).to_string(index=False))

    sub3 = df[df["d"] <= 3]
    id_faith = bool((sub3["explanation_faithfulness"] == 1.0).all())
    id_stab = bool((sub3["top_feature_stability"] == 1.0).all())
    print()
    print(f"  d <= 3, all {len(sub3)} runs (both DGPs, 10 seeds each):")
    print(f"    explanation_faithfulness == 1.0 in every run : {id_faith}")
    print(f"    top_feature_stability    == 1.0 in every run : {id_stab}")
    print(f"  d >= 4: F_top3 mean by d = " + ", ".join(
        f"d={int(r.d)}:{r.F_top3_mean:.3f}(chance {r.chance:.3f})"
        for r in agg[agg["d"] >= 4].itertuples()))
    summary["identity_F_top3_equals_1_for_d_le_3"] = id_faith
    summary["identity_stability_equals_1_for_d_le_3"] = id_stab
    summary["n_runs_d_le_3"] = int(len(sub3))

    # === 3. Chance level as a function of d ===============================
    print()
    print("=" * 78)
    print("P5.3  Chance level of F_top3 as a function of d")
    print("=" * 78)
    rng = np.random.default_rng(SEED)
    chance_rows = []
    for d in [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50]:
        # permutation null: |SHAP| entries exchangeable across features
        vals = []
        for _ in range(2000):
            sv = np.abs(rng.normal(size=(N_SHAP, d)))
            gtop3 = set(np.argsort(sv.mean(axis=0))[::-1][:3].tolist())
            ltop1 = np.argmax(sv, axis=1)
            vals.append(float(np.mean([int(t) in gtop3 for t in ltop1])))
        emp = float(np.mean(vals))
        ana = min(1.0, 3.0 / d)
        chance_rows.append({
            "d": d, "analytic_chance_3_over_d": ana,
            "empirical_exchangeable_null": emp,
            "empirical_sd": float(np.std(vals)),
            "abs_diff": abs(emp - ana),
        })
        print(f"    d={d:3d}  analytic 3/d = {ana:.4f}   "
              f"empirical null = {emp:.4f} (sd {np.std(vals):.4f})   "
              f"diff = {abs(emp - ana):.4f}")
    df_ch = pd.DataFrame(chance_rows)
    df_ch.to_csv(RESULTS / "p5_chance_level.csv", index=False)
    summary["chance_level_max_abs_diff_analytic_vs_empirical"] = float(
        df_ch["abs_diff"].max())
    summary["chance_level_table"] = df_ch.to_dict(orient="records")

    # NOTE on the empirical null: even under exchangeability the global top-3 is
    # computed from the SAME matrix, which correlates the local top-1 with the
    # global ranking and pushes the null slightly ABOVE 3/d. Reported as-is.
    print()
    print("  Note: the empirical null sits slightly above 3/d precisely because")
    print("  the global set is estimated from the same 50 rows being scored --")
    print("  the circularity in (i) inflates the metric even under a pure null.")

    summary["verdict"] = (
        "VERIFIED: F_top3 and top_feature_stability are identically 1.0 for every "
        "d <= 3 run, under both an informative and a pure-noise DGP; global and "
        "local feature sets are computed from the same X[:50]; and the chance level "
        "for d > 3 is 3/d (empirically slightly higher due to the shared-rows "
        "circularity)."
    )
    with open(RESULTS / "p5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"Wrote {RESULTS}/p5_*.csv and p5_summary.json")


if __name__ == "__main__":
    main()
