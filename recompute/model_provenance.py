"""
What model each cohort actually fits, recorded from the fitted object.

    python -m recompute.model_provenance

Writes ``recompute/results/model_provenance.csv`` and
``recompute/results/model_provenance.json`` so the manuscript can cite the
hyperparameters instead of describing them from memory.

Why introspect rather than transcribe
-------------------------------------
The manuscript states that every cohort model is a gradient-boosted tree. That
is checkable, and it is checked here by loading each cohort through the very
loader the published numbers came from and interrogating the **fitted estimator
object** -- its class, its module, and its full parameter dictionary. Reading it
off the source would only establish what the source says.

Two things this turns up, both of which the manuscript has to state:

1. **Three of the ten cohorts are not gradient-boosted trees.** Adult Income,
   ACS-Income and German Credit are fitted with
   ``Pipeline(StandardScaler, LogisticRegression(max_iter=1000))``
   (``recompute/cohorts.py`` ``_logreg``). Only the six clinical cohorts and the
   Synthea baseline use XGBoost. "Every cohort model is a gradient-boosted tree"
   is false as written; "every *clinical* cohort model is a gradient-boosted
   tree" is true.

2. **The Synthea baseline goes through a different builder.**
   ``load_synthetic`` calls ``rised.datasets.train_baseline_model`` rather than
   ``recompute.cohorts._xgb``. Both produce an ``XGBClassifier`` at
   ``n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8,
   colsample_bytree=0.8, random_state=42``, so the model is the same family with
   the same hyperparameters -- but ``_xgb`` additionally pins
   ``eval_metric="logloss"`` and ``seed=42``, and ``train_baseline_model``
   carries a silent fallback to ``HistGradientBoostingClassifier`` when xgboost
   is not importable. In an environment without xgboost the Synthea cohort would
   quietly become a different model class while the other six stayed unfitted
   (they import xgboost directly and would raise). ``xgboost_importable`` in the
   output records which branch was taken on the run that produced these numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS = HERE / "results"
OUT_CSV = RESULTS / "model_provenance.csv"
OUT_JSON = RESULTS / "model_provenance.json"

#: Which builder each loader calls, for cross-checking against the fitted object.
BUILDER = {
    "synthetic": "rised.datasets.train_baseline_model",
    "uci_heart": "recompute.cohorts._xgb",
    "diabetes130": "recompute.cohorts._xgb",
    "nhis2024": "recompute.cohorts._xgb",
    "nhis2023": "recompute.cohorts._xgb",
    "nhanes2123": "recompute.cohorts._xgb",
    "brfss2024": "recompute.cohorts._xgb",
    "adult_income": "recompute.cohorts._logreg",
    "acs_income": "recompute.cohorts._logreg",
    "german_credit": "recompute.cohorts._logreg",
}

#: Estimator classes that are gradient-boosted decision-tree ensembles.
GBT_CLASSES = {"XGBClassifier", "LGBMClassifier", "CatBoostClassifier",
               "HistGradientBoostingClassifier", "GradientBoostingClassifier"}

#: The hyperparameters the manuscript needs to quote for the tree models.
KEY_XGB = ("n_estimators", "max_depth", "learning_rate", "subsample",
           "colsample_bytree", "eval_metric", "random_state", "seed",
           "objective", "reg_alpha", "reg_lambda", "min_child_weight", "gamma")
KEY_LR = ("max_iter", "C", "penalty", "solver", "random_state",
          "class_weight", "fit_intercept", "tol")


def _terminal(model: Any) -> Any:
    """The estimator that actually does the classifying, unwrapping Pipelines."""
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def probe(name: str) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    from recompute.cohorts import LOADERS

    b = LOADERS[name]()
    model = b["model"]
    term = _terminal(model)
    cls = type(term).__name__
    params = {k: _jsonable(v) for k, v in term.get_params().items()}
    keys = KEY_XGB if cls in GBT_CLASSES else KEY_LR

    X_te = np.asarray(b["X_test"], dtype=float)
    y_te = np.asarray(b["y_test"]).astype(int)
    s = model.predict_proba(X_te)[:, 1]

    return {
        "cohort": name,
        "builder": BUILDER.get(name, "<unknown>"),
        "wrapper_class": type(model).__name__,
        "estimator_class": cls,
        "estimator_module": type(term).__module__,
        "is_gradient_boosted_tree": cls in GBT_CLASSES,
        "is_linear_model": cls in ("LogisticRegression",),
        "n_features": int(np.asarray(b["X_train"]).shape[1]),
        "n_train": int(len(b["y_train"])),
        "n_test": int(len(y_te)),
        "prevalence_test": float(y_te.mean()),
        "test_auroc": float(roc_auc_score(y_te, s)),
        "test_size": 0.20,
        "split_random_state": 42,
        "split_stratified": True,
        **{f"hp_{k}": params.get(k) for k in keys},
        "all_hyperparameters_json": json.dumps(params, sort_keys=True),
    }


def main() -> int:
    try:
        import xgboost  # noqa: F401
        xgb_ok, xgb_ver = True, xgboost.__version__
    except ImportError:
        xgb_ok, xgb_ver = False, ""

    rows: List[Dict[str, Any]] = []
    for name in BUILDER:
        try:
            r = probe(name)
        except Exception as exc:                                # noqa: BLE001
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        r["xgboost_importable"] = xgb_ok
        r["xgboost_version"] = xgb_ver
        rows.append(r)
        print(f"[ok ] {name:<15} {r['estimator_class']:<22} "
              f"GBT={r['is_gradient_boosted_tree']}  "
              f"AUROC={r['test_auroc']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    n_gbt = int(df["is_gradient_boosted_tree"].sum())
    clinical = ("uci_heart", "diabetes130", "nhis2024", "nhis2023",
                "nhanes2123", "brfss2024")
    cl = df[df["cohort"].isin(clinical)]
    verdict = {
        "claim": "every cohort model is a gradient-boosted tree",
        "holds_for_all_cohorts": bool(n_gbt == len(df)),
        "n_gradient_boosted_tree": n_gbt,
        "n_cohorts": int(len(df)),
        "non_tree_cohorts": sorted(
            df.loc[~df["is_gradient_boosted_tree"], "cohort"]),
        "holds_for_clinical_cohorts": bool(
            len(cl) and cl["is_gradient_boosted_tree"].all()),
        "synthea_uses_a_different_builder": bool(
            df.loc[df["cohort"] == "synthetic", "builder"].iloc[0]
            != "recompute.cohorts._xgb") if "synthetic" in set(df["cohort"])
        else None,
        "xgboost_importable_on_this_run": xgb_ok,
        "xgboost_version": xgb_ver,
    }
    OUT_JSON.write_text(json.dumps(
        {"verdict": verdict, "models": rows}, indent=2), encoding="utf-8")

    print(f"\nwrote {OUT_CSV}\nwrote {OUT_JSON}\n")
    print(f"claim 'every cohort model is a gradient-boosted tree': "
          f"{'HOLDS' if verdict['holds_for_all_cohorts'] else 'FALSE'} "
          f"({n_gbt}/{len(df)} cohorts)")
    if verdict["non_tree_cohorts"]:
        print(f"  not trees: {', '.join(verdict['non_tree_cohorts'])}")
    print(f"restricted to the six clinical cohorts: "
          f"{'HOLDS' if verdict['holds_for_clinical_cohorts'] else 'FALSE'}")
    with pd.option_context("display.width", 200):
        print()
        print(df[["cohort", "builder", "estimator_class",
                  "is_gradient_boosted_tree", "n_features", "n_train",
                  "test_auroc"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
