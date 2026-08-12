"""
Refitting each cohort under several model classes and several seeds.

Why
---
``recompute/results/cohort_sd_ratios.csv`` reports rho-hat, the ratio of the
largest to the smallest per-level standard deviation of the linear predictor,
for every demographic partition of every cohort. Every one of those numbers
comes from **one** fitted model per cohort: fixed hyperparameters, seed 42, one
train/test split. But the linear predictor is a property of the *fitted model*,
not of the cohort alone -- a random forest and a logistic regression applied to
the same rows produce different linear predictors and therefore different
per-level spreads. If rho-hat moves materially under a different model class or
a different split, then the manuscript's empirical anchor is a property of one
arbitrary modelling choice and must be reported as such.

This module supplies the refits. It deliberately does **not** tune anything
toward the published numbers: the four model classes are stated once, below,
with off-the-shelf configurations, and are applied unchanged to all ten cohorts.

How the data is reconstructed
-----------------------------
``recompute.cohorts.LOADERS`` performs data preparation, the split and the fit
inside one function and returns only the split halves. The full design matrix is
recovered by concatenating the train and test halves the loader returns, in that
order, which is a deterministic function of the loader. Re-splitting that
concatenation at ``random_state=42`` does **not** reproduce the published split
(``train_test_split`` depends on row order, and the concatenation reorders the
rows); that is intended. The published fit is carried alongside as its own
specification, ``model_class="published"``, and
``tests/test_sd_ratio_robustness.py`` asserts it reproduces
``cohort_sd_ratios.csv`` exactly.

Diabetes 130 keeps its ``GroupShuffleSplit`` on ``patient_nbr`` under every seed,
so no refit reintroduces the row-level leakage that the group split exists to
remove. ``GroupShuffleSplit`` selects a subset of ``np.unique(groups)``, which is
sorted, so the selection is invariant to the row reordering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

#: Seeds used for BOTH the train/test split and the estimator's own
#: initialisation, so a seed varies the resampling and the fit together. Six
#: seeds; the task floor is five and the count is never reduced silently.
SEEDS: Tuple[int, ...] = (42, 43, 44, 45, 46, 47)

TEST_SIZE = 0.2

#: The label used for the single published fit that
#: ``recompute/results/cohort_sd_ratios.csv`` was computed from.
PUBLISHED = "published"


# ── model classes ────────────────────────────────────────────────────────────
def _xgb_published(seed: int):
    """The manuscript's own gradient-boosting configuration.

    Hyperparameters verbatim from ``recompute.cohorts._xgb``; only the seed
    moves. This is the *current* class for the six clinical cohorts, so its
    spread across seeds isolates seed sensitivity from class sensitivity.
    """
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.80,
        eval_metric="logloss", random_state=seed, verbosity=0, seed=seed,
        n_jobs=4,
    )


def _logreg_l2(seed: int):
    """Penalised (L2) logistic regression, standardised inputs.

    ``LogisticRegression`` is L2-penalised by default at ``C=1.0``. This is the
    *current* class for the three cross-domain cohorts and a new class for the
    six clinical ones. Its linear predictor is the fitted linear predictor
    exactly, with no margin/logit correspondence to argue about.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
    ])


def _random_forest(seed: int):
    """Off-the-shelf random forest.

    Chosen because its score is an average of leaf frequencies rather than a
    sum of additive terms, so its ``logit(p)`` has a materially different shape
    from a boosted or linear model's -- exactly the kind of difference the
    editor's objection is about. ``min_samples_leaf=5`` keeps ``logit(p)`` off
    the 0/1 boundary for most rows; the clipping in
    ``comparators.cohort_casemix.linear_predictor`` handles the rest and its
    effect is reported (``frac_lp_clipped``).
    """
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, max_features="sqrt",
        n_jobs=4, random_state=seed,
    )


def _hgb_deep(seed: int):
    """A differently-configured gradient booster: deeper, faster, more rounds.

    ``max_leaf_nodes=63`` against the published ``max_depth=4`` (<= 16 leaves),
    ``learning_rate=0.10`` against 0.05, 300 rounds against 200. Same family as
    the published model, a genuinely different point in its hyperparameter
    space, so it separates "different algorithm" from "different tuning".
    """
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.10, max_leaf_nodes=63,
        l2_regularization=1.0, early_stopping=False, random_state=seed,
    )


#: The panel. Every cohort gets all four, so every cohort has at least two
#: classes beyond its own published one.
MODEL_CLASSES: Dict[str, Callable[[int], Any]] = {
    "xgb_published": _xgb_published,
    "logreg_l2": _logreg_l2,
    "random_forest": _random_forest,
    "hgb_deep": _hgb_deep,
}

#: Which class each cohort's published model belongs to.
PUBLISHED_CLASS: Dict[str, str] = {
    "synthetic": "xgb_published",   # rised.train_baseline_model: same XGB config
    "uci_heart": "xgb_published",
    "diabetes130": "xgb_published",
    "nhis2024": "xgb_published",
    "nhis2023": "xgb_published",
    "nhanes2123": "xgb_published",
    "brfss2024": "xgb_published",
    "adult_income": "logreg_l2",
    "acs_income": "logreg_l2",
    "german_credit": "logreg_l2",
}


# ── the whole-cohort bundle, reconstructed once ──────────────────────────────
@dataclass
class FullCohort:
    """Every row of one cohort, with the published split recorded alongside."""

    name: str
    X: np.ndarray
    y: np.ndarray
    demo: pd.DataFrame
    subgroup_columns: List[str]
    #: ``patient_nbr`` for Diabetes 130, ``None`` for every other cohort.
    groups: Optional[np.ndarray]
    #: Row indices into ``X`` of the published train and test halves.
    published_train: np.ndarray
    published_test: np.ndarray
    #: The published fitted model object, untouched.
    published_model: Any
    load_runtime_s: float


def build_full_cohort(name: str) -> FullCohort:
    """Run the published loader once and recover the whole cohort from it."""
    from recompute.cohorts import LOADERS

    t0 = time.perf_counter()
    b = LOADERS[name]()
    X_tr = np.asarray(b["X_train"], dtype=float)
    X_te = np.asarray(b["X_test"], dtype=float)
    X = np.vstack([X_tr, X_te])
    y = np.concatenate([np.asarray(b["y_train"]).astype(int),
                        np.asarray(b["y_test"]).astype(int)])
    demo = pd.concat([b["demo_train"], b["demo_test"]],
                     ignore_index=True)
    n_tr = X_tr.shape[0]
    idx_tr = np.arange(n_tr)
    idx_te = np.arange(n_tr, X.shape[0])

    groups = None
    if b.get("groups_train") is not None and b.get("groups_test") is not None:
        groups = np.concatenate([np.asarray(b["groups_train"]),
                                 np.asarray(b["groups_test"])])

    return FullCohort(
        name=name, X=X, y=y, demo=demo,
        subgroup_columns=list(b["subgroup_columns"]),
        groups=groups,
        published_train=idx_tr, published_test=idx_te,
        published_model=b["model"],
        load_runtime_s=time.perf_counter() - t0,
    )


def split_indices(fc: FullCohort, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Train/test row indices for one seed.

    Group split on ``patient_nbr`` where a group vector exists, stratified
    row-level split otherwise -- i.e. the same split *kind* the published
    loader used for that cohort, with only the seed moved.
    """
    idx = np.arange(fc.X.shape[0])
    if fc.groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                random_state=seed)
        tr, te = next(gss.split(idx, fc.y, groups=fc.groups))
        return idx[tr], idx[te]
    return train_test_split(idx, test_size=TEST_SIZE, random_state=seed,
                            stratify=fc.y)


@dataclass
class Fit:
    """One fitted specification, evaluated on its own held-out rows."""

    cohort: str
    model_class: str
    seed: Optional[int]
    y_test: np.ndarray
    scores: np.ndarray
    demo_test: pd.DataFrame
    n_train: int
    n_test: int
    fit_runtime_s: float
    train_prevalence: float
    test_prevalence: float


def fit_spec(fc: FullCohort, model_class: str, seed: int) -> Fit:
    """Split at ``seed``, fit ``model_class`` at ``seed``, score the test half."""
    tr, te = split_indices(fc, seed)
    est = MODEL_CLASSES[model_class](seed)
    t0 = time.perf_counter()
    est.fit(fc.X[tr], fc.y[tr])
    s = np.asarray(est.predict_proba(fc.X[te]), dtype=float)[:, 1]
    dt = time.perf_counter() - t0
    return Fit(
        cohort=fc.name, model_class=model_class, seed=seed,
        y_test=fc.y[te], scores=s,
        demo_test=fc.demo.iloc[te].reset_index(drop=True),
        n_train=int(tr.size), n_test=int(te.size), fit_runtime_s=dt,
        train_prevalence=float(fc.y[tr].mean()),
        test_prevalence=float(fc.y[te].mean()),
    )


def published_fit(fc: FullCohort) -> Fit:
    """The published model on the published test half -- no refitting at all."""
    te = fc.published_test
    t0 = time.perf_counter()
    s = np.asarray(fc.published_model.predict_proba(fc.X[te]),
                   dtype=float)[:, 1]
    return Fit(
        cohort=fc.name, model_class=PUBLISHED, seed=None,
        y_test=fc.y[te], scores=s,
        demo_test=fc.demo.iloc[te].reset_index(drop=True),
        n_train=int(fc.published_train.size), n_test=int(te.size),
        fit_runtime_s=time.perf_counter() - t0,
        train_prevalence=float(fc.y[fc.published_train].mean()),
        test_prevalence=float(fc.y[te].mean()),
    )


def iter_specs(seeds: Sequence[int] = SEEDS,
               classes: Sequence[str] = tuple(MODEL_CLASSES)
               ) -> List[Tuple[str, Optional[int]]]:
    """The specification grid, published fit first."""
    out: List[Tuple[str, Optional[int]]] = [(PUBLISHED, None)]
    for c in classes:
        for s in seeds:
            out.append((c, int(s)))
    return out
