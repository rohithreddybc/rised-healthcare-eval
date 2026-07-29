"""
Perturbation generators for the Reliability dimension.

Two disjoint families
---------------------
**Semantics-preserving perturbations** re-encode the *same* patient: measurement
noise on a continuous quantity, a jitter of a date-derived feature, a genuine
unit conversion. A model that changes its decision under these is unstable, and
only these enter the Judge Sensitivity Score.

**Covariate-shift perturbations** change the patient, not the encoding.
Multiplying age by 1.05 does not re-encode anyone: there is no unit of age that
differs from another by 5%, so the perturbed row describes a different (and
often non-existent) person. Such perturbations are still useful — they probe
robustness to population drift — but they are *not* reliability, and RISED keeps
them in a separate set (see :data:`COVARIATE_SHIFT_TYPES` and
:func:`perturbation_semantics`).

Typed feature schema
--------------------
Adding Gaussian noise to every column produces impossible patients: a binary
``diabetes`` flag becomes 0.9993, a 5-level ordinal severity code becomes 2.47.
:class:`FeatureSchema` types each column as ``continuous``, ``ordinal``,
``binary`` or ``categorical``; continuous noise is applied to continuous columns
only. Binary and categorical columns never receive continuous noise — silently
skipped when the columns were selected implicitly, and a hard error when a
caller names them explicitly.

Perturbation spec format: ``{"type": str, ...kwargs}``
Supported types: ``gaussian_noise``, ``ordinal_jitter``, ``temporal_jitter``,
``unit_rescaling``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ── Feature type vocabulary ──────────────────────────────────────────────────
CONTINUOUS = "continuous"
ORDINAL = "ordinal"
BINARY = "binary"
CATEGORICAL = "categorical"

FEATURE_TYPES = (CONTINUOUS, ORDINAL, BINARY, CATEGORICAL)

#: Column types that may receive additive continuous (Gaussian) noise.
NOISE_ELIGIBLE_TYPES = (CONTINUOUS,)

#: Perturbations that re-encode the same patient. Only these enter JSS.
SEMANTICS_PRESERVING_TYPES = ("gaussian_noise", "ordinal_jitter", "temporal_jitter")

#: Perturbations that change the patient rather than the encoding.
COVARIATE_SHIFT_TYPES = ("unit_rescaling",)

SEMANTICS_PRESERVING = "semantics_preserving"
COVARIATE_SHIFT = "covariate_shift"


@dataclass
class FeatureSchema:
    """Per-column semantic type for a feature matrix.

    Parameters
    ----------
    types : list of str
        One of ``continuous``/``ordinal``/``binary``/``categorical`` per column.
    names : list of str, optional
        Column names, for error messages only.
    """

    types: List[str]
    names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        bad = [t for t in self.types if t not in FEATURE_TYPES]
        if bad:
            raise ValueError(
                f"Unknown feature type(s) {sorted(set(bad))!r}. "
                f"Supported: {list(FEATURE_TYPES)}."
            )
        if self.names is not None and len(self.names) != len(self.types):
            raise ValueError("names and types must have the same length.")

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def infer(
        cls,
        X,
        names: Optional[Sequence[str]] = None,
        max_discrete_levels: int = 10,
    ) -> "FeatureSchema":
        """Infer a conservative schema from the data.

        Rules, in order:

        1. at most 2 distinct values -> ``binary``;
        2. all values integral and at most ``max_discrete_levels`` distinct
           values -> ``categorical``;
        3. otherwise -> ``continuous``.

        Rule 2 returns ``categorical`` rather than ``ordinal`` deliberately:
        ordering cannot be recovered from the values alone, and treating a
        nominal code as ordinal would licence a meaningless +/-1 jitter. Declare
        ``ordinal`` explicitly when a column really is ordered.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X_arr.shape}.")
        types: List[str] = []
        for j in range(X_arr.shape[1]):
            col = X_arr[:, j]
            finite = col[np.isfinite(col)]
            uniq = np.unique(finite)
            if len(uniq) <= 2:
                types.append(BINARY)
            elif len(uniq) <= max_discrete_levels and np.all(uniq == np.round(uniq)):
                types.append(CATEGORICAL)
            else:
                types.append(CONTINUOUS)
        return cls(types=types, names=list(names) if names is not None else None)

    # ── queries ──────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.types)

    def type_of(self, index: int) -> str:
        return self.types[index]

    def indices_of(self, *kinds: str) -> List[int]:
        """Column indices whose type is one of ``kinds``."""
        return [i for i, t in enumerate(self.types) if t in kinds]

    def label(self, index: int) -> str:
        if self.names is not None:
            return f"{self.names[index]!r} (column {index})"
        return f"column {index}"

    def summary(self) -> Dict[str, int]:
        return {t: sum(1 for x in self.types if x == t) for t in FEATURE_TYPES}


def _resolve_schema(X_arr: np.ndarray, schema: Optional[FeatureSchema]) -> FeatureSchema:
    if schema is None:
        return FeatureSchema.infer(X_arr)
    if len(schema) != X_arr.shape[1]:
        raise ValueError(
            f"schema has {len(schema)} columns but X has {X_arr.shape[1]}."
        )
    return schema


# ── Perturbations ────────────────────────────────────────────────────────────
def gaussian_noise(
    X,
    feature_indices: Optional[List[int]] = None,
    scale: float = 0.01,
    random_state: Optional[int] = None,
    schema: Optional[FeatureSchema] = None,
    respect_schema: bool = True,
):
    """
    Add small Gaussian noise to *continuous* columns, scaled by each column's
    standard deviation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    feature_indices : list of int, optional
        Columns to perturb. If None, every noise-eligible column is perturbed
        and non-eligible columns are left untouched. If given explicitly, a
        binary or categorical column in the list raises ``ValueError`` rather
        than silently producing an impossible patient.
    scale : float
        Noise magnitude as a fraction of each column's std. Default 0.01 = 1%.
    random_state : int, optional
        Seed for reproducibility.
    schema : FeatureSchema, optional
        Column types. Inferred conservatively from ``X`` when omitted.
    respect_schema : bool
        Set False to restore the pre-0.2 behaviour of perturbing every column
        regardless of type. Emits a warning, because it generates rows that
        cannot occur in the data-generating process (fractional binary flags).

    Returns
    -------
    np.ndarray : perturbed copy of X
    """
    X_arr = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(random_state)

    if not respect_schema:
        warnings.warn(
            "gaussian_noise(respect_schema=False) applies continuous noise to "
            "binary and categorical columns, producing rows that cannot occur "
            "in the data-generating process. Results are not interpretable as "
            "semantics-preserving reliability.",
            UserWarning,
            stacklevel=2,
        )
        indices = (
            feature_indices
            if feature_indices is not None
            else list(range(X_arr.shape[1]))
        )
    else:
        sch = _resolve_schema(X_arr, schema)
        eligible = set(sch.indices_of(*NOISE_ELIGIBLE_TYPES))
        if feature_indices is None:
            indices = sorted(eligible)
        else:
            offenders = [i for i in feature_indices if i not in eligible]
            if offenders:
                detail = ", ".join(
                    f"{sch.label(i)} is {sch.type_of(i)}" for i in offenders
                )
                raise ValueError(
                    "Refusing to add continuous Gaussian noise to non-continuous "
                    f"columns: {detail}. Use 'ordinal_jitter' for ordered discrete "
                    "columns, or pass respect_schema=False to override "
                    "explicitly (not recommended)."
                )
            indices = list(feature_indices)

    for idx in indices:
        col_std = X_arr[:, idx].std()
        if col_std == 0:
            continue
        X_arr[:, idx] += rng.normal(0.0, scale * col_std, size=X_arr.shape[0])
    return X_arr


def ordinal_jitter(
    X,
    feature_indices: Sequence[int],
    max_step: int = 1,
    random_state: Optional[int] = None,
    schema: Optional[FeatureSchema] = None,
):
    """
    Move ordered discrete columns by at most ``max_step`` levels, staying on the
    observed level grid.

    This is the discrete analogue of measurement noise: a severity code recorded
    as 3 might plausibly have been recorded as 2 or 4, but never as 2.47.
    Columns must be declared ``ordinal`` in ``schema`` — ordering cannot be
    inferred from values, so an inferred schema will never authorise this.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    feature_indices : sequence of int
        Ordinal columns to jitter.
    max_step : int
        Maximum absolute move in levels. Default 1.
    random_state : int, optional
    schema : FeatureSchema, optional
        Required for validation; when omitted the caller's declaration is
        trusted and only the level grid is enforced.
    """
    X_arr = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(random_state)

    if schema is not None:
        if len(schema) != X_arr.shape[1]:
            raise ValueError(
                f"schema has {len(schema)} columns but X has {X_arr.shape[1]}."
            )
        offenders = [i for i in feature_indices if schema.type_of(i) != ORDINAL]
        if offenders:
            detail = ", ".join(
                f"{schema.label(i)} is {schema.type_of(i)}" for i in offenders
            )
            raise ValueError(f"ordinal_jitter requires ordinal columns; got {detail}.")

    for idx in feature_indices:
        levels = np.unique(X_arr[:, idx])
        if len(levels) < 2:
            continue
        pos = np.searchsorted(levels, X_arr[:, idx])
        step = rng.integers(-max_step, max_step + 1, size=X_arr.shape[0])
        new_pos = np.clip(pos + step, 0, len(levels) - 1)
        X_arr[:, idx] = levels[new_pos]
    return X_arr


def unit_rescaling(X, feature_index: int, factor: float):
    """
    Rescale a single feature column by a constant factor.

    A *genuine* unit conversion (kg -> lb, factor 2.20462; mg/dL -> mmol/L for
    glucose, factor 0.0555) re-encodes the same patient and is
    semantics-preserving. An arbitrary small percentage (e.g. 1.05) is **not** a
    unit conversion: it produces a different patient. Because the factor alone
    cannot be validated, ``unit_rescaling`` is classified as
    :data:`COVARIATE_SHIFT` by default; a spec may opt in with
    ``{"semantics": "semantics_preserving"}`` when the factor is a documented
    unit conversion.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    feature_index : int
        Column index of the feature to rescale.
    factor : float
        Multiplicative rescaling factor.
    """
    X_arr = np.array(X, dtype=float, copy=True)
    X_arr[:, feature_index] *= factor
    return X_arr


def temporal_jitter(
    X,
    date_feature_index: int,
    max_days: int = 3,
    random_state: Optional[int] = None,
):
    """
    Add uniform integer noise in [-max_days, max_days] to a date-encoded feature.

    Simulates minor encoding differences in date-derived features (e.g.,
    age in days, days-since-last-visit).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    date_feature_index : int
        Column index of the date-encoded feature.
    max_days : int
        Maximum absolute jitter in days. Default 3.
    random_state : int, optional
        Seed for reproducibility.
    """
    X_arr = np.array(X, dtype=float, copy=True)
    rng = np.random.default_rng(random_state)
    jitter = rng.integers(-max_days, max_days + 1, size=X_arr.shape[0]).astype(float)
    X_arr[:, date_feature_index] += jitter
    return X_arr


def perturbation_semantics(spec: Dict[str, Any]) -> str:
    """Classify a spec as :data:`SEMANTICS_PRESERVING` or :data:`COVARIATE_SHIFT`.

    A spec may declare ``"semantics"`` explicitly to override the default for
    its type (e.g. a documented unit conversion).
    """
    declared = spec.get("semantics")
    if declared is not None:
        if declared not in (SEMANTICS_PRESERVING, COVARIATE_SHIFT):
            raise ValueError(
                f"Unknown semantics {declared!r}. Expected "
                f"{SEMANTICS_PRESERVING!r} or {COVARIATE_SHIFT!r}."
            )
        return declared
    ptype = spec["type"]
    if ptype in COVARIATE_SHIFT_TYPES:
        return COVARIATE_SHIFT
    if ptype in SEMANTICS_PRESERVING_TYPES:
        return SEMANTICS_PRESERVING
    raise ValueError(f"Unknown perturbation type: {ptype!r}.")


def apply_perturbation(X, spec: Dict[str, Any], schema: Optional[FeatureSchema] = None):
    """
    Dispatch to the appropriate perturbation function given a spec dict.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Original feature matrix.
    spec : dict
        Must contain ``"type"``. Additional keys are perturbation-specific:

        - ``gaussian_noise``: feature_indices (optional), scale (default 0.01),
          random_state (optional), respect_schema (default True)
        - ``ordinal_jitter``: feature_indices (required), max_step (default 1),
          random_state (optional)
        - ``unit_rescaling``: feature_index (required), factor (required)
        - ``temporal_jitter``: date_feature_index (required), max_days
          (default 3), random_state (optional)

        The optional ``"semantics"`` key overrides the default classification;
        ``"label"`` names the perturbation in reports.
    schema : FeatureSchema, optional
        Column types; inferred conservatively when omitted.

    Returns
    -------
    np.ndarray : perturbed copy of X
    """
    ptype = spec["type"]
    if ptype == "gaussian_noise":
        return gaussian_noise(
            X,
            feature_indices=spec.get("feature_indices"),
            scale=spec.get("scale", 0.01),
            random_state=spec.get("random_state"),
            schema=schema,
            respect_schema=spec.get("respect_schema", True),
        )
    elif ptype == "ordinal_jitter":
        return ordinal_jitter(
            X,
            feature_indices=spec["feature_indices"],
            max_step=spec.get("max_step", 1),
            random_state=spec.get("random_state"),
            schema=schema,
        )
    elif ptype == "unit_rescaling":
        return unit_rescaling(
            X,
            feature_index=spec["feature_index"],
            factor=spec["factor"],
        )
    elif ptype == "temporal_jitter":
        return temporal_jitter(
            X,
            date_feature_index=spec["date_feature_index"],
            max_days=spec.get("max_days", 3),
            random_state=spec.get("random_state"),
        )
    else:
        raise ValueError(
            f"Unknown perturbation type: {ptype!r}. Supported: "
            "'gaussian_noise', 'ordinal_jitter', 'unit_rescaling', "
            "'temporal_jitter'."
        )
