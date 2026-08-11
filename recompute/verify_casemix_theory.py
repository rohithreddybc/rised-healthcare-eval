"""Verification driver for :mod:`recompute.casemix_theory`.

Six checks, in the order they appear in ``CASEMIX_DERIVATION.md``:

1. Closed form vs the existing trapezoid quadrature in ``simulate.py``, for every
   level of every case-mix geometry.
2. Grid refinement: is the residual the *quadrature's* O(dt^2) truncation error
   or the closed form's? Answered by halving dt and watching the ratio.
3. Proposition 1 is distribution-free: Gini form vs the AUROC definition for
   skewed, bimodal, heavy-tailed and bounded linear predictors.
4. Monotonicity in the scale parameter, inside symmetric location-free families.
5. The counterexample: linear-predictor SD does *not* order AUROC across
   different shapes.
6. Empirical rejection rate of a studentized max-min test as n grows, against the
   Hoeffding lower bound.

Run: ``python -m recompute.verify_casemix_theory``. Writes
``recompute/results/casemix_theory.json``. Every stochastic step is seeded from
``casemix_theory.CASEMIX_SEED``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.stats import norm

from recompute.casemix_theory import (
    CASEMIX_SEED,
    auroc_definition_from_risk,
    auroc_from_risk,
    auroc_gaussian_lp,
    auroc_rare_outcome_limit,
    expit,
    n_for_power,
    power_lower_bound,
    subgroup_auroc_gap,
    subgroup_prevalence,
)
from recompute.comparators.simulate import GEOMETRIES, true_subgroup_auc

RESULTS = Path(__file__).resolve().parent / "results"


# ── 1. closed form vs the repository's quadrature ────────────────────────────

def _is_gaussian_shared_intercept(case_mix) -> bool:
    """Does this geometry match the contract Proposition 2 is stated under?

    Proposition 2 assumes (a) a Gaussian linear predictor and (b) one shared
    intercept, so that level ``k``'s linear-predictor mean is ``b0 + loc_k``.
    ``getattr`` with defaults is deliberate: at the time of writing another
    branch is adding ``lp_dist`` (heavy-tailed / skewed linear predictors) and
    ``equalize_prevalence`` (a per-level intercept solve) to ``CaseMix``. Those
    geometries are outside Proposition 2's assumptions -- and the non-Gaussian
    ones are outside it *by design*, since they vary the shape at fixed SD. They
    are skipped here with a recorded reason rather than silently mis-compared;
    :func:`check_distribution_free` is where the non-Gaussian case is handled,
    using Proposition 1, which needs no shape assumption at all.
    """
    return (getattr(case_mix, "lp_dist", "normal") == "normal"
            and not getattr(case_mix, "equalize_prevalence", False))


def check_against_quadrature() -> List[Dict]:
    rows: List[Dict] = []
    for geom in GEOMETRIES:
        if geom.case_mix is None:
            continue
        if not _is_gaussian_shared_intercept(geom.case_mix):
            rows.append({
                "geometry": geom.name, "level": "SKIPPED", "loc": None,
                "sd": None, "intercept": None, "prevalence": None,
                "quadrature": None, "closed_form": None, "abs_diff": 0.0,
                "skip_reason": (
                    f"lp_dist={getattr(geom.case_mix, 'lp_dist', 'normal')!r}, "
                    f"equalize_prevalence="
                    f"{getattr(geom.case_mix, 'equalize_prevalence', False)!r}"
                    " -- outside Proposition 2's assumptions"),
            })
            continue
        truth = true_subgroup_auc(geom)
        b0 = truth["intercept"]
        for k, (loc, scale) in enumerate(zip(geom.case_mix.locs, geom.case_mix.scales)):
            quad_val = truth[f"level_{k}"]
            closed = auroc_gaussian_lp(b0 + loc, scale)
            rows.append({
                "geometry": geom.name, "level": k, "loc": loc, "sd": scale,
                "intercept": b0,
                "prevalence": subgroup_prevalence(b0 + loc, scale),
                "quadrature": quad_val, "closed_form": closed,
                "abs_diff": abs(quad_val - closed),
            })
        gap_q = truth.get("max_gap", float("nan"))
        gap_c = subgroup_auroc_gap([b0 + l for l in geom.case_mix.locs],
                                   geom.case_mix.scales)
        rows.append({
            "geometry": geom.name, "level": "max_gap", "loc": None, "sd": None,
            "intercept": b0, "prevalence": None,
            "quadrature": gap_q, "closed_form": gap_c,
            "abs_diff": abs(gap_q - gap_c),
        })
    return rows


# ── 2. whose error is it? refine the trapezoid grid ──────────────────────────

def _repo_style_quadrature(b0: float, loc: float, scale: float, npts: int,
                           half: float = 40.0) -> float:
    """``simulate.true_subgroup_auc`` for one level, with the grid size exposed."""
    t = np.linspace(-half, half, npts)
    phi = np.exp(-0.5 * ((t - loc) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    p = expit(b0 + t)
    f_pos = phi * p
    f_neg = phi * (1.0 - p)
    f_pos = f_pos / np.trapz(f_pos, t)
    f_neg = f_neg / np.trapz(f_neg, t)
    dt = t[1] - t[0]
    cdf_neg = np.cumsum(f_neg) * dt - 0.5 * f_neg * dt
    return float(np.trapz(f_pos * cdf_neg, t))


def check_grid_refinement() -> List[Dict]:
    geom = next(g for g in GEOMETRIES if g.name == "casemix_moderate_3")
    b0 = true_subgroup_auc(geom)["intercept"]
    loc, scale = geom.case_mix.locs[0], geom.case_mix.scales[0]
    ref = auroc_gaussian_lp(b0 + loc, scale)
    rows: List[Dict] = []
    for npts in (40_001, 80_001, 160_001, 320_001):
        val = _repo_style_quadrature(b0, loc, scale, npts)
        dt = 80.0 / (npts - 1)
        rows.append({"npts": npts, "dt": dt, "quadrature": val,
                     "closed_form": ref, "signed_diff": val - ref,
                     "diff_over_dt2": (val - ref) / dt ** 2})
    return rows


# ── 3. Proposition 1 without the Gaussian assumption ─────────────────────────

def _shapes(rng: np.random.Generator, n: int) -> Dict[str, np.ndarray]:
    return {
        "gaussian(0,1)": rng.standard_normal(n),
        "laplace": rng.laplace(0.0, 1.0 / np.sqrt(2.0), n),
        "right-skew (lognormal - 3)": rng.lognormal(0.0, 0.8, n) - 3.0,
        "left-skew (1 - lognormal)": 1.0 - rng.lognormal(0.0, 0.8, n),
        "bimodal (+-2)": np.where(rng.random(n) < 0.5, -2.0, 2.0)
                         + 0.3 * rng.standard_normal(n),
        "uniform(-3,3)": rng.uniform(-3.0, 3.0, n),
        "student-t(3)": rng.standard_t(3, n),
        "chi-square(5) - 5": rng.chisquare(5, n) - 5.0,
    }


def check_distribution_free(n: int = 2_000_000) -> List[Dict]:
    rng = np.random.default_rng(CASEMIX_SEED)
    rows: List[Dict] = []
    for name, lp in _shapes(rng, n).items():
        risk = expit(lp)
        gini = auroc_from_risk(risk)
        defn = auroc_definition_from_risk(risk)
        rows.append({"shape": name, "sd_lp": float(np.std(lp)),
                     "prevalence": float(np.mean(risk)),
                     "gini_form": gini, "definition": defn,
                     "abs_diff": abs(gini - defn)})
    return rows


# ── 4. monotone in the scale parameter, any symmetric shape ──────────────────

def check_scale_monotonicity(n: int = 1_000_000) -> List[Dict]:
    rng = np.random.default_rng(CASEMIX_SEED + 1)
    bases = {
        "gaussian": rng.standard_normal(n),
        "laplace": rng.laplace(0.0, 1.0, n),
        "uniform": rng.uniform(-1.0, 1.0, n),
        "bimodal": np.where(rng.random(n) < 0.5, -1.0, 1.0)
                   + 0.2 * rng.standard_normal(n),
        "student-t(3)": rng.standard_t(3, n),
    }
    scales = np.linspace(0.1, 4.0, 40)
    rows: List[Dict] = []
    for name, x in bases.items():
        x = x - np.median(x)
        sym = np.concatenate([x, -x])       # exact symmetry about 0
        vals = [auroc_from_risk(expit(c * sym)) for c in scales]
        d = np.diff(vals)
        rows.append({"shape": name, "min_forward_diff": float(d.min()),
                     "auroc_at_min_scale": vals[0], "auroc_at_max_scale": vals[-1],
                     "monotone_increasing": bool(d.min() >= 0.0)})
    return rows


# ── 5. SD does not order AUROC across shapes ─────────────────────────────────

def check_sd_counterexample(n: int = 2_000_000) -> List[Dict]:
    rng = np.random.default_rng(CASEMIX_SEED + 2)
    sign = lambda: np.where(rng.random(n) < 0.5, -1.0, 1.0)
    cases = {
        "N(0, 0.6^2)": 0.6 * rng.standard_normal(n),
        "two-atom +-0.6": 0.6 * sign(),
        "spike-and-slab: 99% at +-0.35, 1% at +-25": np.where(
            rng.random(n) < 0.01, 25.0 * sign(), 0.35 * sign()),
    }
    rows: List[Dict] = []
    for name, lp in cases.items():
        rows.append({"shape": name, "sd_lp": float(np.std(lp)),
                     "auroc": auroc_from_risk(expit(lp))})
    return rows


# ── 6. the equal-AUROC null rejects with probability -> 1 ────────────────────

def check_power(reps: int = 400) -> List[Dict]:
    """Studentized max-min test at nominal 5%, Bonferroni over pairs."""
    from recompute.comparators.core import auc_delong

    geom = next(g for g in GEOMETRIES if g.name == "casemix_moderate_3")
    b0 = true_subgroup_auc(geom)["intercept"]
    locs = np.asarray(geom.case_mix.locs, dtype=float)
    sds = np.asarray(geom.case_mix.scales, dtype=float)
    n_groups = len(locs)
    true_gap = subgroup_auroc_gap(b0 + locs, sds)
    n_pairs = n_groups * (n_groups - 1) // 2
    crit = float(norm.ppf(1.0 - 0.05 / (2.0 * n_pairs)))

    rows: List[Dict] = []
    for n_total in (500, 1_000, 2_000, 5_000, 10_000, 20_000):
        rng = np.random.default_rng([CASEMIX_SEED, n_total])
        rejects = 0
        min_events = []
        var_extreme: List[float] = []
        for _ in range(reps):
            codes = rng.integers(0, n_groups, size=n_total)
            lp = b0 + locs[codes] + sds[codes] * rng.standard_normal(n_total)
            risk = expit(lp)
            y = (rng.random(n_total) < risk).astype(int)
            est = []
            for k in range(n_groups):
                sel = codes == k
                a, v = auc_delong(y[sel], risk[sel])
                if np.isfinite(a) and np.isfinite(v):
                    est.append((a, v))
                n_pos = int(y[sel].sum())
                min_events.append(min(n_pos, int(sel.sum()) - n_pos))
            hit = False
            for i in range(len(est)):
                for j in range(i + 1, len(est)):
                    den = est[i][1] + est[j][1]
                    if den > 0 and abs(est[i][0] - est[j][0]) / np.sqrt(den) > crit:
                        hit = True
            rejects += int(hit)
            if len(est) == n_groups:
                var_extreme.append(est[0][1] + est[-1][1])
        m_min = int(np.min(min_events))
        # Sharp (asymptotic-normal) prediction for the widest pair, which is the
        # pair the maximum is almost always attained at here: the studentized
        # difference is approximately N(gap / sqrt(v_0 + v_{G-1}), 1).
        v_pair = float(np.mean(var_extreme)) if var_extreme else float("nan")
        ncp = true_gap / np.sqrt(v_pair) if v_pair > 0 else float("nan")
        predicted = float(norm.cdf(ncp - crit) + norm.cdf(-ncp - crit))
        rows.append({
            "n_total": n_total, "reps": reps, "true_gap": true_gap,
            "empirical_rejection_rate": rejects / reps,
            "mean_min_class_count": float(np.mean(min_events)),
            "mean_pair_variance": v_pair,
            "noncentrality": float(ncp),
            "predicted_rejection_rate": predicted,
            "hoeffding_lower_bound": power_lower_bound(true_gap, m_min, n_groups),
        })
    return rows


# ── the Austin & Steyerberg limit ────────────────────────────────────────────

def check_rare_outcome_limit() -> List[Dict]:
    rows: List[Dict] = []
    for sd in (0.6, 0.7, 0.9, 1.0, 1.3, 1.4, 1.9):
        rows.append({
            "sd_lp": sd,
            "auroc_at_prev_half": auroc_gaussian_lp(0.0, sd),
            "auroc_at_mean_-12": auroc_gaussian_lp(-12.0, sd),
            "austin_steyerberg_Phi_sd_over_sqrt2": auroc_rare_outcome_limit(sd),
        })
    return rows


def main() -> Dict:
    out = {
        "seed": CASEMIX_SEED,
        "quadrature_agreement": check_against_quadrature(),
        "grid_refinement": check_grid_refinement(),
        "distribution_free": check_distribution_free(),
        "scale_monotonicity": check_scale_monotonicity(),
        "sd_counterexample": check_sd_counterexample(),
        "rare_outcome_limit": check_rare_outcome_limit(),
        "power": check_power(),
    }
    compared = [r for r in out["quadrature_agreement"] if r["level"] != "SKIPPED"]
    out["n_levels_compared"] = len(compared)
    out["geometries_skipped"] = [r["geometry"] for r in out["quadrature_agreement"]
                                 if r["level"] == "SKIPPED"]
    out["max_abs_diff_vs_quadrature"] = max(r["abs_diff"] for r in compared)
    out["n_for_power_80"] = {
        "gap_0.036_mild": n_for_power(0.036, 0.8, 3),
        "gap_0.123_moderate": n_for_power(0.123, 0.8, 3),
        "gap_0.199_strong": n_for_power(0.199, 0.8, 4),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "casemix_theory.json").write_text(
        json.dumps(out, indent=2, default=float), encoding="utf-8")

    print(f"levels compared: {out['n_levels_compared']}   "
          f"geometries skipped: {out['geometries_skipped']}")
    print(f"max |closed form - quadrature|: "
          f"{out['max_abs_diff_vs_quadrature']:.3e}")
    for r in out["grid_refinement"]:
        print(f"  dt={r['dt']:.2e}  signed diff={r['signed_diff']:+.3e}  "
              f"diff/dt^2={r['diff_over_dt2']:+.5f}")
    for r in out["power"]:
        print(f"  n={r['n_total']:>6}  empirical={r['empirical_rejection_rate']:.3f}"
              f"  predicted={r['predicted_rejection_rate']:.3f}"
              f"  Hoeffding>={r['hoeffding_lower_bound']:.3f}")
    return out


if __name__ == "__main__":
    main()
