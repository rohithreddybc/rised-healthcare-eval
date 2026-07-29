"""Tests for the Equity dimension."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from rised.equity import (
    CEILING_070_ATTAINABLE_PREVALENCE,
    attainable_rho_ceiling,
    evaluate_equity,
)
from rised.results import EquityResult


# ── F8: an independent proxy is required ─────────────────────────────────────
def test_f8_missing_need_column_raises(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    with pytest.raises(ValueError, match="requires an explicit need_column"):
        evaluate_equity(fitted_lr, X, y, demographic_df)


def test_f8_unknown_need_column_raises(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    with pytest.raises(ValueError, match="not a column of demographic_df"):
        evaluate_equity(fitted_lr, X, y, demographic_df, need_column="nope")


def test_f8_outcome_copy_proxy_raises_not_warns(
    fitted_lr, small_cohort, demographic_with_need
):
    """A proxy that *is* y_true must raise, not warn."""
    X, y = small_cohort
    with pytest.raises(ValueError, match="outcome-derived"):
        evaluate_equity(
            fitted_lr, X, y, demographic_with_need, need_column="outcome_copy"
        )


def test_f8_affinely_rescaled_outcome_proxy_raises(
    fitted_lr, small_cohort, demographic_with_need
):
    """Relabelling the outcome does not make it independent."""
    X, y = small_cohort
    with pytest.raises(ValueError, match="outcome-derived"):
        evaluate_equity(
            fitted_lr, X, y, demographic_with_need, need_column="outcome_rescaled"
        )


def test_f8_independent_proxy_is_accepted(
    fitted_lr, small_cohort, demographic_with_need
):
    X, y = small_cohort
    result = evaluate_equity(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity", subgroup_columns=["sex"],
    )
    assert isinstance(result, EquityResult)
    assert result.details["need_source"] == "comorbidity"
    assert -1.0 <= result.need_prediction_correlation <= 1.0


def test_f8_evaluate_all_skips_equity_without_a_proxy(
    fitted_lr, small_cohort, demographic_df
):
    """No silent fallback to y_true; the dimension is skipped with a reason."""
    from rised import evaluate_all

    X, y = small_cohort
    report = evaluate_all(fitted_lr, X, y, demographic_df)
    assert report.equity is None
    assert "equity_skipped_reason" in report.metadata
    assert "y_true" in report.metadata["equity_skipped_reason"]


def test_f8_evaluate_all_uses_the_supplied_proxy(
    fitted_lr, small_cohort, demographic_with_need
):
    from rised import evaluate_all

    X, y = small_cohort
    report = evaluate_all(
        fitted_lr, X, y, demographic_with_need, need_column="comorbidity"
    )
    assert report.equity is not None
    assert report.equity.details["need_source"] == "comorbidity"


# ── F8: the attainable ceiling replaces the 0.70 threshold ───────────────────
def test_f8_verified_identity_rho_equals_affine_function_of_auc():
    """P1: with a binary proxy, rho is an affine reparameterisation of AUROC."""
    rng = np.random.default_rng(0)
    n = 5000
    proxy = (rng.random(n) < 0.25).astype(int)
    scores = np.where(proxy == 1, rng.normal(1.0, 1.0, n), rng.normal(0.0, 1.0, n))
    scores = scores + rng.random(n) * 1e-9  # break ties

    rho = float(spearmanr(scores, proxy).statistic)
    auc = roc_auc_score(proxy, scores)
    p = proxy.mean()
    predicted = np.sqrt(12 * p * (1 - p)) * (n / np.sqrt(n * n - 1)) * (auc - 0.5)
    assert rho == pytest.approx(predicted, abs=1e-8)


def test_f8_ceiling_matches_the_closed_form():
    for p in (0.05, 0.112, 0.3, 0.5, 0.8):
        assert attainable_rho_ceiling(p) == pytest.approx(np.sqrt(3 * p * (1 - p)))
    assert attainable_rho_ceiling(0.5) == pytest.approx(np.sqrt(0.75))
    assert attainable_rho_ceiling(0.112) == pytest.approx(0.5462, abs=1e-3)


def test_f8_070_target_is_unreachable_outside_the_verified_prevalence_band():
    lo, hi = CEILING_070_ATTAINABLE_PREVALENCE
    assert attainable_rho_ceiling(lo) == pytest.approx(0.70, abs=1e-3)
    assert attainable_rho_ceiling(hi) == pytest.approx(0.70, abs=1e-3)
    for p in (0.05, 0.10, 0.15, 0.85, 0.95):
        assert attainable_rho_ceiling(p) < 0.70
    for p in (0.3, 0.5, 0.7):
        assert attainable_rho_ceiling(p) > 0.70


def test_f8_binary_proxy_reports_ceiling_and_prevalence(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    rng = np.random.default_rng(21)
    demo = demographic_df.copy()
    demo["housing_instability"] = (rng.random(len(X)) < 0.15).astype(float)
    result = evaluate_equity(
        fitted_lr, X, y, demo, need_column="housing_instability",
        subgroup_columns=["sex"],
    )
    assert result.proxy_prevalence == pytest.approx(
        demo["housing_instability"].mean()
    )
    assert result.attainable_rho_ceiling == pytest.approx(
        attainable_rho_ceiling(result.proxy_prevalence, n=len(X)), abs=1e-9
    )
    assert result.attainable_rho_ceiling < 0.70
    assert "0.2056" in result.details["ceiling_note"]


def test_f8_multilevel_proxy_reports_no_binary_ceiling(
    fitted_lr, small_cohort, demographic_with_need
):
    X, y = small_cohort
    result = evaluate_equity(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity", subgroup_columns=["sex"],
    )
    assert result.attainable_rho_ceiling is None
    assert result.proxy_prevalence is None
    assert "does not apply" in result.details["ceiling_note"]


def test_f8_no_hardcoded_070_threshold_survives():
    """EquityResult no longer carries a pass rule of any kind."""
    with pytest.raises(NotImplementedError, match="withdrawn"):
        EquityResult(need_prediction_correlation=0.80).passed()


def test_f8_policy_reports_equity_as_diagnostic(
    fitted_lr, small_cohort, demographic_with_need
):
    from rised import evaluate_all
    from rised.policy import Verdict, evaluate_policy

    X, y = small_cohort
    report = evaluate_all(
        fitted_lr, X, y, demographic_with_need, need_column="comorbidity"
    )
    policy = evaluate_policy(report)
    assert policy.dimensions["equity"].verdict is Verdict.DIAGNOSTIC


# ── general behaviour ────────────────────────────────────────────────────────
def test_group_need_gaps_cover_requested_subgroups(
    fitted_lr, small_cohort, demographic_with_need
):
    X, y = small_cohort
    result = evaluate_equity(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity", subgroup_columns=["sex"],
    )
    assert len(result.group_need_gaps) >= 1
    for key in result.group_need_gaps:
        assert key.startswith("sex=")


def test_proxy_bias_flags_are_subset_of_gap_keys(
    fitted_lr, small_cohort, demographic_with_need
):
    X, y = small_cohort
    result = evaluate_equity(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity", subgroup_columns=["race", "sex"],
    )
    for flag in result.proxy_bias_flags:
        assert flag in result.group_need_gaps


def test_small_groups_below_5_skipped(fitted_lr, small_cohort):
    X, y = small_cohort
    n = len(X)
    rng = np.random.default_rng(1)
    demo = pd.DataFrame({
        "group": ["majority"] * (n - 3) + ["tiny"] * 3,
        "comorbidity": rng.integers(0, 10, size=n).astype(float),
    })
    result = evaluate_equity(fitted_lr, X, y, demo, need_column="comorbidity")
    for key in result.group_need_gaps:
        assert "tiny" not in key


def test_gap_flag_threshold_is_configurable(
    fitted_lr, small_cohort, demographic_with_need
):
    X, y = small_cohort
    loose = evaluate_equity(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity", subgroup_columns=["race"], gap_flag_threshold=0.9,
    )
    tight = evaluate_equity(
        fitted_lr, X, y, demographic_with_need,
        need_column="comorbidity", subgroup_columns=["race"], gap_flag_threshold=0.0,
    )
    assert len(loose.proxy_bias_flags) <= len(tight.proxy_bias_flags)
    assert tight.details["gap_flag_threshold"] == 0.0
