"""Tests for the Equity dimension (Session 3 implementation)."""

import pytest

from rised.equity import evaluate_equity
from rised.results import EquityResult


def test_evaluate_equity_raises_before_implementation(
    fitted_lr, small_cohort, demographic_df
):
    X, y = small_cohort
    with pytest.raises(NotImplementedError):
        evaluate_equity(fitted_lr, X, y, demographic_df)


def test_equity_result_passed_none():
    r = EquityResult()
    assert r.passed() is False
