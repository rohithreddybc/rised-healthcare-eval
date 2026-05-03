"""Tests for the Inclusivity dimension (Session 3 implementation)."""

import pytest

from rised.inclusivity import evaluate_inclusivity
from rised.results import InclusivityResult


def test_evaluate_inclusivity_raises_before_implementation(
    fitted_lr, small_cohort, demographic_df
):
    X, y = small_cohort
    with pytest.raises(NotImplementedError):
        evaluate_inclusivity(fitted_lr, X, y, demographic_df)


def test_inclusivity_result_passed_none():
    r = InclusivityResult()
    assert r.passed() is False
