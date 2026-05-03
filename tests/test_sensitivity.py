"""Tests for the Sensitivity dimension (Session 3 implementation)."""

import pytest

from rised.results import SensitivityResult
from rised.sensitivity import evaluate_sensitivity


def test_evaluate_sensitivity_raises_before_implementation(fitted_lr, small_cohort):
    X, y = small_cohort
    with pytest.raises(NotImplementedError):
        evaluate_sensitivity(fitted_lr, X, y)


def test_sensitivity_result_passed_no_data():
    r = SensitivityResult()
    assert r.passed() is False
