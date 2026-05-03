"""Tests for the Reliability dimension (Session 3 implementation)."""

import pytest

from rised.reliability import evaluate_reliability
from rised.results import ReliabilityResult


def test_evaluate_reliability_raises_before_implementation(fitted_lr, small_cohort):
    X, _ = small_cohort
    with pytest.raises(NotImplementedError):
        evaluate_reliability(fitted_lr, X)


def test_reliability_result_passed_none():
    r = ReliabilityResult()
    assert r.passed() is False
