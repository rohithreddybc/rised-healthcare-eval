"""Tests for the Deployability dimension (Session 3 implementation)."""

import pytest

from rised.deployability import evaluate_deployability
from rised.results import DeployabilityResult


def test_evaluate_deployability_raises_before_implementation(fitted_lr, small_cohort):
    X, _ = small_cohort
    with pytest.raises(NotImplementedError):
        evaluate_deployability(fitted_lr, X)


def test_deployability_result_passed_none():
    r = DeployabilityResult()
    assert r.passed() is False
