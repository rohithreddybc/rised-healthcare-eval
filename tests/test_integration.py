"""
Integration tests for the RISED evaluate_all() entry point.
Full integration tests run in Session 3+ once the implementations exist.
"""

import pytest

import rised
from rised import evaluate_all


def test_evaluate_all_raises_before_implementation(fitted_lr, small_cohort, demographic_df):
    X, y = small_cohort
    with pytest.raises(NotImplementedError):
        evaluate_all(fitted_lr, X, y, demographic_df)


def test_framework_report_summary_all_none():
    from rised.results import FrameworkReport

    report = FrameworkReport()
    s = report.summary()
    assert all(v is None for v in s.values())
    assert report.all_passed() is False


def test_package_version():
    assert rised.__version__ == "0.1.0"
