"""
RISED: Reliability, Inclusivity, Sensitivity, Equity, and Deployability
Framework for Healthcare AI Decision-Support Evaluation.

Reference
---------
[AUTHOR et al., 2025] "Evaluating AI-Assisted Decision Support in High-Stakes
Healthcare: A Framework for Reliability, Sensitivity, and Equity."
arXiv: ARXIV_ID_PLACEHOLDER
"""

__version__ = "0.1.0"
__author__ = "Rohith Reddy Bellibatlu"

from rised.results import (
    ReliabilityResult,
    InclusivityResult,
    SensitivityResult,
    EquityResult,
    DeployabilityResult,
    FrameworkReport,
)

__all__ = [
    "ReliabilityResult",
    "InclusivityResult",
    "SensitivityResult",
    "EquityResult",
    "DeployabilityResult",
    "FrameworkReport",
    "evaluate_all",
]


def evaluate_all(
    model,
    X,
    y_true,
    demographic_df,
    perturbation_specs=None,
    threshold_range=None,
    feature_names=None,
    **kwargs,
) -> "FrameworkReport":
    """
    Run all five RISED dimensions and return a combined FrameworkReport.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted model with predict_proba method.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y_true : array-like of shape (n_samples,)
        Ground-truth binary labels.
    demographic_df : pd.DataFrame
        Demographic/subgroup columns aligned to X.
    perturbation_specs : list of dict, optional
        Perturbation specifications for the Reliability dimension.
    threshold_range : array-like, optional
        Decision thresholds to sweep for the Sensitivity dimension.
    feature_names : list of str, optional
        Feature names for interpretability evaluation.

    Returns
    -------
    FrameworkReport
        Combined results across all five dimensions.
    """
    raise NotImplementedError("evaluate_all() will be implemented in Session 3.")
