"""
Model selection and hyperparameter tuning utilities for fd2 notebook.

This module provides functions for:
- Model comparison and visualization
- Hyperparameter tuning with logging
- Cross-validation results analysis
- Bias-variance diagnostics

Example:
    >>> from src.fd2_nb import compare_models, create_search_object, analyze_cv_results
    >>> from src.fd2_nb import analyze_train_val_gaps, track_xgboost_iterations
"""

from .model_comparison import (
    compare_models,
    get_best_model,
)

from .hyperparameter_tuning import (
    create_search_object,
    tune_with_logging,
    get_best_params_summary,
)

from .cv_analysis import (
    analyze_cv_results,
    get_cv_statistics,
)

from .bias_variance import (
    calculate_train_val_gap,
    analyze_train_val_gaps,
    track_xgboost_iterations,
    analyze_cv_fold_variance,
    generate_bias_variance_report,
)

__all__ = [
    # Model comparison
    'compare_models',
    'get_best_model',
    # Hyperparameter tuning
    'create_search_object',
    'tune_with_logging',
    'get_best_params_summary',
    # CV analysis
    'analyze_cv_results',
    'get_cv_statistics',
    # Bias-variance
    'calculate_train_val_gap',
    'analyze_train_val_gaps',
    'track_xgboost_iterations',
    'analyze_cv_fold_variance',
    'generate_bias_variance_report',
]
