"""
Model selection and hyperparameter tuning utilities for fd2 notebook.

This module provides functions for:
- Model comparison and visualization
- Hyperparameter tuning with logging
- Cross-validation results analysis
- Bias-variance diagnostics

Example:
    >>> from src.fd2_nb import compare_models, create_search_object, analyze_cv_results
    >>> from src.fd2_nb import analyze_cv_train_val_gap, analyze_iteration_performance
"""

from .model_comparison import (
    compare_models,
    get_best_model,
    plot_comprehensive_comparison,
    plot_model_comparison,
)

from .hyperparameter_tuning import (
    DEFAULT_SCORING,
    create_search_object,
    tune_with_logging,
    extract_cv_metrics,
    get_best_params_summary,
)

from .cv_analysis import (
    analyze_cv_results,
    analyze_cv_train_val_gap,
    analyze_iteration_performance,
    get_cv_statistics,
)

from .bias_variance import (
    analyze_cv_fold_variance,
)

__all__ = [
    # Model comparison
    'compare_models',
    'get_best_model',
    'plot_comprehensive_comparison',
    'plot_model_comparison',
    # Hyperparameter tuning
    'DEFAULT_SCORING',
    'create_search_object',
    'tune_with_logging',
    'extract_cv_metrics',
    'get_best_params_summary',
    # CV analysis
    'analyze_cv_results',
    'analyze_cv_train_val_gap',
    'analyze_iteration_performance',
    'get_cv_statistics',
    # Bias-variance
    'analyze_cv_fold_variance',
]
