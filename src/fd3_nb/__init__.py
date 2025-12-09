"""
Model evaluation and deployment utilities for fd3 notebook.

This module provides functions for:
- Model evaluation and performance comparison
- ROC/PR curve and feature importance visualization
- Threshold optimization for different business requirements
- Deployment artifact generation

Example:
    >>> from src.fd3_nb import evaluate_model, compare_val_test_performance
    >>> from src.fd3_nb import plot_roc_pr_curves, plot_feature_importance
    >>> from src.fd3_nb import optimize_thresholds, create_threshold_comparison_df
"""

from .evaluation import (
    evaluate_model,
    compare_val_test_performance,
)

from .visualization import (
    plot_roc_pr_curves,
    plot_feature_importance,
    plot_shap_importance,
    plot_importance_comparison,
    plot_threshold_optimization,
)

from .threshold_optimization import (
    find_threshold_for_recall,
    find_optimal_f1_threshold,
    find_target_performance_threshold,
    optimize_thresholds,
    create_threshold_comparison_df,
)

from .feature_importance import (
    extract_feature_importance,
    print_feature_importance_summary,
    compute_shap_importance,
    compare_importance_methods,
    print_shap_importance_summary,
    print_importance_comparison,
)

from .deployment import (
    save_production_model,
    save_threshold_config,
    save_model_metadata,
    print_deployment_summary,
)

__all__ = [
    # Evaluation
    'evaluate_model',
    'compare_val_test_performance',
    # Visualization
    'plot_roc_pr_curves',
    'plot_feature_importance',
    'plot_shap_importance',
    'plot_importance_comparison',
    'plot_threshold_optimization',
    # Threshold optimization
    'find_threshold_for_recall',
    'find_optimal_f1_threshold',
    'find_target_performance_threshold',
    'optimize_thresholds',
    'create_threshold_comparison_df',
    # Feature importance
    'extract_feature_importance',
    'print_feature_importance_summary',
    'compute_shap_importance',
    'compare_importance_methods',
    'print_shap_importance_summary',
    'print_importance_comparison',
    # Deployment
    'save_production_model',
    'save_threshold_config',
    'save_model_metadata',
    'print_deployment_summary',
]
