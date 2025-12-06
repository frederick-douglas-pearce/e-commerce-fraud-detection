"""
Hyperparameter tuning utilities for model selection.

This module provides general-purpose functions for creating and executing
hyperparameter searches with logging capabilities. All functions are designed
to be reusable across different models and projects.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# Default multi-metric scoring for fraud detection
DEFAULT_SCORING = {
    'pr_auc': 'average_precision',
    'roc_auc': 'roc_auc',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
    'accuracy': 'accuracy'
}


def create_search_object(
    search_type: str,
    estimator: Any,
    param_grid: Dict,
    scoring: Union[str, Dict, List] = None,
    refit: Union[str, bool] = 'pr_auc',
    cv: Any = None,
    n_iter: Optional[int] = None,
    verbose: int = 1,
    random_state: Optional[int] = None,
    n_jobs: int = -1
) -> Union[GridSearchCV, RandomizedSearchCV]:
    """
    Create GridSearchCV or RandomizedSearchCV based on search_type parameter.

    Factory function that abstracts the creation of hyperparameter search objects,
    providing a unified interface for both grid and random search strategies.
    Supports multi-metric scoring for comprehensive model evaluation.

    Args:
        search_type: Type of search - 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        estimator: The estimator object to fit (e.g., Pipeline, classifier)
        param_grid: Dictionary with parameter names as keys and lists of settings.
            For RandomizedSearchCV, can include distributions.
        scoring: Strategy to evaluate performance. Can be:
            - None: Uses DEFAULT_SCORING (pr_auc, roc_auc, f1, precision, recall, accuracy)
            - str: Single metric (e.g., 'average_precision')
            - dict: Multiple metrics (e.g., {'pr_auc': 'average_precision', 'f1': 'f1'})
            - list: Multiple metric names
        refit: Which metric to use for selecting best model and refitting.
            - str: Metric name (must be key in scoring dict if scoring is dict)
            - True: Use first/only metric (only valid with single metric scoring)
            - False: Don't refit (best_estimator_ won't be available)
            Default: 'pr_auc' (for fraud detection optimization)
        cv: Cross-validation splitting strategy. If None, uses 5-fold.
        n_iter: Number of parameter settings to sample (RandomizedSearchCV only).
            If None and search_type='random', defaults to 10.
        verbose: Verbosity level for search output
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 uses all processors)

    Returns:
        Configured GridSearchCV or RandomizedSearchCV object ready to fit

    Raises:
        ValueError: If search_type is not 'grid' or 'random'

    Example:
        >>> # Multi-metric scoring (recommended)
        >>> search = create_search_object(
        ...     search_type='grid',
        ...     estimator=pipeline,
        ...     param_grid={'classifier__n_estimators': [100, 200]},
        ...     scoring=None,  # Uses default multi-metric
        ...     refit='pr_auc',  # Select best by PR-AUC
        ...     cv=cv_strategy
        ... )

        >>> # Single metric scoring (legacy)
        >>> search = create_search_object(
        ...     search_type='grid',
        ...     estimator=pipeline,
        ...     param_grid={'classifier__n_estimators': [100, 200]},
        ...     scoring='average_precision',
        ...     refit=True,
        ...     cv=cv_strategy
        ... )
    """
    # Use default multi-metric scoring if not specified
    if scoring is None:
        scoring = DEFAULT_SCORING.copy()

    search_type_lower = search_type.lower()

    if search_type_lower == 'grid':
        search_object = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            refit=refit,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs
        )
        total_combinations = _calculate_total_combinations(param_grid)
        print(f"Using GridSearchCV - will test all {total_combinations:,} combinations")

    elif search_type_lower == 'random':
        if n_iter is None:
            n_iter = 10
            print("Warning: n_iter not specified for RandomizedSearchCV, using default=10")

        search_object = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            refit=refit,
            cv=cv,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs
        )
        total_combinations = _calculate_total_combinations(param_grid)
        print(f"Using RandomizedSearchCV - sampling {n_iter} from {total_combinations:,} possible combinations")

    else:
        raise ValueError(f"search_type must be 'grid' or 'random', got '{search_type}'")

    # Print scoring info
    if isinstance(scoring, dict):
        print(f"Multi-metric scoring: {list(scoring.keys())}")
        print(f"Refit metric: {refit}")
    else:
        print(f"Single-metric scoring: {scoring}")

    return search_object


def _calculate_total_combinations(param_grid: Dict) -> int:
    """
    Calculate total number of parameter combinations.

    Args:
        param_grid: Dictionary of parameter names to lists of values

    Returns:
        Total number of combinations
    """
    total = 1
    for param_values in param_grid.values():
        if hasattr(param_values, '__len__'):
            total *= len(param_values)
        else:
            # For distributions without __len__, estimate as 1
            total *= 1
    return total


def tune_with_logging(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    log_dir: str = 'models/logs',
    verbose: bool = True
) -> Tuple[Union[GridSearchCV, RandomizedSearchCV], str, str]:
    """
    Fit hyperparameter search with verbose output redirected to log file.

    Executes the hyperparameter search while capturing output to a timestamped
    log file. Also saves detailed CV results to a CSV file for analysis.

    Note:
        When using n_jobs=-1 (parallel processing), sklearn spawns subprocesses
        that don't inherit stdout redirection. Only the main summary line will
        be captured in the log file. Detailed CV results are saved to CSV.

    Args:
        search_object: GridSearchCV or RandomizedSearchCV object to fit
        X: Training features
        y: Training target
        model_name: Name of the model for file naming (e.g., 'random_forest', 'xgboost')
        log_dir: Directory to save log files (default: 'models/logs')
        verbose: If True, print progress updates

    Returns:
        Tuple of (fitted_search_object, log_file_path, cv_results_path)

    Example:
        >>> search = create_search_object('grid', pipeline, param_grid, cv=cv_strategy)
        >>> fitted_search, log_path, results_path = tune_with_logging(
        ...     search, X_train_val, y_train_val, 'random_forest'
        ... )
        >>> print(f"Best score: {fitted_search.best_score_:.4f}")
    """
    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Create timestamped file paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir_path / f"{model_name}_tuning_{timestamp}.log"
    csv_path = log_dir_path / f"{model_name}_cv_results_{timestamp}.csv"

    if verbose:
        print("\nStarting hyperparameter search...")
        print(f"Verbose output will be saved to: {log_path}")
        print(f"CV results will be saved to: {csv_path}")

    # Fit with output redirection
    with open(log_path, 'w') as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            sys.stdout = log_file
            sys.stderr = log_file
            search_object.fit(X, y)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    # Save detailed CV results to CSV
    cv_results_df = pd.DataFrame(search_object.cv_results_)
    cv_results_df.to_csv(csv_path, index=False)

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"{model_name} Tuning Complete!")
        print("=" * 80)
        print(f"Best cross-validation score: {search_object.best_score_:.4f}")
        print(f"\nBest hyperparameters:")
        for param, value in search_object.best_params_.items():
            # Remove 'classifier__' prefix if present for cleaner display
            param_name = param.replace('classifier__', '')
            print(f"  - {param_name}: {value}")
        print("=" * 80)
        print(f"\nLog saved to: {log_path}")
        print(f"CV results saved to: {csv_path}")

    return search_object, str(log_path), str(csv_path)


def extract_cv_metrics(
    search_object: Union[GridSearchCV, RandomizedSearchCV]
) -> Dict[str, float]:
    """
    Extract all CV metrics for the best model from cv_results_.

    Works with both single-metric and multi-metric scoring. For multi-metric,
    extracts all metrics for the parameter combination that was best according
    to the refit metric (determined by best_index_ from the search object).

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV

    Returns:
        Dictionary mapping metric names to their CV scores for the best model.
        For multi-metric: {'pr_auc': 0.87, 'roc_auc': 0.98, 'f1': 0.78, ...}
        For single-metric: {'score': 0.87}

    Example:
        >>> metrics = extract_cv_metrics(fitted_search)
        >>> print(f"Best PR-AUC: {metrics['pr_auc']:.4f}")
        >>> print(f"Best F1: {metrics['f1']:.4f}")
    """
    best_idx = search_object.best_index_
    cv_results = search_object.cv_results_

    metrics = {}
    for key in cv_results:
        if key.startswith('mean_test_'):
            metric_name = key.replace('mean_test_', '')
            metrics[metric_name] = float(cv_results[key][best_idx])

    # If no mean_test_* columns found (single metric with old naming)
    if not metrics and 'mean_test_score' in cv_results:
        metrics['score'] = float(cv_results['mean_test_score'][best_idx])

    return metrics


def get_best_params_summary(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    model_name: str = "Model",
    refit_metric: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extract and display best parameters from fitted search object.

    Handles both single-metric and multi-metric scoring. For multi-metric,
    uses the refit_metric to identify the correct rank column.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV
        model_name: Name of model for display
        refit_metric: The metric used for refit (for multi-metric scoring).
            If None, assumes single-metric scoring.
        verbose: If True, print formatted summary

    Returns:
        Dictionary with 'best_params', 'best_score', 'cv_metrics', and 'cv_results_summary'

    Example:
        >>> summary = get_best_params_summary(fitted_search, "Random Forest", refit_metric='pr_auc')
        >>> print(f"Best PR-AUC: {summary['cv_metrics']['pr_auc']:.4f}")
    """
    # Extract best parameters (removing 'classifier__' prefix)
    best_params_clean = {
        k.replace('classifier__', ''): v
        for k, v in search_object.best_params_.items()
    }

    # Extract all CV metrics for best model
    cv_metrics = extract_cv_metrics(search_object)

    # Get CV results summary - handle multi-metric column naming
    cv_results = pd.DataFrame(search_object.cv_results_)

    # Determine rank and score column names based on scoring type
    if refit_metric and f'rank_test_{refit_metric}' in cv_results.columns:
        rank_col = f'rank_test_{refit_metric}'
        score_col = f'mean_test_{refit_metric}'
    else:
        rank_col = 'rank_test_score'
        score_col = 'mean_test_score'

    cv_summary = {
        'n_candidates': len(cv_results),
    }

    # Add rank info if available
    if rank_col in cv_results.columns:
        cv_summary['best_rank'] = int(cv_results[rank_col].min())

    # Add score stats if available
    if score_col in cv_results.columns:
        cv_summary['mean_score_all'] = cv_results[score_col].mean()
        cv_summary['std_score_all'] = cv_results[score_col].std()

    result = {
        'best_params': best_params_clean,
        'best_score': search_object.best_score_,
        'cv_metrics': cv_metrics,
        'cv_results_summary': cv_summary
    }

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"{model_name} - Best Parameters Summary")
        print("=" * 80)
        print(f"Best cross-validation score ({refit_metric or 'primary'}): {search_object.best_score_:.4f}")

        if len(cv_metrics) > 1:
            print(f"\nAll CV metrics for best model:")
            for metric, value in cv_metrics.items():
                print(f"  - {metric}: {value:.4f}")

        print(f"\nBest hyperparameters:")
        for param, value in best_params_clean.items():
            print(f"  - {param}: {value}")

        print(f"\nSearch summary:")
        print(f"  - Total candidates evaluated: {cv_summary['n_candidates']}")
        if 'mean_score_all' in cv_summary:
            print(f"  - Mean {refit_metric or 'score'} across all candidates: {cv_summary['mean_score_all']:.4f}")
            print(f"  - Std {refit_metric or 'score'} across all candidates: {cv_summary['std_score_all']:.4f}")
        print("=" * 80)

    return result
