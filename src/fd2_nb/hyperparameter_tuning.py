"""
Hyperparameter tuning utilities for model selection.

This module provides general-purpose functions for creating and executing
hyperparameter searches with logging capabilities. All functions are designed
to be reusable across different models and projects.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def create_search_object(
    search_type: str,
    estimator: Any,
    param_grid: Dict,
    scoring: str = 'average_precision',
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

    Args:
        search_type: Type of search - 'grid' for GridSearchCV, 'random' for RandomizedSearchCV
        estimator: The estimator object to fit (e.g., Pipeline, classifier)
        param_grid: Dictionary with parameter names as keys and lists of settings.
            For RandomizedSearchCV, can include distributions.
        scoring: Strategy to evaluate performance (default: 'average_precision' for PR-AUC)
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
        >>> # GridSearchCV - exhaustive search
        >>> search = create_search_object(
        ...     search_type='grid',
        ...     estimator=pipeline,
        ...     param_grid={'classifier__n_estimators': [100, 200], 'classifier__max_depth': [10, 20]},
        ...     scoring='average_precision',
        ...     cv=cv_strategy
        ... )

        >>> # RandomizedSearchCV - random sampling
        >>> search = create_search_object(
        ...     search_type='random',
        ...     estimator=pipeline,
        ...     param_grid={'classifier__n_estimators': [100, 200, 300]},
        ...     scoring='average_precision',
        ...     cv=cv_strategy,
        ...     n_iter=40
        ... )
    """
    search_type_lower = search_type.lower()

    if search_type_lower == 'grid':
        search_object = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
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
            cv=cv,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs
        )
        total_combinations = _calculate_total_combinations(param_grid)
        print(f"Using RandomizedSearchCV - sampling {n_iter} from {total_combinations:,} possible combinations")

    else:
        raise ValueError(f"search_type must be 'grid' or 'random', got '{search_type}'")

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


def get_best_params_summary(
    search_object: Union[GridSearchCV, RandomizedSearchCV],
    model_name: str = "Model",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Extract and display best parameters from fitted search object.

    Args:
        search_object: Fitted GridSearchCV or RandomizedSearchCV
        model_name: Name of model for display
        verbose: If True, print formatted summary

    Returns:
        Dictionary with 'best_params', 'best_score', and 'cv_results_summary'

    Example:
        >>> summary = get_best_params_summary(fitted_search, "Random Forest")
        >>> print(f"Best score: {summary['best_score']:.4f}")
    """
    # Extract best parameters (removing 'classifier__' prefix)
    best_params_clean = {
        k.replace('classifier__', ''): v
        for k, v in search_object.best_params_.items()
    }

    # Get CV results summary
    cv_results = pd.DataFrame(search_object.cv_results_)
    cv_summary = {
        'n_candidates': len(cv_results),
        'best_rank': int(cv_results['rank_test_score'].min()),
        'mean_score_all': cv_results['mean_test_score'].mean(),
        'std_score_all': cv_results['mean_test_score'].std(),
    }

    result = {
        'best_params': best_params_clean,
        'best_score': search_object.best_score_,
        'cv_results_summary': cv_summary
    }

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"{model_name} - Best Parameters Summary")
        print("=" * 80)
        print(f"Best cross-validation score: {search_object.best_score_:.4f}")
        print(f"\nBest hyperparameters:")
        for param, value in best_params_clean.items():
            print(f"  - {param}: {value}")
        print(f"\nSearch summary:")
        print(f"  - Total candidates evaluated: {cv_summary['n_candidates']}")
        print(f"  - Mean score across all candidates: {cv_summary['mean_score_all']:.4f}")
        print(f"  - Std score across all candidates: {cv_summary['std_score_all']:.4f}")
        print("=" * 80)

    return result
