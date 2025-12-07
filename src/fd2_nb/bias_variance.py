"""
Bias-variance analysis utilities for model diagnostics.

This module provides functions for analyzing bias-variance tradeoffs in
classification models, specifically cross-validation fold variance analysis.

Note: Train-validation gap analysis and iteration performance tracking are now
handled by functions in cv_analysis.py (analyze_cv_train_val_gap,
analyze_iteration_performance) which leverage GridSearchCV results directly.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_cv_fold_variance(
    cv_results_paths: Dict[str, str],
    refit_metric: str = 'pr_auc',
    figsize: Tuple[int, int] = (14, 5),
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze variance across CV folds from saved CV results.

    Computes coefficient of variation (CV) to assess model stability:
    - CV < 3%: Low variance (stable)
    - CV 3-5%: Moderate variance
    - CV > 5%: High variance (unstable)

    Args:
        cv_results_paths: Dictionary mapping model names to CV results CSV paths.
            Supports glob patterns (e.g., 'models/logs/rf_cv_results_*.csv')
        refit_metric: The metric used for refit in multi-metric scoring (default: 'pr_auc').
            With multi-metric scoring, columns are named 'mean_test_{metric}' instead of
            'mean_test_score'. Set to None for single-metric scoring (legacy).
        figsize: Figure size for variance plot
        verbose: If True, print analysis and create plots

    Returns:
        DataFrame with variance analysis for each model

    Example:
        >>> variance_df = analyze_cv_fold_variance({
        ...     'Random Forest': 'models/logs/random_forest_cv_results_*.csv',
        ...     'XGBoost': 'models/logs/xgboost_cv_results_*.csv'
        ... }, refit_metric='pr_auc')
    """
    results = []

    for model_name, path_pattern in cv_results_paths.items():
        # Handle glob patterns
        path = Path(path_pattern)
        if '*' in str(path):
            matching_files = sorted(path.parent.glob(path.name))
            if not matching_files:
                if verbose:
                    print(f"Warning: No files found matching {path_pattern}")
                continue
            cv_path = matching_files[-1]  # Use most recent
        else:
            cv_path = path

        if not cv_path.exists():
            if verbose:
                print(f"Warning: {cv_path} not found")
            continue

        # Load and analyze
        cv_results = pd.read_csv(cv_path)

        # Determine column names based on scoring type (multi-metric vs single-metric)
        if refit_metric and f'mean_test_{refit_metric}' in cv_results.columns:
            mean_score_col = f'mean_test_{refit_metric}'
            std_score_col = f'std_test_{refit_metric}'
            rank_col = f'rank_test_{refit_metric}'
        else:
            mean_score_col = 'mean_test_score'
            std_score_col = 'std_test_score'
            rank_col = 'rank_test_score'

        best_idx = cv_results[rank_col].idxmin()
        best_row = cv_results.loc[best_idx]

        mean_score = best_row[mean_score_col]
        std_score = best_row[std_score_col]
        cv_coef = (std_score / mean_score) * 100 if mean_score > 0 else 0

        results.append({
            'model': model_name,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_coef_pct': cv_coef
        })

        if verbose:
            print(f"\n{model_name} (Best Config):")
            print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
            print(f"  CV Coefficient: {cv_coef:.2f}%")

    variance_df = pd.DataFrame(results)

    if verbose and len(variance_df) > 0:
        print("\n" + "-" * 60)
        print("Stability Diagnosis:")
        for _, row in variance_df.iterrows():
            if row['cv_coef_pct'] > 5:
                diag = "High variance (unstable)"
            elif row['cv_coef_pct'] > 3:
                diag = "Moderate variance"
            else:
                diag = "Low variance (stable)"
            print(f"  {row['model']}: {diag}")

        _plot_cv_variance(variance_df, figsize)

    return variance_df


def _plot_cv_variance(variance_df: pd.DataFrame, figsize: Tuple[int, int]) -> None:
    """Create CV variance visualization."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean score with error bars
    axes[0].bar(variance_df['model'], variance_df['mean_score'],
                yerr=variance_df['std_score'], capsize=10, alpha=0.7)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Mean Score Across CV Folds')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: CV coefficient
    colors = ['red' if cv > 5 else 'orange' if cv > 3 else 'green'
              for cv in variance_df['cv_coef_pct']]
    axes[1].bar(variance_df['model'], variance_df['cv_coef_pct'], color=colors, alpha=0.7)
    axes[1].axhline(5, color='red', linestyle='--', alpha=0.5, label='High (5%)')
    axes[1].axhline(3, color='orange', linestyle='--', alpha=0.5, label='Moderate (3%)')
    axes[1].set_ylabel('Coefficient of Variation (%)')
    axes[1].set_title('Model Stability (Lower is Better)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
