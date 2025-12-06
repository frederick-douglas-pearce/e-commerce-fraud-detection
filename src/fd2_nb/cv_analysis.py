"""
Cross-validation results analysis utilities.

This module provides functions for analyzing and visualizing cross-validation
results from hyperparameter tuning, with focus on production deployment criteria.
All functions are designed to be reusable across different models and projects.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_cv_results(
    cv_results_path: str,
    top_n: int = 5,
    model_name: str = "Model",
    refit_metric: str = 'pr_auc',
    figsize: Tuple[int, int] = (16, 12),
    stability_threshold: float = 0.01,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze cross-validation results with focus on production deployment criteria.

    This function examines:
    - Model stability (std_test_{metric}) - RELIABLE: consistency across CV folds
    - Performance (mean_test_{metric}) - RELIABLE: metric for model selection
    - Prediction time (mean_score_time) - UNRELIABLE: affected by parallel processing
    - Training time (mean_fit_time) - UNRELIABLE: includes CV overhead

    Note:
        Timing measurements from parallel CV are unreliable - use as rough indicators only!
        Production API latency testing will provide definitive performance numbers.

    Args:
        cv_results_path: Path to the CV results CSV file
        top_n: Number of top candidates to analyze in detail (default: 5)
        model_name: Name of the model for display purposes
        refit_metric: The metric used for refit in multi-metric scoring (default: 'pr_auc').
            With multi-metric scoring, columns are named 'mean_test_{metric}' instead of
            'mean_test_score'. Set to None for single-metric scoring (legacy).
        figsize: Figure size for the 4-panel visualization
        stability_threshold: Threshold for std_test_score below which model is considered stable
        verbose: If True, display analysis and create visualizations

    Returns:
        DataFrame with top N candidates and detailed metrics

    Example:
        >>> # Multi-metric scoring (default)
        >>> top_candidates = analyze_cv_results(
        ...     'models/logs/random_forest_cv_results_20241201_120000.csv',
        ...     top_n=5,
        ...     model_name='Random Forest',
        ...     refit_metric='pr_auc'
        ... )
        >>> print(top_candidates['mean_test_pr_auc'].iloc[0])  # Best score

        >>> # Single-metric scoring (legacy)
        >>> top_candidates = analyze_cv_results(
        ...     'models/logs/old_cv_results.csv',
        ...     refit_metric=None  # Falls back to 'mean_test_score'
        ... )
    """
    # Load CV results
    cv_results = pd.read_csv(cv_results_path)

    # Determine column names based on scoring type (multi-metric vs single-metric)
    if refit_metric and f'mean_test_{refit_metric}' in cv_results.columns:
        # Multi-metric scoring
        mean_score_col = f'mean_test_{refit_metric}'
        std_score_col = f'std_test_{refit_metric}'
        rank_col = f'rank_test_{refit_metric}'
    else:
        # Single-metric scoring (legacy) or fallback
        mean_score_col = 'mean_test_score'
        std_score_col = 'std_test_score'
        rank_col = 'rank_test_score'

    # Extract key columns
    key_cols = [mean_score_col, std_score_col, 'mean_fit_time',
                'std_fit_time', 'mean_score_time', 'std_score_time', rank_col]

    # Filter to only columns that exist in the results
    key_cols = [c for c in key_cols if c in cv_results.columns]

    # Add parameter columns
    param_cols = [col for col in cv_results.columns if col.startswith('param_')]
    display_cols = key_cols + param_cols

    # Get top N candidates by test score
    top_candidates = cv_results.nlargest(top_n, mean_score_col)[display_cols].copy()

    # Get best model
    best_idx = cv_results[rank_col].idxmin()
    best_model = cv_results.loc[best_idx]

    # Create column name mapping for helper functions
    col_mapping = {
        'mean_score': mean_score_col,
        'std_score': std_score_col,
        'rank': rank_col
    }

    if verbose:
        _print_analysis_header(model_name, refit_metric)
        _display_top_candidates(top_candidates, top_n, col_mapping)
        _display_statistical_summary(cv_results, key_cols, col_mapping)
        _display_best_model_details(best_model, stability_threshold, col_mapping)
        _create_analysis_plots(cv_results, best_model, top_candidates, top_n, figsize, col_mapping)
        _print_recommendations(best_model, top_candidates, stability_threshold, col_mapping)

    return top_candidates


def _print_analysis_header(model_name: str, refit_metric: Optional[str] = None) -> None:
    """Print analysis header with timing caveat."""
    print("\n" + "=" * 100)
    print(f"{model_name} - Cross-Validation Results Analysis")
    print("=" * 100)
    if refit_metric:
        print(f"Refit metric: {refit_metric}")
    print("TIMING CAVEAT: Due to parallel processing (n_jobs=-1), timing measurements may be")
    print("   unreliable. Small differences (< 20-30%) are often just measurement noise.")
    print("   Focus on PR-AUC and stability for model selection. Production API testing will")
    print("   provide definitive latency numbers.")
    print("=" * 100)


def _display_top_candidates(
    top_candidates: pd.DataFrame,
    top_n: int,
    col_mapping: Dict[str, str]
) -> None:
    """Display top N candidates with styled formatting."""
    mean_score_col = col_mapping['mean_score']
    std_score_col = col_mapping['std_score']

    print(f"\nTop {top_n} Candidates by Test Score:")
    print("-" * 100)

    # Format dictionary for display - use actual column names
    format_dict = {
        mean_score_col: '{:.6f}',
        std_score_col: '{:.6f}',
        'mean_fit_time': '{:.2f}',
        'std_fit_time': '{:.2f}',
        'mean_score_time': '{:.4f}',
        'std_score_time': '{:.4f}'
    }
    # Filter to only columns that exist
    format_dict = {k: v for k, v in format_dict.items() if k in top_candidates.columns}

    try:
        from IPython.display import display
        styled = top_candidates.style.format(format_dict)
        if mean_score_col in top_candidates.columns:
            styled = styled.background_gradient(cmap='RdYlGn', subset=[mean_score_col])
        display(styled)
    except ImportError:
        print(top_candidates.to_string())


def _display_statistical_summary(
    cv_results: pd.DataFrame,
    key_cols: List[str],
    col_mapping: Dict[str, str]
) -> None:
    """Display statistical summary across all candidates."""
    print("\n" + "-" * 100)
    print("Statistical Summary Across All Candidates:")
    print("-" * 100)

    # Exclude rank column from statistics
    rank_col = col_mapping['rank']
    stat_cols = [c for c in key_cols if c != rank_col and c in cv_results.columns]
    summary_stats = cv_results[stat_cols].describe().loc[['mean', 'std', 'min', 'max']]

    try:
        from IPython.display import display
        display(summary_stats.style.format('{:.6f}'))
    except ImportError:
        print(summary_stats.to_string())


def _display_best_model_details(
    best_model: pd.Series,
    stability_threshold: float = 0.01,
    col_mapping: Optional[Dict[str, str]] = None
) -> None:
    """Display detailed metrics for best model."""
    # Get column names from mapping or use defaults
    if col_mapping:
        mean_score_col = col_mapping['mean_score']
        std_score_col = col_mapping['std_score']
    else:
        mean_score_col = 'mean_test_score'
        std_score_col = 'std_test_score'

    print("\n" + "-" * 100)
    print("Best Model (Rank 1) - Detailed Metrics:")
    print("-" * 100)

    std_score = best_model[std_score_col]
    mean_score = best_model[mean_score_col]
    stability_status = "Stable" if std_score < stability_threshold else "Variable"

    print(f"  - Test Score (mean +/- std):    {mean_score:.6f} +/- {std_score:.6f}")
    print(f"  - Stability (CV std):           {std_score:.6f} ({stability_status})")
    print(f"  - Training time (mean +/- std): {best_model['mean_fit_time']:.2f}s +/- {best_model['std_fit_time']:.2f}s (unreliable)")
    print(f"  - Prediction time (mean +/- std): {best_model['mean_score_time']:.4f}s +/- {best_model['std_score_time']:.4f}s (unreliable)")

    if best_model['mean_score_time'] > 0:
        print(f"  - Est. throughput:              ~{1/best_model['mean_score_time']:.0f} predictions/sec (unreliable)")

    print("-" * 100)


def _create_analysis_plots(
    cv_results: pd.DataFrame,
    best_model: pd.Series,
    top_candidates: pd.DataFrame,
    top_n: int,
    figsize: Tuple[int, int],
    col_mapping: Dict[str, str]
) -> None:
    """Create 4-panel analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    _plot_performance_vs_stability(cv_results, best_model, axes[0, 0], col_mapping)
    _plot_performance_vs_time(cv_results, best_model, axes[0, 1], col_mapping)
    _plot_top_candidates_comparison(cv_results, top_candidates, top_n, axes[1, 0], col_mapping)
    _plot_training_vs_prediction_time(cv_results, best_model, axes[1, 1], col_mapping)

    plt.tight_layout()
    plt.show()


def _plot_performance_vs_stability(
    cv_results: pd.DataFrame,
    best_model: pd.Series,
    ax: plt.Axes,
    col_mapping: Dict[str, str]
) -> None:
    """Plot 1: Performance vs Stability scatter plot."""
    mean_score_col = col_mapping['mean_score']
    std_score_col = col_mapping['std_score']
    rank_col = col_mapping['rank']

    scatter = ax.scatter(
        cv_results[mean_score_col],
        cv_results[std_score_col],
        c=cv_results[rank_col],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )

    # Highlight best model
    ax.scatter(
        best_model[mean_score_col],
        best_model[std_score_col],
        c='red',
        s=300,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Best Model',
        zorder=10
    )

    ax.set_xlabel('Mean Test Score - Reliable', fontsize=12)
    ax.set_ylabel('Std Test Score (Stability) - Reliable', fontsize=12)
    ax.set_title('Performance vs Stability (Both Metrics Reliable)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Rank')


def _plot_performance_vs_time(
    cv_results: pd.DataFrame,
    best_model: pd.Series,
    ax: plt.Axes,
    col_mapping: Dict[str, str]
) -> None:
    """Plot 2: Performance vs Prediction Time scatter plot."""
    mean_score_col = col_mapping['mean_score']
    rank_col = col_mapping['rank']

    scatter = ax.scatter(
        cv_results[mean_score_col],
        cv_results['mean_score_time'],
        c=cv_results[rank_col],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )

    # Highlight best model
    ax.scatter(
        best_model[mean_score_col],
        best_model['mean_score_time'],
        c='red',
        s=300,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Best Model',
        zorder=10
    )

    ax.set_xlabel('Mean Test Score - Reliable', fontsize=12)
    ax.set_ylabel('Mean Prediction Time - Unreliable', fontsize=12)
    ax.set_title('Performance vs Prediction Time (Time Unreliable)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Rank')


def _plot_top_candidates_comparison(
    cv_results: pd.DataFrame,
    top_candidates: pd.DataFrame,
    top_n: int,
    ax: plt.Axes,
    col_mapping: Dict[str, str]
) -> None:
    """Plot 3: Top N Candidates normalized comparison."""
    mean_score_col = col_mapping['mean_score']
    rank_col = col_mapping['rank']

    x = np.arange(len(top_candidates))
    width = 0.35

    # Normalize scores for comparison
    score_min = cv_results[mean_score_col].min()
    score_max = cv_results[mean_score_col].max()
    time_min = cv_results['mean_score_time'].min()
    time_max = cv_results['mean_score_time'].max()

    score_normalized = (top_candidates[mean_score_col] - score_min) / (score_max - score_min + 1e-10)
    time_normalized = 1 - (top_candidates['mean_score_time'] - time_min) / (time_max - time_min + 1e-10)

    ax.bar(x - width/2, score_normalized, width, label='Performance (Reliable)', color='steelblue')
    ax.bar(x + width/2, time_normalized, width, label='Speed (Unreliable)', color='coral')

    ax.set_xlabel('Candidate Rank', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title(f'Top {top_n} Candidates: Performance vs Speed', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {int(r)}" for r in top_candidates[rank_col]])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])


def _plot_training_vs_prediction_time(
    cv_results: pd.DataFrame,
    best_model: pd.Series,
    ax: plt.Axes,
    col_mapping: Dict[str, str]
) -> None:
    """Plot 4: Training Time vs Prediction Time scatter plot."""
    mean_score_col = col_mapping['mean_score']

    scatter = ax.scatter(
        cv_results['mean_fit_time'],
        cv_results['mean_score_time'],
        c=cv_results[mean_score_col],
        cmap='RdYlGn',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )

    # Highlight best model
    ax.scatter(
        best_model['mean_fit_time'],
        best_model['mean_score_time'],
        c='red',
        s=300,
        marker='*',
        edgecolors='black',
        linewidths=2,
        label='Best Model',
        zorder=10
    )

    ax.set_xlabel('Mean Training Time - Unreliable', fontsize=12)
    ax.set_ylabel('Mean Prediction Time - Unreliable', fontsize=12)
    ax.set_title('Training vs Prediction Time (Both Unreliable)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Test Score')


def _print_recommendations(
    best_model: pd.Series,
    top_candidates: pd.DataFrame,
    stability_threshold: float = 0.01,
    col_mapping: Optional[Dict[str, str]] = None
) -> None:
    """Print model selection recommendations."""
    # Get column names from mapping or use defaults
    if col_mapping:
        mean_score_col = col_mapping['mean_score']
        std_score_col = col_mapping['std_score']
    else:
        mean_score_col = 'mean_test_score'
        std_score_col = 'std_test_score'

    print("\n" + "=" * 100)
    print("Recommendations:")
    print("=" * 100)

    # Check stability
    if best_model[std_score_col] < stability_threshold:
        print("  - Best model shows good stability (low CV variance)")
    else:
        print(f"  - Best model shows some variance across folds (std > {stability_threshold})")
        print("    Consider if a more stable alternative in top candidates might be preferred")

    # Check if top candidates are close in performance
    if len(top_candidates) > 1 and mean_score_col in top_candidates.columns:
        score_range = top_candidates[mean_score_col].max() - top_candidates[mean_score_col].min()
        if score_range < 0.005:
            print("  - Top candidates have very similar performance (< 0.5% difference)")
            print("    Consider selecting based on model simplicity or inference speed")

    print("\nIMPORTANT: Timing metrics are unreliable from parallel CV.")
    print("For production deployment decisions, conduct dedicated latency testing.")
    print("=" * 100)


def get_cv_statistics(
    cv_results_path: str,
    refit_metric: str = 'pr_auc'
) -> Dict[str, float]:
    """
    Get summary statistics from CV results file.

    Args:
        cv_results_path: Path to the CV results CSV file
        refit_metric: The metric used for refit in multi-metric scoring (default: 'pr_auc').
            Set to None for single-metric scoring (legacy).

    Returns:
        Dictionary with summary statistics

    Example:
        >>> stats = get_cv_statistics('models/logs/rf_cv_results.csv', refit_metric='pr_auc')
        >>> print(f"Best score: {stats['best_score']:.4f}")
    """
    cv_results = pd.read_csv(cv_results_path)

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
    best_model = cv_results.loc[best_idx]

    return {
        'best_score': best_model[mean_score_col],
        'best_std': best_model[std_score_col],
        'n_candidates': len(cv_results),
        'mean_score_all': cv_results[mean_score_col].mean(),
        'std_score_all': cv_results[mean_score_col].std(),
        'best_fit_time': best_model['mean_fit_time'],
        'best_score_time': best_model['mean_score_time']
    }


def analyze_cv_train_val_gap(
    cv_results_path: str,
    refit_metric: str = 'pr_auc',
    gap_threshold_warning: float = 0.05,
    gap_threshold_severe: float = 0.10,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (14, 5),
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze train-validation gap from GridSearchCV/RandomizedSearchCV results.

    This function examines the difference between training and validation scores
    to detect overfitting. Requires that the search object was created with
    `return_train_score=True`.

    Args:
        cv_results_path: Path to the CV results CSV file
        refit_metric: The metric used for refit (default: 'pr_auc')
        gap_threshold_warning: Gap percentage above which to show warning (default: 0.05 = 5%)
        gap_threshold_severe: Gap percentage above which to show severe warning (default: 0.10 = 10%)
        model_name: Name of the model for display purposes
        figsize: Figure size for visualization
        verbose: If True, display analysis and create visualizations

    Returns:
        Dictionary with:
        - best_train_score: Training score for best model
        - best_val_score: Validation score for best model
        - gap: Absolute gap (train - val)
        - gap_pct: Gap as percentage of training score
        - diagnosis: 'Good fit', 'MODERATE OVERFITTING', or 'SEVERE OVERFITTING'
        - overfitting_detected: Boolean flag
        - recommendation: Actionable recommendation string

    Example:
        >>> gap_analysis = analyze_cv_train_val_gap(
        ...     'models/logs/random_forest_cv_results.csv',
        ...     refit_metric='pr_auc',
        ...     model_name='Random Forest'
        ... )
        >>> if gap_analysis['overfitting_detected']:
        ...     print(gap_analysis['recommendation'])
    """
    # Load CV results
    cv_results = pd.read_csv(cv_results_path)

    # Determine column names based on scoring type
    if refit_metric and f'mean_test_{refit_metric}' in cv_results.columns:
        mean_train_col = f'mean_train_{refit_metric}'
        mean_val_col = f'mean_test_{refit_metric}'
        rank_col = f'rank_test_{refit_metric}'
    else:
        mean_train_col = 'mean_train_score'
        mean_val_col = 'mean_test_score'
        rank_col = 'rank_test_score'

    # Check if training scores are available
    if mean_train_col not in cv_results.columns:
        raise ValueError(
            f"Training scores not found in CV results. "
            f"Column '{mean_train_col}' not found. "
            f"Ensure GridSearchCV/RandomizedSearchCV was created with return_train_score=True."
        )

    # Get best model index
    best_idx = cv_results[rank_col].idxmin()
    best_model = cv_results.loc[best_idx]

    # Calculate gap metrics
    best_train_score = best_model[mean_train_col]
    best_val_score = best_model[mean_val_col]
    gap = best_train_score - best_val_score
    gap_pct = gap / best_train_score if best_train_score > 0 else 0

    # Determine diagnosis
    if gap_pct >= gap_threshold_severe:
        diagnosis = 'SEVERE OVERFITTING'
        overfitting_detected = True
    elif gap_pct >= gap_threshold_warning:
        diagnosis = 'MODERATE OVERFITTING'
        overfitting_detected = True
    else:
        diagnosis = 'Good fit'
        overfitting_detected = False

    # Generate recommendation
    recommendation = _generate_gap_recommendation(
        model_name, gap_pct, diagnosis, gap_threshold_warning, gap_threshold_severe
    )

    result = {
        'best_train_score': float(best_train_score),
        'best_val_score': float(best_val_score),
        'gap': float(gap),
        'gap_pct': float(gap_pct),
        'diagnosis': diagnosis,
        'overfitting_detected': overfitting_detected,
        'recommendation': recommendation,
        'model_name': model_name,
        'refit_metric': refit_metric
    }

    if verbose:
        _print_cv_gap_analysis(result, gap_threshold_warning, gap_threshold_severe)
        _plot_cv_gap_analysis(cv_results, best_idx, mean_train_col, mean_val_col,
                              rank_col, model_name, refit_metric, figsize)

    return result


def _generate_gap_recommendation(
    model_name: str,
    gap_pct: float,
    diagnosis: str,
    gap_threshold_warning: float,
    gap_threshold_severe: float
) -> str:
    """Generate actionable recommendation based on gap analysis."""
    if diagnosis == 'Good fit':
        return f"{model_name} shows healthy generalization with minimal overfitting."

    if 'Random Forest' in model_name or 'RF' in model_name:
        if diagnosis == 'SEVERE OVERFITTING':
            return (
                f"{model_name} shows severe overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Reduce model complexity:\n"
                "     - Decrease max_depth (e.g., 20 -> 15 or 10)\n"
                "     - Increase min_samples_leaf (e.g., 5 -> 10 or 20)\n"
                "  2. Increase regularization:\n"
                "     - Increase min_samples_split (e.g., 10 -> 20)\n"
                "  3. Consider using XGBoost instead (typically better regularization)"
            )
        else:
            return (
                f"{model_name} shows moderate overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Consider reducing max_depth slightly\n"
                "  2. Increase min_samples_leaf if possible"
            )
    elif 'XGBoost' in model_name or 'XGB' in model_name:
        if diagnosis == 'SEVERE OVERFITTING':
            return (
                f"{model_name} shows severe overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Increase regularization:\n"
                "     - Increase reg_alpha (L1) or reg_lambda (L2)\n"
                "     - Decrease learning_rate (and increase n_estimators)\n"
                "  2. Reduce model complexity:\n"
                "     - Decrease max_depth\n"
                "     - Increase min_child_weight\n"
                "  3. Use early stopping with more patience"
            )
        else:
            return (
                f"{model_name} shows moderate overfitting ({gap_pct:.1%} train-val gap).\n"
                "RECOMMENDED ACTIONS:\n"
                "  1. Consider increasing regularization slightly\n"
                "  2. Monitor validation loss during training"
            )
    else:
        return (
            f"{model_name} shows {diagnosis.lower()} ({gap_pct:.1%} train-val gap).\n"
            "Consider reducing model complexity or increasing regularization."
        )


def _print_cv_gap_analysis(
    result: Dict[str, Any],
    gap_threshold_warning: float,
    gap_threshold_severe: float
) -> None:
    """Print formatted train-validation gap analysis."""
    model_name = result['model_name']
    refit_metric = result['refit_metric']
    gap_pct = result['gap_pct']
    diagnosis = result['diagnosis']

    print("\n" + "=" * 80)

    if result['overfitting_detected']:
        print(f"!!! OVERFITTING WARNING - {model_name} !!!")
    else:
        print(f"{model_name} - Train-Validation Gap Analysis")

    print("=" * 80)
    print(f"\nMetric: {refit_metric}")
    print(f"  Training score:   {result['best_train_score']:.4f}")
    print(f"  Validation score: {result['best_val_score']:.4f}")
    print(f"  Gap:              {result['gap']:.4f} ({gap_pct:.1%})")
    print(f"\nDiagnosis: {diagnosis}")
    print(f"  (Warning threshold: {gap_threshold_warning:.0%}, Severe threshold: {gap_threshold_severe:.0%})")

    if result['overfitting_detected']:
        print("\n" + "-" * 80)
        print(result['recommendation'])
        print("-" * 80)

    print("=" * 80)


def _plot_cv_gap_analysis(
    cv_results: pd.DataFrame,
    best_idx: int,
    mean_train_col: str,
    mean_val_col: str,
    rank_col: str,
    model_name: str,
    refit_metric: str,
    figsize: Tuple[int, int]
) -> None:
    """Create visualization of train-validation gap across all candidates."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sort by validation score in descending order (best on left)
    # Keep original index to track best model position after sorting
    sorted_results = cv_results.sort_values(mean_val_col, ascending=False).reset_index()

    # Find position of best model in the sorted dataframe
    # The best model is the one with rank=1 (minimum rank)
    best_sorted_idx = sorted_results[rank_col].idxmin()

    x = np.arange(len(sorted_results))

    # Plot 1: Train vs Validation scores
    ax1 = axes[0]
    ax1.plot(x, sorted_results[mean_train_col], 'b-', label='Training', alpha=0.7, linewidth=2)
    ax1.plot(x, sorted_results[mean_val_col], 'g-', label='Validation', alpha=0.7, linewidth=2)
    ax1.fill_between(x, sorted_results[mean_val_col], sorted_results[mean_train_col],
                     alpha=0.3, color='red', label='Gap (Overfitting)')

    # Highlight best model
    ax1.axvline(x=best_sorted_idx, color='red', linestyle='--', linewidth=2,
                label=f'Best Model (Rank 1)', alpha=0.8)

    ax1.set_xlabel('Candidate (sorted by validation score, best on left)', fontsize=11)
    ax1.set_ylabel(f'{refit_metric}', fontsize=11)
    ax1.set_title(f'{model_name}: Training vs Validation Scores', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # Plot 2: Gap distribution
    ax2 = axes[1]
    gaps = sorted_results[mean_train_col] - sorted_results[mean_val_col]
    gap_pcts = gaps / sorted_results[mean_train_col] * 100

    colors = ['green' if g < 5 else 'orange' if g < 10 else 'red' for g in gap_pcts]
    ax2.bar(x, gap_pcts, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add threshold lines
    ax2.axhline(y=5, color='orange', linestyle='--', linewidth=2, label='Warning (5%)')
    ax2.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Severe (10%)')

    # Highlight best model
    ax2.bar(best_sorted_idx, gap_pcts.iloc[best_sorted_idx], color='purple',
            alpha=0.9, edgecolor='black', linewidth=2, label='Best Model')

    ax2.set_xlabel('Candidate (sorted by validation score, best on left)', fontsize=11)
    ax2.set_ylabel('Train-Val Gap (%)', fontsize=11)
    ax2.set_title(f'{model_name}: Overfitting Gap by Candidate', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
