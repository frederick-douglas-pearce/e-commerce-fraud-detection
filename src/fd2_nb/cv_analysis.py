"""
Cross-validation results analysis utilities.

This module provides functions for analyzing and visualizing cross-validation
results from hyperparameter tuning, with focus on production deployment criteria.
All functions are designed to be reusable across different models and projects.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def analyze_cv_results(
    cv_results_path: str,
    top_n: int = 5,
    model_name: str = "Model",
    figsize: Tuple[int, int] = (16, 12),
    stability_threshold: float = 0.01,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze cross-validation results with focus on production deployment criteria.

    This function examines:
    - Model stability (std_test_score) - RELIABLE: consistency across CV folds
    - Performance (mean_test_score) - RELIABLE: metric for model selection
    - Prediction time (mean_score_time) - UNRELIABLE: affected by parallel processing
    - Training time (mean_fit_time) - UNRELIABLE: includes CV overhead

    Note:
        Timing measurements from parallel CV are unreliable - use as rough indicators only!
        Production API latency testing will provide definitive performance numbers.

    Args:
        cv_results_path: Path to the CV results CSV file
        top_n: Number of top candidates to analyze in detail (default: 5)
        model_name: Name of the model for display purposes
        figsize: Figure size for the 4-panel visualization
        stability_threshold: Threshold for std_test_score below which model is considered stable
        verbose: If True, display analysis and create visualizations

    Returns:
        DataFrame with top N candidates and detailed metrics

    Example:
        >>> top_candidates = analyze_cv_results(
        ...     'models/logs/random_forest_cv_results_20241201_120000.csv',
        ...     top_n=5,
        ...     model_name='Random Forest'
        ... )
        >>> print(top_candidates['mean_test_score'].iloc[0])  # Best score
    """
    # Load CV results
    cv_results = pd.read_csv(cv_results_path)

    # Extract key columns
    key_cols = ['mean_test_score', 'std_test_score', 'mean_fit_time',
                'std_fit_time', 'mean_score_time', 'std_score_time', 'rank_test_score']

    # Add parameter columns
    param_cols = [col for col in cv_results.columns if col.startswith('param_')]
    display_cols = key_cols + param_cols

    # Get top N candidates by test score
    top_candidates = cv_results.nlargest(top_n, 'mean_test_score')[display_cols].copy()

    # Get best model
    best_idx = cv_results['rank_test_score'].idxmin()
    best_model = cv_results.loc[best_idx]

    if verbose:
        _print_analysis_header(model_name)
        _display_top_candidates(top_candidates, top_n)
        _display_statistical_summary(cv_results, key_cols)
        _display_best_model_details(best_model, stability_threshold)
        _create_analysis_plots(cv_results, best_model, top_candidates, top_n, figsize)
        _print_recommendations(best_model, top_candidates, stability_threshold)

    return top_candidates


def _print_analysis_header(model_name: str) -> None:
    """Print analysis header with timing caveat."""
    print("\n" + "=" * 100)
    print(f"{model_name} - Cross-Validation Results Analysis")
    print("=" * 100)
    print("TIMING CAVEAT: Due to parallel processing (n_jobs=-1), timing measurements may be")
    print("   unreliable. Small differences (< 20-30%) are often just measurement noise.")
    print("   Focus on PR-AUC and stability for model selection. Production API testing will")
    print("   provide definitive latency numbers.")
    print("=" * 100)


def _display_top_candidates(top_candidates: pd.DataFrame, top_n: int) -> None:
    """Display top N candidates with styled formatting."""
    print(f"\nTop {top_n} Candidates by Test Score:")
    print("-" * 100)

    # Format dictionary for display
    format_dict = {
        'mean_test_score': '{:.6f}',
        'std_test_score': '{:.6f}',
        'mean_fit_time': '{:.2f}',
        'std_fit_time': '{:.2f}',
        'mean_score_time': '{:.4f}',
        'std_score_time': '{:.4f}'
    }

    try:
        from IPython.display import display
        styled = top_candidates.style.format(format_dict)
        styled = styled.background_gradient(cmap='RdYlGn', subset=['mean_test_score'])
        display(styled)
    except ImportError:
        print(top_candidates.to_string())


def _display_statistical_summary(cv_results: pd.DataFrame, key_cols: List[str]) -> None:
    """Display statistical summary across all candidates."""
    print("\n" + "-" * 100)
    print("Statistical Summary Across All Candidates:")
    print("-" * 100)

    # Exclude rank column from statistics
    stat_cols = [c for c in key_cols if c != 'rank_test_score']
    summary_stats = cv_results[stat_cols].describe().loc[['mean', 'std', 'min', 'max']]

    try:
        from IPython.display import display
        display(summary_stats.style.format('{:.6f}'))
    except ImportError:
        print(summary_stats.to_string())


def _display_best_model_details(
    best_model: pd.Series,
    stability_threshold: float = 0.01
) -> None:
    """Display detailed metrics for best model."""
    print("\n" + "-" * 100)
    print("Best Model (Rank 1) - Detailed Metrics:")
    print("-" * 100)

    stability_status = "Stable" if best_model['std_test_score'] < stability_threshold else "Variable"

    print(f"  - Test Score (mean +/- std):    {best_model['mean_test_score']:.6f} +/- {best_model['std_test_score']:.6f}")
    print(f"  - Stability (CV std):           {best_model['std_test_score']:.6f} ({stability_status})")
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
    figsize: Tuple[int, int]
) -> None:
    """Create 4-panel analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    _plot_performance_vs_stability(cv_results, best_model, axes[0, 0])
    _plot_performance_vs_time(cv_results, best_model, axes[0, 1])
    _plot_top_candidates_comparison(cv_results, top_candidates, top_n, axes[1, 0])
    _plot_training_vs_prediction_time(cv_results, best_model, axes[1, 1])

    plt.tight_layout()
    plt.show()


def _plot_performance_vs_stability(
    cv_results: pd.DataFrame,
    best_model: pd.Series,
    ax: plt.Axes
) -> None:
    """Plot 1: Performance vs Stability scatter plot."""
    scatter = ax.scatter(
        cv_results['mean_test_score'],
        cv_results['std_test_score'],
        c=cv_results['rank_test_score'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )

    # Highlight best model
    ax.scatter(
        best_model['mean_test_score'],
        best_model['std_test_score'],
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
    ax: plt.Axes
) -> None:
    """Plot 2: Performance vs Prediction Time scatter plot."""
    scatter = ax.scatter(
        cv_results['mean_test_score'],
        cv_results['mean_score_time'],
        c=cv_results['rank_test_score'],
        cmap='RdYlGn_r',
        s=100,
        alpha=0.6,
        edgecolors='black'
    )

    # Highlight best model
    ax.scatter(
        best_model['mean_test_score'],
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
    ax: plt.Axes
) -> None:
    """Plot 3: Top N Candidates normalized comparison."""
    x = np.arange(len(top_candidates))
    width = 0.35

    # Normalize scores for comparison
    score_min = cv_results['mean_test_score'].min()
    score_max = cv_results['mean_test_score'].max()
    time_min = cv_results['mean_score_time'].min()
    time_max = cv_results['mean_score_time'].max()

    score_normalized = (top_candidates['mean_test_score'] - score_min) / (score_max - score_min + 1e-10)
    time_normalized = 1 - (top_candidates['mean_score_time'] - time_min) / (time_max - time_min + 1e-10)

    ax.bar(x - width/2, score_normalized, width, label='Performance (Reliable)', color='steelblue')
    ax.bar(x + width/2, time_normalized, width, label='Speed (Unreliable)', color='coral')

    ax.set_xlabel('Candidate Rank', fontsize=12)
    ax.set_ylabel('Normalized Score', fontsize=12)
    ax.set_title(f'Top {top_n} Candidates: Performance vs Speed', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Rank {int(r)}" for r in top_candidates['rank_test_score']])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])


def _plot_training_vs_prediction_time(
    cv_results: pd.DataFrame,
    best_model: pd.Series,
    ax: plt.Axes
) -> None:
    """Plot 4: Training Time vs Prediction Time scatter plot."""
    scatter = ax.scatter(
        cv_results['mean_fit_time'],
        cv_results['mean_score_time'],
        c=cv_results['mean_test_score'],
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
    stability_threshold: float = 0.01
) -> None:
    """Print model selection recommendations."""
    print("\n" + "=" * 100)
    print("Recommendations:")
    print("=" * 100)

    # Check stability
    if best_model['std_test_score'] < stability_threshold:
        print("  - Best model shows good stability (low CV variance)")
    else:
        print(f"  - Best model shows some variance across folds (std > {stability_threshold})")
        print("    Consider if a more stable alternative in top candidates might be preferred")

    # Check if top candidates are close in performance
    if len(top_candidates) > 1:
        score_range = top_candidates['mean_test_score'].max() - top_candidates['mean_test_score'].min()
        if score_range < 0.005:
            print("  - Top candidates have very similar performance (< 0.5% difference)")
            print("    Consider selecting based on model simplicity or inference speed")

    print("\nIMPORTANT: Timing metrics are unreliable from parallel CV.")
    print("For production deployment decisions, conduct dedicated latency testing.")
    print("=" * 100)


def get_cv_statistics(cv_results_path: str) -> Dict[str, float]:
    """
    Get summary statistics from CV results file.

    Args:
        cv_results_path: Path to the CV results CSV file

    Returns:
        Dictionary with summary statistics

    Example:
        >>> stats = get_cv_statistics('models/logs/rf_cv_results.csv')
        >>> print(f"Best score: {stats['best_score']:.4f}")
    """
    cv_results = pd.read_csv(cv_results_path)

    best_idx = cv_results['rank_test_score'].idxmin()
    best_model = cv_results.loc[best_idx]

    return {
        'best_score': best_model['mean_test_score'],
        'best_std': best_model['std_test_score'],
        'n_candidates': len(cv_results),
        'mean_score_all': cv_results['mean_test_score'].mean(),
        'std_score_all': cv_results['mean_test_score'].std(),
        'best_fit_time': best_model['mean_fit_time'],
        'best_score_time': best_model['mean_score_time']
    }
