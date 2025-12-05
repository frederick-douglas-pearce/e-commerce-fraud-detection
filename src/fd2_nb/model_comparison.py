"""
Model comparison utilities for evaluating and comparing multiple models.

This module provides general-purpose functions for comparing classification models
on key metrics, creating comparison tables, and visualizing performance differences.
All functions are designed to be reusable across different datasets and projects.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compare_models(
    model_metrics: List[Dict[str, float]],
    metrics_to_display: Optional[List[str]] = None,
    metrics_to_highlight: Optional[List[str]] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (16, 6),
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple models on key metrics with visualization.

    Creates a comparison DataFrame from model metrics dictionaries and optionally
    displays styled tables and bar chart visualizations.

    Args:
        model_metrics: List of dictionaries containing model metrics.
            Each dict should have 'model' key and metric keys like 'pr_auc', 'roc_auc', etc.
        metrics_to_display: List of metrics to include in comparison (default: all except 'model', 'dataset')
        metrics_to_highlight: List of metrics to apply gradient styling (default: ['roc_auc', 'pr_auc', 'f1'])
        title: Title for the comparison display
        figsize: Figure size for visualization plots
        verbose: If True, display styled table and create visualization

    Returns:
        pd.DataFrame: Comparison DataFrame with model names as index

    Example:
        >>> lr_metrics = {'model': 'Logistic Regression', 'pr_auc': 0.75, 'roc_auc': 0.85}
        >>> rf_metrics = {'model': 'Random Forest', 'pr_auc': 0.82, 'roc_auc': 0.90}
        >>> comparison = compare_models([lr_metrics, rf_metrics])
        >>> print(comparison)
                             pr_auc  roc_auc
        model
        Logistic Regression    0.75     0.85
        Random Forest          0.82     0.90
    """
    # Create DataFrame
    comparison_df = pd.DataFrame(model_metrics)

    # Set model as index
    if 'model' in comparison_df.columns:
        comparison_df = comparison_df.set_index('model')

    # Drop dataset column if present
    if 'dataset' in comparison_df.columns:
        comparison_df = comparison_df.drop(columns=['dataset'])

    # Filter to requested metrics
    if metrics_to_display is not None:
        available_metrics = [m for m in metrics_to_display if m in comparison_df.columns]
        comparison_df = comparison_df[available_metrics]

    if verbose:
        _display_comparison_table(comparison_df, metrics_to_highlight, title)
        _plot_comparison_charts(comparison_df, figsize)

    return comparison_df


def _display_comparison_table(
    comparison_df: pd.DataFrame,
    metrics_to_highlight: Optional[List[str]] = None,
    title: str = "Model Comparison"
) -> None:
    """
    Display styled comparison table with gradient highlighting.

    Args:
        comparison_df: DataFrame with model comparison metrics
        metrics_to_highlight: Metrics to apply gradient styling
        title: Title for the display
    """
    if metrics_to_highlight is None:
        metrics_to_highlight = ['roc_auc', 'pr_auc', 'f1']

    # Filter to available metrics for highlighting
    highlight_cols = [m for m in metrics_to_highlight if m in comparison_df.columns]

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    # Format all numeric columns to 4 decimal places
    format_dict = {col: '{:.4f}' for col in comparison_df.columns}

    # Apply styling
    styled = comparison_df.style.format(format_dict)
    if highlight_cols:
        styled = styled.background_gradient(cmap='RdYlGn', subset=highlight_cols)

    # Use IPython display if available, otherwise print
    try:
        from IPython.display import display
        display(styled)
    except ImportError:
        print(comparison_df.to_string())

    # Show best model for each metric
    print("\n" + "=" * 80)
    print("Best Performing Model by Metric:")
    print("=" * 80)
    for metric in comparison_df.columns:
        best_model = comparison_df[metric].idxmax()
        best_value = comparison_df[metric].max()
        print(f"  {metric.upper():15s}: {best_model:25s} ({best_value:.4f})")
    print("=" * 80)


def _plot_comparison_charts(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Create comparison bar charts for model metrics.

    Args:
        comparison_df: DataFrame with model comparison metrics
        figsize: Figure size for the plots
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Key metrics comparison (ROC-AUC, PR-AUC, F1)
    ax = axes[0]
    key_metrics = ['roc_auc', 'pr_auc', 'f1']
    available_key_metrics = [m for m in key_metrics if m in comparison_df.columns]

    if available_key_metrics:
        comparison_df[available_key_metrics].plot(
            kind='bar',
            ax=ax,
            color=['steelblue', 'coral', 'lightgreen'][:len(available_key_metrics)]
        )
        ax.set_title('Key Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        ax.legend([m.upper().replace('_', '-') for m in available_key_metrics], loc='lower right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

    # Plot 2: Precision vs Recall tradeoff
    ax = axes[1]
    if 'precision' in comparison_df.columns and 'recall' in comparison_df.columns:
        x = np.arange(len(comparison_df))
        width = 0.35

        bars1 = ax.bar(x - width/2, comparison_df['precision'], width,
                       label='Precision', color='steelblue')
        bars2 = ax.bar(x + width/2, comparison_df['recall'], width,
                       label='Recall', color='coral')

        ax.set_title('Precision vs Recall Tradeoff', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def get_best_model(
    comparison_df: pd.DataFrame,
    primary_metric: str = 'pr_auc'
) -> Tuple[str, Dict[str, float]]:
    """
    Get the best model based on a primary metric.

    Args:
        comparison_df: DataFrame with model comparison metrics (model names as index)
        primary_metric: Metric to use for selecting best model (default: 'pr_auc')

    Returns:
        Tuple of (best_model_name, metrics_dict)

    Example:
        >>> best_name, best_metrics = get_best_model(comparison_df, 'pr_auc')
        >>> print(f"Best model: {best_name} with PR-AUC: {best_metrics['pr_auc']:.4f}")
    """
    if primary_metric not in comparison_df.columns:
        raise ValueError(f"Metric '{primary_metric}' not found in comparison DataFrame. "
                        f"Available metrics: {list(comparison_df.columns)}")

    best_model_name = comparison_df[primary_metric].idxmax()
    best_metrics = comparison_df.loc[best_model_name].to_dict()

    return best_model_name, best_metrics
