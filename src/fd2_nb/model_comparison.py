"""
Model comparison utilities for evaluating and comparing multiple models.

This module provides general-purpose functions for comparing classification models
on key metrics, creating comparison tables, and visualizing performance differences.
All functions are designed to be reusable across different datasets and projects.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided, creating directories as needed."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Figure saved: {save_path}")


def compare_models(
    model_metrics: List[Dict[str, float]],
    metrics_to_display: Optional[List[str]] = None,
    metrics_to_highlight: Optional[List[str]] = None,
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (16, 6),
    verbose: bool = True,
    save_path: Optional[str] = None
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
        save_path: Optional path to save the figure (e.g., 'images/fd2/comparison.png')

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
        _plot_comparison_charts(comparison_df, figsize, save_path)

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
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Create comparison bar charts for model metrics.

    Args:
        comparison_df: DataFrame with model comparison metrics
        figsize: Figure size for the plots
        save_path: Optional path to save the figure
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
    _save_figure(fig, save_path)
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    plot_configs: Optional[List[Dict]] = None,
    figsize: Tuple[int, int] = (16, 6),
    suptitle: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create customizable comparison visualizations for model metrics.

    Generates a multi-panel figure with configurable subplots. Each subplot can be
    a grouped bar chart (for comparing two metrics side-by-side) or a simple bar chart
    (for a single metric or multiple metrics stacked).

    Args:
        comparison_df: DataFrame with model comparison metrics (model names as index)
        plot_configs: List of plot configuration dictionaries. Each dict can have:
            - 'metrics': List of metric column names to plot (required)
            - 'title': Subplot title (default: 'Metrics Comparison')
            - 'labels': Display labels for metrics (default: uppercase metric names)
            - 'colors': Colors for bars (default: ['steelblue', 'coral', 'lightgreen'])
            - 'plot_type': 'grouped' for side-by-side bars, 'stacked' for grouped bars (default: 'stacked')
            - 'show_values': Whether to show value labels on bars (default: True for grouped, False for stacked)
            If None, uses default 2-panel layout: key metrics + precision vs recall
        figsize: Figure size (width, height)
        suptitle: Optional super title for the entire figure
        save_path: Optional path to save the figure (e.g., 'images/fd2/comparison.png')

    Example:
        >>> plot_configs = [
        ...     {
        ...         'metrics': ['roc_auc', 'pr_auc', 'f1'],
        ...         'title': 'Key Metrics',
        ...         'labels': ['ROC-AUC', 'PR-AUC', 'F1'],
        ...         'colors': ['steelblue', 'coral', 'lightgreen']
        ...     },
        ...     {
        ...         'metrics': ['precision', 'recall'],
        ...         'title': 'Precision vs Recall',
        ...         'plot_type': 'grouped',
        ...         'show_values': True
        ...     }
        ... ]
        >>> plot_model_comparison(comparison_df, plot_configs)
    """
    # Default configuration if none provided
    if plot_configs is None:
        plot_configs = [
            {
                'metrics': ['roc_auc', 'pr_auc', 'f1'],
                'title': 'Key Metrics Comparison',
                'labels': ['ROC-AUC', 'PR-AUC', 'F1 Score'],
                'colors': ['steelblue', 'coral', 'lightgreen'],
                'plot_type': 'stacked'
            },
            {
                'metrics': ['precision', 'recall'],
                'title': 'Precision vs Recall Tradeoff',
                'labels': ['Precision', 'Recall'],
                'colors': ['steelblue', 'coral'],
                'plot_type': 'grouped',
                'show_values': True
            }
        ]

    n_plots = len(plot_configs)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    # Handle single plot case
    if n_plots == 1:
        axes = [axes]

    for ax, config in zip(axes, plot_configs):
        metrics = config['metrics']
        title = config.get('title', 'Metrics Comparison')
        labels = config.get('labels', [m.upper().replace('_', '-') for m in metrics])
        colors = config.get('colors', ['steelblue', 'coral', 'lightgreen', 'gold'][:len(metrics)])
        plot_type = config.get('plot_type', 'stacked')
        show_values = config.get('show_values', plot_type == 'grouped')

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        available_labels = [labels[i] for i, m in enumerate(metrics) if m in comparison_df.columns]
        available_colors = [colors[i] for i, m in enumerate(metrics) if m in comparison_df.columns]

        if not available_metrics:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        if plot_type == 'grouped':
            # Side-by-side grouped bars
            x = np.arange(len(comparison_df))
            width = 0.8 / len(available_metrics)

            bars_list = []
            for i, (metric, label, color) in enumerate(zip(available_metrics, available_labels, available_colors)):
                offset = (i - len(available_metrics) / 2 + 0.5) * width
                bars = ax.bar(x + offset, comparison_df[metric], width, label=label, color=color)
                bars_list.append(bars)

            ax.set_xticks(x)
            ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')

            # Add value labels if requested
            if show_values:
                for bars in bars_list:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{height:.3f}',
                                ha='center', va='bottom', fontsize=9)

        else:
            # Stacked/grouped bar chart (default matplotlib style)
            comparison_df[available_metrics].plot(
                kind='bar',
                ax=ax,
                color=available_colors
            )
            ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
            ax.legend(available_labels, loc='lower right')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])

        if plot_type == 'grouped':
            ax.legend()

    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def plot_comprehensive_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (18, 12),
    metrics: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create a 2x2 grid of horizontal bar charts for comprehensive model comparison.

    Displays four metrics (default: PR-AUC, F1, Precision, Recall) as horizontal
    bar charts with value labels, designed for easy visual comparison across models.

    Args:
        comparison_df: DataFrame with model names as index and metrics as columns
        figsize: Figure size (width, height)
        metrics: List of 4 metric column names to plot. Default: ['pr_auc', 'f1', 'precision', 'recall']
        colors: List of 4 colors for each subplot. Default: ['coral', 'lightgreen', 'steelblue', 'gold']
        titles: List of 4 subplot titles. Default: metric-specific titles
        save_path: Optional path to save the figure (e.g., 'images/fd2/comprehensive.png')

    Example:
        >>> plot_comprehensive_comparison(all_models_comparison)
        >>> # Or with custom metrics:
        >>> plot_comprehensive_comparison(df, metrics=['roc_auc', 'pr_auc', 'f1', 'accuracy'])
    """
    # Defaults
    if metrics is None:
        metrics = ['pr_auc', 'f1', 'precision', 'recall']

    if colors is None:
        colors = ['coral', 'lightgreen', 'steelblue', 'gold']

    if titles is None:
        titles = [
            'PR-AUC Comparison (Primary Metric)',
            'F1 Score Comparison',
            'Precision Comparison (Minimize False Positives)',
            'Recall Comparison (Catch More Fraud)'
        ]

    # Validate we have exactly 4 metrics for 2x2 grid
    if len(metrics) != 4:
        raise ValueError(f"Expected 4 metrics for 2x2 grid, got {len(metrics)}")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes_flat = axes.flatten()

    for ax, metric, color, title in zip(axes_flat, metrics, colors, titles):
        if metric not in comparison_df.columns:
            ax.text(0.5, 0.5, f'Metric "{metric}" not available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue

        comparison_df[metric].plot(kind='barh', ax=ax, color=color)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Model', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim([0, 1])

        # Add value labels
        for i, v in enumerate(comparison_df[metric]):
            ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    _save_figure(fig, save_path)
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
