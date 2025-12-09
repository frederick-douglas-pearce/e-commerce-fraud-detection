"""
Visualization utilities for fd3 notebook.

This module provides functions for creating model evaluation plots
including ROC/PR curves, feature importance, and threshold analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """Save figure to path if provided, creating directories as needed."""
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Figure saved: {save_path}")


def plot_roc_pr_curves(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    roc_auc: float,
    pr_auc: float,
    model_name: str = "XGBoost (Final)",
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None
) -> None:
    """
    Plot ROC and Precision-Recall curves for model evaluation.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        roc_auc: ROC-AUC score
        pr_auc: PR-AUC score
        model_name: Name of model for legend
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    ax = axes[0]
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'{model_name} - AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Test Set (Final Model)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 2: Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    baseline = y_true.mean()

    ax = axes[1]
    ax.plot(recall, precision, color='coral', lw=2, label=f'{model_name} - AUC = {pr_auc:.4f}')
    ax.axhline(y=baseline, color='gray', linestyle='--', lw=1,
               label=f'Baseline (No Skill) = {baseline:.4f}')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve - Test Set (Final Model)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print("✓ ROC and PR curves generated for test set (final retrained model)")


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot horizontal bar chart of feature importance.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    top_features = feature_importance_df.head(top_n)

    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features - XGBoost Built-in (Gain)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(row['importance'], i, f"  {row['importance']:.4f}", va='center', fontsize=9)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print(f"\n✓ XGBoost built-in feature importance analyzed ({top_n} features shown)")


def plot_shap_importance(
    shap_importance_df: pd.DataFrame,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot horizontal bar chart of SHAP-based feature importance.

    Shows mean |SHAP value| for each feature, with color indicating
    whether the feature increases (red) or decreases (blue) fraud risk on average.

    Args:
        shap_importance_df: DataFrame with 'feature', 'shap_importance', 'mean_shap' columns
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    top_features = shap_importance_df.head(top_n)

    # Color based on direction of effect
    colors = ['#d62728' if ms > 0 else '#1f77b4' for ms in top_features['mean_shap']]

    ax.barh(range(len(top_features)), top_features['shap_importance'], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Top {top_n} Most Important Features - SHAP Values',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels with direction indicator
    for i, (_, row) in enumerate(top_features.iterrows()):
        direction = "↑" if row['mean_shap'] > 0 else "↓"
        ax.text(row['shap_importance'], i,
                f"  {row['shap_importance']:.4f} {direction}",
                va='center', fontsize=9)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Increases fraud risk'),
        Patch(facecolor='#1f77b4', label='Decreases fraud risk')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print(f"\n✓ SHAP feature importance analyzed ({top_n} features shown)")


def plot_importance_comparison(
    comparison_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> None:
    """
    Plot side-by-side comparison of XGBoost Gain vs SHAP importance.

    Args:
        comparison_df: DataFrame from compare_importance_methods()
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    top_n = len(comparison_df)

    # Normalize for comparison (both sum to 1)
    gain_normalized = comparison_df['importance'] / comparison_df['importance'].sum()
    shap_normalized = comparison_df['shap_importance'] / comparison_df['shap_importance'].sum()

    # Left plot: Gain importance
    axes[0].barh(range(top_n), gain_normalized, color='steelblue')
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(comparison_df['feature'])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Normalized Importance', fontsize=11)
    axes[0].set_title('XGBoost Gain', fontsize=12, fontweight='bold')
    axes[0].grid(axis='x', alpha=0.3)

    # Right plot: SHAP importance
    colors = ['#d62728' if ms > 0 else '#1f77b4' for ms in comparison_df['mean_shap']]
    axes[1].barh(range(top_n), shap_normalized, color=colors)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(comparison_df['feature'])
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Normalized Importance', fontsize=11)
    axes[1].set_title('SHAP Values', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    fig.suptitle('Feature Importance: XGBoost Gain vs SHAP Comparison',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print(f"\n✓ Feature importance comparison plotted ({top_n} features)")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (14, 12),
    save_path: Optional[str] = None
) -> None:
    """
    Plot SHAP beeswarm showing distribution of SHAP values colored by feature value.

    This visualization shows how each feature affects predictions across all samples,
    with color indicating whether the feature value was high (red) or low (blue).
    This is the standard SHAP visualization that reveals non-linear relationships.

    Args:
        shap_values: SHAP values matrix (n_samples, n_features)
        X: Original feature DataFrame (for getting feature values)
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=figsize)

    # For each feature, create a scatter of SHAP values
    for i, feat_idx in enumerate(top_indices):
        feat_name = feature_names[feat_idx]
        shap_vals = shap_values[:, feat_idx]

        # Get feature values and normalize to 0-1 for coloring
        if feat_name in X.columns:
            feat_vals = X[feat_name].values
            # Handle categorical/string columns
            if feat_vals.dtype == object or feat_vals.dtype.name == 'category':
                # Convert to numeric codes
                unique_vals = np.unique(feat_vals)
                val_to_code = {v: i for i, v in enumerate(unique_vals)}
                feat_vals = np.array([val_to_code[v] for v in feat_vals], dtype=float)
        else:
            # Handle features that might have been transformed
            feat_vals = np.zeros(len(shap_vals))

        # Normalize feature values to 0-1 for color mapping
        feat_min, feat_max = float(feat_vals.min()), float(feat_vals.max())
        if feat_max > feat_min:
            feat_normalized = (feat_vals - feat_min) / (feat_max - feat_min)
        else:
            feat_normalized = np.zeros_like(feat_vals, dtype=float)

        # Sample points for visualization (too many points makes plot unreadable)
        n_samples = len(shap_vals)
        if n_samples > 1000:
            sample_idx = np.random.choice(n_samples, 1000, replace=False)
        else:
            sample_idx = np.arange(n_samples)

        # Add jitter to y-position
        y_jitter = np.random.normal(0, 0.15, len(sample_idx))

        # Plot scatter with color based on feature value
        scatter = ax.scatter(
            shap_vals[sample_idx],
            i + y_jitter,
            c=feat_normalized[sample_idx],
            cmap='RdBu_r',  # Red=high, Blue=low
            alpha=0.5,
            s=10,
            vmin=0,
            vmax=1
        )

    # Set y-axis labels
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[idx] for idx in top_indices])
    ax.invert_yaxis()

    # Add vertical line at 0
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=16)
    ax.set_title(f'SHAP Beeswarm Plot - Top {top_n} Features\n'
                 '(Red = high feature value, Blue = low feature value)',
                 fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(axis='x', alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Feature Value\n(normalized)', fontsize=14)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['Low', 'Mid', 'High'])
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print(f"\n✓ SHAP beeswarm plot generated ({top_n} features shown)")


def plot_threshold_optimization(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    threshold_results: List[Dict[str, Any]],
    baseline_rate: float,
    optimal_f1_result: Optional[Dict[str, Any]] = None,
    target_performance_result: Optional[Dict[str, Any]] = None,
    figsize: Tuple[int, int] = (20, 16),
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive threshold optimization visualization (2x2 grid).

    Args:
        precisions: Precision values from precision_recall_curve
        recalls: Recall values from precision_recall_curve
        thresholds: Threshold values from precision_recall_curve
        threshold_results: List of dicts with recall-targeted threshold results
        baseline_rate: Baseline fraud rate (for no-skill line)
        optimal_f1_result: Optional dict with optimal F1 threshold results
        target_performance_result: Optional dict with target performance threshold results
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Build combined list of all threshold results for bottom panels
    all_results = []
    if optimal_f1_result is not None:
        all_results.append({
            'name': 'Optimal F1',
            'threshold': optimal_f1_result['threshold'],
            'precision': optimal_f1_result['precision'],
            'recall': optimal_f1_result['recall'],
            'f1': optimal_f1_result['f1'],
            'fp': optimal_f1_result.get('fp', 0),
            'fn': optimal_f1_result.get('fn', 0),
            'color': 'purple'
        })
    if target_performance_result is not None:
        all_results.append({
            'name': 'Target Perf',
            'threshold': target_performance_result['threshold'],
            'precision': target_performance_result['precision'],
            'recall': target_performance_result['recall'],
            'f1': target_performance_result['f1'],
            'fp': target_performance_result.get('fp', 0),
            'fn': target_performance_result.get('fn', 0),
            'color': 'blue'
        })
    # Add recall-targeted thresholds
    recall_colors = ['red', 'orange', 'green']
    for i, result in enumerate(threshold_results):
        all_results.append({
            'name': f"{result['target_recall']*100:.0f}% recall",
            'threshold': result['threshold'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'fp': result.get('fp', 0),
            'fn': result.get('fn', 0),
            'color': recall_colors[i]
        })

    # Markers for recall-targeted thresholds
    recall_markers = ['*', 's', 'D']

    # Plot 1: Precision-Recall Curve with marked thresholds
    ax = axes[0, 0]
    ax.plot(recalls, precisions, color='steelblue', lw=2.5, label='PR Curve')
    ax.axhline(y=baseline_rate, color='gray', linestyle='--', lw=1.5,
               label=f'No Skill = {baseline_rate:.3f}')

    # Mark optimal F1 threshold (if provided)
    if optimal_f1_result is not None:
        ax.scatter(optimal_f1_result['recall'], optimal_f1_result['precision'],
                   c='purple', s=400, marker='P',
                   edgecolors='black', linewidths=2,
                   label=f"Optimal F1: θ={optimal_f1_result['threshold']:.3f}",
                   zorder=11)

    # Mark target performance threshold (if provided)
    if target_performance_result is not None:
        ax.scatter(target_performance_result['recall'], target_performance_result['precision'],
                   c='blue', s=350, marker='X',
                   edgecolors='black', linewidths=2,
                   label=f"Target Perf: θ={target_performance_result['threshold']:.3f}",
                   zorder=11)

    # Mark recall-targeted thresholds
    for i, result in enumerate(threshold_results):
        ax.scatter(result['recall'], result['precision'],
                   c=recall_colors[i], s=300, marker=recall_markers[i],
                   edgecolors='black', linewidths=2,
                   label=f"{result['target_recall']*100:.0f}% recall: θ={result['threshold']:.3f}",
                   zorder=10)

    ax.set_xlabel('Recall', fontsize=16)
    ax.set_ylabel('Precision', fontsize=16)
    ax.set_title('Precision-Recall Curve with Optimal Thresholds', fontsize=18, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10, markerscale=0.6)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 2: Precision/Recall/F1 vs Threshold
    ax = axes[0, 1]

    # Sample thresholds for clarity
    step = max(1, len(thresholds) // 1000)
    ax.plot(thresholds[::step], precisions[:-1][::step], 'steelblue', lw=2.5, label='Precision')
    ax.plot(thresholds[::step], recalls[:-1][::step], 'coral', lw=2.5, label='Recall')

    # Calculate F1 for all thresholds
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    ax.plot(thresholds[::step], f1_scores[::step], 'lightgreen', lw=2.5, label='F1 Score')

    # Mark optimal F1 threshold (if provided)
    if optimal_f1_result is not None:
        ax.axvline(x=optimal_f1_result['threshold'], color='purple', linestyle='--', lw=2,
                   label=f"θ={optimal_f1_result['threshold']:.3f} (Optimal F1)",
                   alpha=0.8)

    # Mark target performance threshold (if provided)
    if target_performance_result is not None:
        ax.axvline(x=target_performance_result['threshold'], color='blue', linestyle='--', lw=2,
                   label=f"θ={target_performance_result['threshold']:.3f} (Target Perf)",
                   alpha=0.8)

    # Mark recall-targeted thresholds
    for i, result in enumerate(threshold_results):
        ax.axvline(x=result['threshold'], color=recall_colors[i], linestyle='--', lw=2,
                   label=f"θ={result['threshold']:.3f} ({result['target_recall']*100:.0f}% recall)",
                   alpha=0.7)

    ax.set_xlabel('Threshold', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_title('Precision/Recall/F1 vs Threshold', fontsize=18, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 3: False Positives vs False Negatives (all thresholds)
    ax = axes[1, 0]

    x = np.arange(len(all_results))
    width = 0.35

    fp_counts = [r['fp'] for r in all_results]
    fn_counts = [r['fn'] for r in all_results]
    bar_colors = [r['color'] for r in all_results]

    bars1 = ax.bar(x - width/2, fp_counts, width, label='False Positives (FP)',
                   color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_counts, width, label='False Negatives (FN)',
                   color='steelblue', alpha=0.8)

    ax.set_xlabel('Threshold Scenario', fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title('False Positives vs False Negatives by Threshold', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([r['name'] for r in all_results], fontsize=12, rotation=15, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Plot 4: Metrics Comparison (all thresholds)
    ax = axes[1, 1]

    metrics_data = pd.DataFrame([
        {
            'Scenario': r['name'],
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1']
        }
        for r in all_results
    ])

    x = np.arange(len(metrics_data))
    width = 0.25

    bars1 = ax.bar(x - width, metrics_data['Precision'], width, label='Precision',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, metrics_data['Recall'], width, label='Recall',
                   color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, metrics_data['F1'], width, label='F1 Score',
                   color='lightgreen', alpha=0.8)

    ax.set_xlabel('Threshold Scenario', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_title('Performance Metrics Comparison', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data['Scenario'], fontsize=12, rotation=15, ha='right')
    ax.tick_params(axis='y', labelsize=14)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print("✓ Threshold optimization visualizations complete")
