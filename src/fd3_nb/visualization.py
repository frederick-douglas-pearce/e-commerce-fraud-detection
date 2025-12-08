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


def plot_threshold_optimization(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray,
    threshold_results: List[Dict[str, Any]],
    baseline_rate: float,
    figsize: Tuple[int, int] = (18, 14),
    save_path: Optional[str] = None
) -> None:
    """
    Create comprehensive threshold optimization visualization (2x2 grid).

    Args:
        precisions: Precision values from precision_recall_curve
        recalls: Recall values from precision_recall_curve
        thresholds: Threshold values from precision_recall_curve
        threshold_results: List of dicts with threshold optimization results
        baseline_rate: Baseline fraud rate (for no-skill line)
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    colors = ['red', 'orange', 'green']
    markers = ['*', 's', 'D']

    # Plot 1: Precision-Recall Curve with marked thresholds
    ax = axes[0, 0]
    ax.plot(recalls, precisions, color='steelblue', lw=2, label='PR Curve')
    ax.axhline(y=baseline_rate, color='gray', linestyle='--', lw=1,
               label=f'No Skill = {baseline_rate:.3f}')

    # Mark optimal thresholds
    for i, result in enumerate(threshold_results):
        ax.scatter(result['recall'], result['precision'],
                   c=colors[i], s=300, marker=markers[i],
                   edgecolors='black', linewidths=2,
                   label=f"{result['target_recall']*100:.0f}% recall: θ={result['threshold']:.3f}",
                   zorder=10)

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve with Optimal Thresholds', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 2: Precision/Recall/F1 vs Threshold
    ax = axes[0, 1]

    # Sample thresholds for clarity
    step = max(1, len(thresholds) // 1000)
    ax.plot(thresholds[::step], precisions[:-1][::step], 'steelblue', lw=2, label='Precision')
    ax.plot(thresholds[::step], recalls[:-1][::step], 'coral', lw=2, label='Recall')

    # Calculate F1 for all thresholds
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    ax.plot(thresholds[::step], f1_scores[::step], 'lightgreen', lw=2, label='F1 Score')

    # Mark optimal thresholds
    for i, result in enumerate(threshold_results):
        ax.axvline(x=result['threshold'], color=colors[i], linestyle='--', lw=1.5,
                   label=f"θ={result['threshold']:.3f} ({result['target_recall']*100:.0f}% recall)",
                   alpha=0.7)

    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision/Recall/F1 vs Threshold', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Plot 3: False Positives vs False Negatives
    ax = axes[1, 0]

    x = np.arange(len(threshold_results))
    width = 0.35

    fp_counts = [r['fp'] for r in threshold_results]
    fn_counts = [r['fn'] for r in threshold_results]

    bars1 = ax.bar(x - width/2, fp_counts, width, label='False Positives (FP)',
                   color='coral', alpha=0.8)
    bars2 = ax.bar(x + width/2, fn_counts, width, label='False Negatives (FN)',
                   color='steelblue', alpha=0.8)

    ax.set_xlabel('Threshold Scenario', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('False Positives vs False Negatives by Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['target_recall']*100:.0f}% recall" for r in threshold_results])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 4: Metrics Comparison
    ax = axes[1, 1]

    metrics_data = pd.DataFrame([
        {
            'Scenario': f"{r['target_recall']*100:.0f}% recall",
            'Precision': r['precision'],
            'Recall': r['recall'],
            'F1': r['f1']
        }
        for r in threshold_results
    ])

    x = np.arange(len(metrics_data))
    width = 0.25

    bars1 = ax.bar(x - width, metrics_data['Precision'], width, label='Precision',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x, metrics_data['Recall'], width, label='Recall',
                   color='coral', alpha=0.8)
    bars3 = ax.bar(x + width, metrics_data['F1'], width, label='F1 Score',
                   color='lightgreen', alpha=0.8)

    ax.set_xlabel('Threshold Scenario', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_data['Scenario'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print("✓ Threshold optimization visualizations complete")
