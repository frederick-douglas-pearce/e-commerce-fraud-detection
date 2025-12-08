"""
Model evaluation utilities for fd3 notebook.

This module provides functions for evaluating fraud detection models
and comparing validation vs test set performance.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Model",
    dataset_name: str = "Validation",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate classification model with fraud-appropriate metrics.

    Args:
        model: Trained model with predict and predict_proba methods
        X: Feature DataFrame
        y: Target Series
        model_name: Name of the model for display
        dataset_name: Name of the dataset (e.g., "Validation", "Test")
        verbose: If True, print results

    Returns:
        Dictionary with all metrics including model name and dataset
    """
    # Get predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'pr_auc': average_precision_score(y, y_pred_proba),
        'f1': f1_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'accuracy': (y_pred == y).mean()
    }

    if verbose:
        # Print results
        print(f"\n{model_name} - {dataset_name} Set Performance:")
        print("=" * 60)
        print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
        print(f"  F1 Score:   {metrics['f1']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print("=" * 60)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:,}  |  FP: {cm[0, 1]:,}")
        print(f"  FN: {cm[1, 0]:,}  |  TP: {cm[1, 1]:,}")

    return metrics


def compare_val_test_performance(
    validation_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    val_key: str = 'xgboost_tuned',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare validation vs test set performance and assess generalization.

    Args:
        validation_metrics: Dictionary with validation metrics (from CV)
        test_metrics: Dictionary with test set metrics
        val_key: Key in validation_metrics containing the model metrics
        verbose: If True, print detailed comparison

    Returns:
        DataFrame with comparison metrics
    """
    if verbose:
        print("\n" + "=" * 100)
        print("VALIDATION VS TEST SET PERFORMANCE")
        print("=" * 100)
        print("Note: Validation = CV metrics from Notebook 2 (model tuning)")
        print("      Test = Performance on held-out test set (this notebook)")
        print("=" * 100)

    # Create comparison DataFrame
    val_metrics_for_compare = validation_metrics[val_key].copy()
    val_metrics_for_compare['dataset'] = 'CV Validation'

    test_metrics_for_compare = test_metrics.copy()
    test_metrics_for_compare['dataset'] = 'Test'

    comparison_df = pd.DataFrame([val_metrics_for_compare, test_metrics_for_compare])
    comparison_df = comparison_df.set_index('dataset')

    # Drop non-metric columns if they exist
    cols_to_drop = [c for c in ['model'] if c in comparison_df.columns]
    if cols_to_drop:
        comparison_df = comparison_df.drop(columns=cols_to_drop)

    if verbose:
        # Calculate and print differences
        print("\nPerformance Differences (Test - CV Validation):")
        print("-" * 100)
        for metric in ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']:
            val_score = validation_metrics[val_key][metric]
            test_score = test_metrics[metric]
            diff = test_score - val_score
            diff_pct = (diff / val_score) * 100
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="

            # Determine if difference is concerning
            if abs(diff_pct) < 1:
                status = "✓ Excellent"
            elif abs(diff_pct) < 2:
                status = "✓ Good"
            elif abs(diff_pct) < 5:
                status = "⚠ Acceptable"
            else:
                status = "❌ Concerning"

            print(f"  {metric.upper():12s}: {val_score:.4f} → {test_score:.4f} "
                  f"({symbol} {diff:+.4f}, {diff_pct:+.2f}%) - {status}")

        print("-" * 100)

        # Overall assessment
        avg_diff_pct = abs(
            (test_metrics['pr_auc'] - validation_metrics[val_key]['pr_auc']) /
            validation_metrics[val_key]['pr_auc'] * 100
        )

        if avg_diff_pct < 1:
            print("\n✅ GENERALIZATION: Excellent - model generalizes very well to unseen data")
        elif avg_diff_pct < 2:
            print("\n✅ GENERALIZATION: Good - model shows stable performance on test set")
        elif avg_diff_pct < 5:
            print("\n⚠ GENERALIZATION: Acceptable - minor performance difference, monitor in production")
        else:
            print("\n❌ GENERALIZATION: Poor - significant performance gap, consider regularization")

        # Insight about CV vs test comparison
        if test_metrics['pr_auc'] >= validation_metrics[val_key]['pr_auc'] * 0.98:
            print(f"\n💡 INSIGHT: Test performance is consistent with CV validation metrics")
            print(f"   This confirms the model generalizes well to completely unseen data")
        else:
            print(f"\n💡 INSIGHT: Test performance slightly below CV validation")
            print(f"   This is normal - CV metrics can be slightly optimistic")

        print("=" * 100)

    return comparison_df
