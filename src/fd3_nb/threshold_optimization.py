"""
Threshold optimization utilities for fd3 notebook.

This module provides functions for finding optimal classification thresholds
based on different business requirements (F1 optimization, target recall, etc.).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve


def find_threshold_for_recall(
    target_recall: float,
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Find threshold that achieves target recall and maximizes precision.

    Args:
        target_recall: Target recall level (e.g., 0.90 for 90%)
        precisions: Precision values from precision_recall_curve
        recalls: Recall values from precision_recall_curve
        thresholds: Threshold values from precision_recall_curve

    Returns:
        Tuple of (threshold, precision, recall, f1) or (None, None, None, None) if not found
    """
    # Find indices where recall >= target_recall
    valid_indices = np.where(recalls[:-1] >= target_recall)[0]

    if len(valid_indices) == 0:
        return None, None, None, None

    # Among valid thresholds, find the one with highest precision
    best_idx = valid_indices[np.argmax(precisions[:-1][valid_indices])]

    threshold = thresholds[best_idx]
    precision = precisions[best_idx]
    recall = recalls[best_idx]
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return threshold, precision, recall, f1


def find_optimal_f1_threshold(
    precisions: np.ndarray,
    recalls: np.ndarray,
    thresholds: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Find threshold that maximizes F1 score.

    Args:
        precisions: Precision values from precision_recall_curve
        recalls: Recall values from precision_recall_curve
        thresholds: Threshold values from precision_recall_curve

    Returns:
        Tuple of (threshold, precision, recall, f1_score)
    """
    # Calculate F1 for all thresholds
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

    # Find threshold with maximum F1
    best_f1_idx = np.argmax(f1_scores)

    return (
        thresholds[best_f1_idx],
        precisions[best_f1_idx],
        recalls[best_f1_idx],
        f1_scores[best_f1_idx]
    )


def optimize_thresholds(
    y_true: pd.Series,
    y_pred_proba: np.ndarray,
    recall_targets: List[float] = [0.90, 0.85, 0.80],
    verbose: bool = True
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optimize classification thresholds for different business requirements.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        recall_targets: List of target recall levels
        verbose: If True, print detailed results

    Returns:
        Tuple of (optimal_f1_result, threshold_results_list)
    """
    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    if verbose:
        print("=" * 100)
        print("THRESHOLD OPTIMIZATION - Finding Optimal Thresholds")
        print("=" * 100)
        print("Using test set predictions from final retrained model")
        print("=" * 100)

    # 1. Find optimal F1 threshold
    if verbose:
        print("\n1. OPTIMAL F1 THRESHOLD (Best Precision-Recall Balance)")
        print("-" * 100)

    optimal_f1_threshold, optimal_f1_precision, optimal_f1_recall, optimal_f1_score = \
        find_optimal_f1_threshold(precisions, recalls, thresholds)

    # Calculate confusion matrix for optimal F1
    y_pred_optimal_f1 = (y_pred_proba >= optimal_f1_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_optimal_f1)
    tn, fp, fn, tp = cm.ravel()

    optimal_f1_result = {
        'name': 'optimal_f1',
        'threshold': optimal_f1_threshold,
        'precision': optimal_f1_precision,
        'recall': optimal_f1_recall,
        'f1': optimal_f1_score,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp
    }

    if verbose:
        print(f"Optimal F1 Threshold: {optimal_f1_threshold:.4f}")
        print(f"  • F1 Score:          {optimal_f1_score:.4f} (MAXIMUM)")
        print(f"  • Precision:         {optimal_f1_precision:.4f} ({optimal_f1_precision*100:.2f}%)")
        print(f"  • Recall:            {optimal_f1_recall:.4f} ({optimal_f1_recall*100:.2f}%)")
        print(f"  • Confusion Matrix:  TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
        print(f"  • False Positive Rate: {fp/(fp+tn)*100:.2f}%")
        print(f"  • False Negative Rate: {fn/(fn+tp)*100:.2f}%")
        print(f"\nℹ️  This threshold maximizes F1 score - the harmonic mean of precision and recall")
        print(f"   It provides the best overall balance without targeting a specific recall level")

    # 2. Find recall-targeted thresholds
    if verbose:
        print("\n" + "=" * 100)
        print("2. RECALL-TARGETED THRESHOLDS")
        print("=" * 100)

    threshold_results = []

    for target_recall in recall_targets:
        threshold, precision, recall, f1 = find_threshold_for_recall(
            target_recall, precisions, recalls, thresholds
        )

        if threshold is not None:
            # Calculate confusion matrix at this threshold
            y_pred_custom = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred_custom)
            tn, fp, fn, tp = cm.ravel()

            result = {
                'target_recall': target_recall,
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            }
            threshold_results.append(result)

            if verbose:
                print(f"\nTarget Recall: {target_recall*100:.0f}%")
                print(f"  • Optimal Threshold: {threshold:.4f}")
                print(f"  • Achieved Recall:   {recall:.4f} ({recall*100:.2f}%)")
                print(f"  • Precision:         {precision:.4f} ({precision*100:.2f}%)")
                print(f"  • F1 Score:          {f1:.4f}")
                print(f"  • Confusion Matrix:  TN={tn:,} | FP={fp:,} | FN={fn:,} | TP={tp:,}")
                print(f"  • False Positive Rate: {fp/(fp+tn)*100:.2f}%")
                print(f"  • False Negative Rate: {fn/(fn+tp)*100:.2f}%")

    if verbose:
        print("=" * 100)

    return optimal_f1_result, threshold_results


def create_threshold_comparison_df(
    optimal_f1_result: Dict[str, Any],
    threshold_results: List[Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a DataFrame comparing all threshold strategies.

    Args:
        optimal_f1_result: Dict with optimal F1 threshold results
        threshold_results: List of dicts with recall-targeted threshold results

    Returns:
        DataFrame with threshold comparison
    """
    rows = [
        {
            'strategy': 'Optimal F1 (Best Balance)',
            'threshold': optimal_f1_result['threshold'],
            'precision': optimal_f1_result['precision'],
            'recall': optimal_f1_result['recall'],
            'f1': optimal_f1_result['f1'],
            'fp': optimal_f1_result['fp'],
            'fn': optimal_f1_result['fn']
        }
    ]

    strategy_names = {
        0.90: 'Conservative (90% Recall)',
        0.85: 'Balanced (85% Recall)',
        0.80: 'Aggressive (80% Recall)'
    }

    for result in threshold_results:
        strategy = strategy_names.get(
            result['target_recall'],
            f"{result['target_recall']*100:.0f}% Recall"
        )
        rows.append({
            'strategy': strategy,
            'threshold': result['threshold'],
            'precision': result['precision'],
            'recall': result['recall'],
            'f1': result['f1'],
            'fp': result['fp'],
            'fn': result['fn']
        })

    return pd.DataFrame(rows)
