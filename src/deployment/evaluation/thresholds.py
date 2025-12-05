"""
Threshold optimization for fraud detection.

Provides functions to find optimal classification thresholds
for different recall targets (conservative, balanced, aggressive).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve

from ..config import TrainingConfig


def optimize_thresholds(
    model,
    X_val,
    y_val,
    recall_targets: Dict[str, float] = None,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """Find optimal thresholds for different recall targets.

    For fraud detection, we often want to target specific recall levels
    (e.g., catch 90% of fraud cases) while maximizing precision at that level.

    Args:
        model: Trained model with predict_proba() method
        X_val: Validation feature matrix
        y_val: Validation labels
        recall_targets: Dictionary mapping names to target recall values
                       (default: from TrainingConfig)
        verbose: Whether to print results (default: True)

    Returns:
        Dictionary mapping threshold names to their configuration:
        {
            'threshold_name': {
                'threshold': float,      # Optimal threshold value
                'precision': float,      # Precision at this threshold
                'recall': float,         # Actual recall achieved
                'target_recall': float   # Target recall requested
            }
        }

    Examples:
        >>> config = optimize_thresholds(model, X_val, y_val)
        >>> conservative_threshold = config['conservative_90pct_recall']['threshold']
        >>> y_pred = (model.predict_proba(X_test)[:, 1] >= conservative_threshold).astype(int)
    """
    if recall_targets is None:
        recall_targets = TrainingConfig.get_threshold_targets()

    if verbose:
        print("\n" + "=" * 100)
        print("THRESHOLD OPTIMIZATION")
        print("=" * 100)

    # Get predicted probabilities
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Calculate precision-recall curve
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

    threshold_config = {}

    # 1. Calculate optimal F1 threshold (best overall precision-recall balance)
    if verbose:
        print("\n1. OPTIMAL F1 THRESHOLD (Best Precision-Recall Balance)")
        print("-" * 100)

    # Calculate F1 for all thresholds
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    # Find threshold with maximum F1
    best_f1_idx = np.argmax(f1_scores)
    optimal_f1_threshold = thresholds[best_f1_idx]
    optimal_f1_precision = precisions[best_f1_idx]
    optimal_f1_recall = recalls[best_f1_idx]
    optimal_f1_score = f1_scores[best_f1_idx]

    threshold_config['optimal_f1'] = {
        'threshold': float(optimal_f1_threshold),
        'precision': float(optimal_f1_precision),
        'recall': float(optimal_f1_recall),
        'f1': float(optimal_f1_score),
        'description': 'Optimal F1 score - best precision-recall balance (recommended default)'
    }

    if verbose:
        print(f"Optimal F1 Threshold: {optimal_f1_threshold:.4f}")
        print(f"  • F1 Score:   {optimal_f1_score:.4f} (MAXIMUM)")
        print(f"  • Precision:  {optimal_f1_precision:.4f} ({optimal_f1_precision*100:.2f}%)")
        print(f"  • Recall:     {optimal_f1_recall:.4f} ({optimal_f1_recall*100:.2f}%)")
        print("\nℹ️  This threshold maximizes F1 score - the harmonic mean of precision and recall")
        print("   It provides the best overall balance without targeting a specific recall level")

    # 2. Calculate recall-targeted thresholds
    if verbose:
        print("\n" + "=" * 100)
        print("2. RECALL-TARGETED THRESHOLDS")
        print("=" * 100)

    for name, target_recall in recall_targets.items():
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recalls - target_recall))
        threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        precision = precisions[idx]
        recall = recalls[idx]
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Map threshold names to descriptions
        descriptions = {
            'conservative_90pct_recall': 'Catch most fraud (90% recall), accept more false positives',
            'balanced_85pct_recall': 'Balanced precision-recall trade-off (85% recall target)',
            'aggressive_80pct_recall': 'Prioritize precision (80% recall), reduce false positives'
        }

        threshold_config[name] = {
            "threshold": float(threshold),
            "target_recall": target_recall,
            "achieved_recall": float(recall),
            "precision": float(precision),
            "f1": float(f1),
            "description": descriptions.get(name, f"Target {target_recall*100:.0f}% recall")
        }

        if verbose:
            print(f"\nTarget Recall: {target_recall*100:.0f}%")
            print(f"  • Optimal Threshold: {threshold:.4f}")
            print(f"  • Achieved Recall:   {recall:.4f} ({recall*100:.2f}%)")
            print(f"  • Precision:         {precision:.4f} ({precision*100:.2f}%)")
            print(f"  • F1 Score:          {f1:.4f}")

    if verbose:
        print("=" * 100)

    return threshold_config
