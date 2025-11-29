"""
Threshold optimization for fraud detection.

Provides functions to find optimal classification thresholds
for different recall targets (conservative, balanced, aggressive).
"""

from typing import Dict

import numpy as np
from sklearn.metrics import precision_recall_curve

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

    for name, target_recall in recall_targets.items():
        # Find threshold closest to target recall
        idx = np.argmin(np.abs(recalls - target_recall))
        threshold = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
        precision = precisions[idx]
        recall = recalls[idx]

        threshold_config[name] = {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "target_recall": target_recall,
        }

        if verbose:
            print(f"\n{name}:")
            print(f"  Target Recall: {target_recall:.1%}")
            print(f"  Actual Recall: {recall:.4f}")
            print(f"  Precision:     {precision:.4f}")
            print(f"  Threshold:     {threshold:.6f}")

    if verbose:
        print("=" * 100)

    return threshold_config
