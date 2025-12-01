"""
Model evaluation metrics.

Provides standardized model evaluation functions across all scripts,
ensuring consistent metric calculation and reporting.
"""

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """Calculate all evaluation metrics without printing.

    Args:
        y_true: True labels
        y_pred: Predicted labels (binary)
        y_proba: Predicted probabilities for positive class

    Returns:
        Dictionary containing all metrics

    Examples:
        >>> metrics = calculate_metrics(y_test, y_pred, y_proba)
        >>> print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    """
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def evaluate_model(
    model,
    X,
    y,
    model_name: str = "Model",
    dataset_name: str = "Dataset",
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate model performance and optionally print results.

    This is the comprehensive evaluation function used across all scripts.
    It calculates all relevant metrics and optionally prints a formatted report.

    Args:
        model: Trained model with predict() and predict_proba() methods
        X: Feature matrix
        y: True labels
        model_name: Name of the model for display (default: "Model")
        dataset_name: Name of the dataset for display (default: "Dataset")
        verbose: Whether to print evaluation results (default: True)

    Returns:
        Dictionary containing all metrics

    Examples:
        >>> metrics = evaluate_model(model, X_test, y_test, "XGBoost", "Test")
        >>> metrics = evaluate_model(model, X_val, y_val, "Random Forest", "Validation", verbose=False)
    """
    # Get predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y, y_pred, y_proba)

    # Print evaluation results if verbose
    if verbose:
        print(f"\n{'=' * 100}")
        print(f"{model_name} - {dataset_name} Set Performance")
        print(f"{'=' * 100}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:  {metrics['f1']:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0, 0]:,}  |  FP: {cm[0, 1]:,}")
        print(f"  FN: {cm[1, 0]:,}  |  TP: {cm[1, 1]:,}")
        print(f"{'=' * 100}\n")

    return metrics
