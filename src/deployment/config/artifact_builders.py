"""
Shared builders for deployment configuration files.

This module provides functions to build threshold_config.json and model_metadata.json
structures, ensuring consistency between train.py and fd3 notebook workflows.

All default values are loaded from deployment_defaults.json (single source of truth).
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Load defaults from JSON file (single source of truth)
_CONFIG_PATH = Path(__file__).parent / "deployment_defaults.json"
with open(_CONFIG_PATH) as f:
    _DEFAULTS = json.load(f)

_DEPLOYMENT = _DEFAULTS["deployment"]

# Export constants for use by other modules
RISK_LEVELS: dict[str, dict[str, float]] = _DEPLOYMENT["risk_levels"]
DEFAULT_THRESHOLD: float = _DEPLOYMENT["default_threshold"]
MODEL_INFO_DEFAULTS: dict[str, str] = _DEPLOYMENT["model_info_defaults"]
PREPROCESSING_INFO: dict[str, str] = _DEPLOYMENT["preprocessing_info"]


def build_threshold_config(
    optimized_thresholds: dict[str, dict[str, Any]],
    note: str = "Thresholds optimized on held-out test set predictions."
) -> dict[str, Any]:
    """
    Build complete threshold_config.json structure.

    Args:
        optimized_thresholds: Dict mapping threshold strategy names to their configs.
            Each config should contain: threshold, precision, recall, f1, description.
            Optional fields: min_precision, target_recall, achieved_recall, tp, fp, tn, fn.
        note: Explanatory note for the config file.

    Returns:
        Complete threshold_config structure ready for JSON serialization.
    """
    recommended = "target_performance" if "target_performance" in optimized_thresholds else "optimal_f1"

    return {
        "default_threshold": DEFAULT_THRESHOLD,
        "recommended_threshold": recommended,
        "risk_levels": RISK_LEVELS,
        "optimized_thresholds": optimized_thresholds,
        "note": note
    }


def build_model_metadata(
    hyperparameters: dict[str, Any],
    test_metrics: dict[str, float],
    dataset_info: dict[str, Any],
    feature_lists: dict[str, list[str]],
    note: str = "Production model",
    cv_metrics: dict[str, Any] | None = None,
    workflow_info: dict[str, str] | None = None
) -> dict[str, Any]:
    """
    Build complete model_metadata.json structure.

    Args:
        hyperparameters: Model hyperparameters dict (keys like n_estimators, max_depth, etc.)
        test_metrics: Test set performance metrics with keys: roc_auc, pr_auc, f1, precision, recall, accuracy
        dataset_info: Dataset information with keys: training_samples, training_sources, test_samples,
            num_features, fraud_rate_train, fraud_rate_test, class_imbalance_ratio
        feature_lists: Feature categorization with keys: continuous_numeric, categorical, binary
        note: Explanatory note for model_info section
        cv_metrics: Optional cross-validation metrics with keys: cv_folds, cv_strategy, cv_pr_auc, note
        workflow_info: Optional workflow documentation with keys like training_notebook, evaluation_notebook

    Returns:
        Complete model_metadata structure ready for JSON serialization.
    """
    # Calculate total feature count from lists
    total_features = (
        len(feature_lists.get("continuous_numeric", []))
        + len(feature_lists.get("categorical", []))
        + len(feature_lists.get("binary", []))
    )

    metadata: dict[str, Any] = {
        "model_info": {
            "model_name": MODEL_INFO_DEFAULTS["model_name"],
            "model_type": MODEL_INFO_DEFAULTS["model_type"],
            "version": MODEL_INFO_DEFAULTS["version"],
            "training_date": datetime.now().strftime("%Y-%m-%d"),
            "framework": MODEL_INFO_DEFAULTS["framework"],
            "python_version": MODEL_INFO_DEFAULTS["python_version"],
            "note": note
        },
        "hyperparameters": {k: v for k, v in hyperparameters.items()},
        "dataset_info": dataset_info,
        "performance": {
            "test_set": {
                "note": "Performance on held-out test set",
                "roc_auc": float(test_metrics["roc_auc"]),
                "pr_auc": float(test_metrics["pr_auc"]),
                "f1_score": float(test_metrics["f1"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "accuracy": float(test_metrics["accuracy"])
            }
        },
        "features": {
            "continuous_numeric": feature_lists.get("continuous_numeric", []),
            "categorical": feature_lists.get("categorical", []),
            "binary": feature_lists.get("binary", []),
            "total_count": total_features
        },
        "preprocessing": PREPROCESSING_INFO.copy()
    }

    # Add optional CV metrics
    if cv_metrics is not None:
        metadata["performance"]["cross_validation"] = cv_metrics

    # Add optional workflow info
    if workflow_info is not None:
        metadata["workflow"] = workflow_info

    return metadata
