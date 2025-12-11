#!/usr/bin/env python3
"""
E-Commerce Fraud Detection Model Training Script

This script trains an XGBoost classifier for fraud detection using engineered features.
It loads raw transaction data, applies the production feature engineering pipeline,
trains the model with pre-optimized hyperparameters (tuned in notebooks), evaluates
performance, and saves all deployment artifacts.

Usage:
    python train.py --data-dir data --output-dir models --random-seed 1

Note: Hyperparameter tuning was performed in notebooks/fd2_model_selection_tuning.ipynb.
This script uses those pre-optimized parameters for reproducible production training.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

# Import production feature engineering pipeline and shared modules
from src.deployment.preprocessing.transformer import FraudFeatureTransformer
from src.deployment.preprocessing import PreprocessingPipelineFactory
from src.deployment.config import (
    FeatureListsConfig,
    ModelConfig,
    build_threshold_config,
    build_model_metadata,
)
from src.deployment.data import load_and_split_data
from src.deployment.evaluation import evaluate_model

# Import threshold optimization from fd3_nb (single source of truth)
from src.fd3_nb.threshold_optimization import optimize_thresholds


def compute_shap_importance(model, X, feature_names, verbose=True):
    """
    Compute SHAP-based feature importance using XGBoost's native pred_contribs.

    This matches the approach used in fd3 notebook and the production API explainability.

    Args:
        model: Trained sklearn Pipeline with XGBClassifier
        X: Feature matrix
        feature_names: List of feature names in order after preprocessing
        verbose: Whether to print progress

    Returns:
        Tuple of (importance_df, shap_values_matrix)
    """
    if verbose:
        print("Computing SHAP values using XGBoost native interface...")

    # Extract XGBoost classifier and get the booster
    xgb_model = model.named_steps['classifier']
    booster = xgb_model.get_booster()

    # Apply preprocessor to get numeric features
    preprocessor = model.named_steps['preprocessor']
    X_processed = preprocessor.transform(X)

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X_processed)

    # Compute SHAP values using pred_contribs
    shap_values = booster.predict(dmatrix, pred_contribs=True)

    # Remove the bias column (last column)
    shap_values = shap_values[:, :-1]

    if verbose:
        print(f"  ✓ SHAP values computed: {shap_values.shape[0]:,} samples x {shap_values.shape[1]} features")

    # Compute global importance as mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)  # Signed mean (direction of effect)

    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_abs_shap,
        'mean_shap': mean_shap
    }).sort_values('shap_importance', ascending=False)

    return importance_df, shap_values


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost fraud detection model with pre-optimized hyperparameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing raw csv data file",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--random-seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output during training"
    )

    return parser.parse_args()


def train_model(train_df, val_df, test_df, target_col="is_fraud", random_seed=1, verbose=False):
    """Train XGBoost model with pre-optimized hyperparameters.

    Methodology:
    1. Combine train+val datasets (80% of total data)
    2. Train model with hyperparameters tuned in notebooks/fd2_model_selection_tuning.ipynb
    3. Evaluate final model on held-out test set (20%)

    Note: Hyperparameter tuning was performed extensively in the fd2 notebook using
    GridSearchCV with 4-fold CV. This script uses those pre-optimized parameters
    for reproducible production training.
    """

    # Load feature categorization from shared config
    feature_config = FeatureListsConfig.load()
    categorical_features = feature_config['categorical']
    continuous_numeric = feature_config['continuous_numeric']
    binary = feature_config['binary']

    # Combine train and validation for final training
    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    X_train_val = train_val_df.drop(columns=[target_col])
    y_train_val = train_val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print("\n" + "=" * 100)
    print("MODEL TRAINING")
    print("=" * 100)
    print(f"Training data: {len(train_val_df):,} samples (train + validation combined)")
    print(f"  - Original training set:   {len(train_df):,} samples")
    print(f"  - Original validation set: {len(val_df):,} samples")
    print(f"Test data: {len(test_df):,} samples (held-out)")
    print("=" * 100)

    # Create preprocessing pipeline using shared factory
    preprocessor = PreprocessingPipelineFactory.create_tree_pipeline(
        categorical_features, continuous_numeric, binary
    )

    # Feature names after preprocessing
    feature_names = categorical_features + continuous_numeric + binary

    # Load pre-optimized hyperparameters (tuned in fd2 notebook)
    optimal_params = ModelConfig.load_hyperparameters(
        model_type='xgboost',
        source='metadata',
        random_seed=random_seed
    )

    print("\nUsing pre-optimized hyperparameters (from fd2 notebook tuning):")
    print(f"  n_estimators: {optimal_params.get('n_estimators', 'N/A')}")
    print(f"  max_depth: {optimal_params.get('max_depth', 'N/A')}")
    print(f"  learning_rate: {optimal_params.get('learning_rate', 'N/A')}")
    print(f"  scale_pos_weight: {optimal_params.get('scale_pos_weight', 'N/A')}")

    # Create final production model with optimal params
    final_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", xgb.XGBClassifier(**optimal_params)),
        ]
    )

    print("\nTraining final model...")
    final_pipeline.fit(X_train_val, y_train_val)
    print("  ✓ Model training complete")

    # Evaluate on test set
    print("\n" + "=" * 100)
    print("TEST SET EVALUATION")
    print("=" * 100)
    test_metrics = evaluate_model(final_pipeline, X_test, y_test, "XGBoost (Final)", "Test")

    # Get test set predictions for threshold optimization
    y_test_proba = final_pipeline.predict_proba(X_test)[:, 1]

    # Optimize thresholds on test set (using shared fd3_nb implementation)
    print("\n" + "=" * 100)
    print("THRESHOLD OPTIMIZATION (on Test Set)")
    print("=" * 100)
    print("Finding optimal thresholds for different business requirements...")
    optimal_f1_result, target_performance_result, threshold_results = optimize_thresholds(
        y_test, y_test_proba, verbose=True
    )

    # Convert to optimized_thresholds dict format for build_threshold_config
    optimized_thresholds = {
        'optimal_f1': {
            'threshold': float(optimal_f1_result['threshold']),
            'precision': float(optimal_f1_result['precision']),
            'recall': float(optimal_f1_result['recall']),
            'f1': float(optimal_f1_result['f1']),
            'tp': int(optimal_f1_result['tp']),
            'fp': int(optimal_f1_result['fp']),
            'tn': int(optimal_f1_result['tn']),
            'fn': int(optimal_f1_result['fn']),
            'description': 'Optimal F1 score - best precision-recall balance'
        }
    }

    if target_performance_result is not None:
        optimized_thresholds['target_performance'] = {
            'threshold': float(target_performance_result['threshold']),
            'precision': float(target_performance_result['precision']),
            'recall': float(target_performance_result['recall']),
            'f1': float(target_performance_result['f1']),
            'min_precision': float(target_performance_result.get('min_precision', 0.70)),
            'tp': int(target_performance_result['tp']),
            'fp': int(target_performance_result['fp']),
            'tn': int(target_performance_result['tn']),
            'fn': int(target_performance_result['fn']),
            'description': 'Max recall while maintaining >=70% precision (recommended)'
        }

    # Add recall-targeted thresholds
    recall_names = ['conservative_90pct_recall', 'balanced_85pct_recall', 'aggressive_80pct_recall']
    recall_descriptions = [
        'Catch most fraud (90% recall), accept more false positives',
        'Balanced precision-recall trade-off (85% recall target)',
        'Prioritize precision (80% recall), reduce false positives'
    ]
    for i, result in enumerate(threshold_results):
        optimized_thresholds[recall_names[i]] = {
            'threshold': float(result['threshold']),
            'target_recall': float(result['target_recall']),
            'achieved_recall': float(result['recall']),
            'precision': float(result['precision']),
            'f1': float(result['f1']),
            'tp': int(result['tp']),
            'fp': int(result['fp']),
            'tn': int(result['tn']),
            'fn': int(result['fn']),
            'description': recall_descriptions[i]
        }

    # Compute SHAP-based feature importance (matches fd3 notebook and API explainability)
    print("\n" + "=" * 100)
    print("FEATURE IMPORTANCE (SHAP Values)")
    print("=" * 100)
    feature_importance_df, shap_values = compute_shap_importance(
        final_pipeline, X_test, feature_names, verbose=True
    )

    print("\nTop 10 Features by Mean |SHAP Value|:")
    print("-" * 100)
    for i, (_, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        direction = "↑ fraud" if row['mean_shap'] > 0 else "↓ fraud"
        print(f"  {i:2d}. {row['feature']:40s} - Importance: {row['shap_importance']:.6f}  ({direction})")
    print("=" * 100)

    return {
        "model": final_pipeline,
        "test_metrics": test_metrics,
        "optimized_thresholds": optimized_thresholds,
        "feature_importance": feature_importance_df,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "continuous_numeric": continuous_numeric,
        "binary": binary,
        "optimal_params": optimal_params,
        "dataset_info": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "train_val_samples": len(train_val_df),
            "test_samples": len(test_df),
            "num_features": len(feature_names),
            "fraud_rate_train_val": float(y_train_val.mean()),
            "fraud_rate_test": float(y_test.mean()),
            "class_imbalance_ratio": float((y_train_val == 0).sum() / (y_train_val == 1).sum()),
        },
    }


def save_artifacts(results, output_dir: Path, random_seed: int):
    """Save all model artifacts for deployment."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("SAVING MODEL ARTIFACTS")
    print("=" * 100)

    # 1. Save model
    model_path = output_dir / "xgb_fraud_detector.joblib"
    joblib.dump(results["model"], model_path)
    print(f"  ✓ Model saved: {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # 2. Save feature transformer configuration
    transformer_config_path = output_dir / "transformer_config.json"
    results["transformer"].save(str(transformer_config_path))
    print(f"  ✓ Transformer config saved: {transformer_config_path}")

    # 3. Save threshold configuration (using shared builder)
    threshold_config = build_threshold_config(results["optimized_thresholds"])
    threshold_path = output_dir / "threshold_config.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_config, f, indent=2)
    print(f"  ✓ Threshold config saved: {threshold_path}")
    print(f"    {len(results['optimized_thresholds'])} threshold strategies available")

    # 4. Save feature lists
    feature_lists = {
        "categorical": results["categorical_features"],
        "continuous_numeric": results["continuous_numeric"],
        "binary": results["binary"],
        "all_features": results["feature_names"],
    }
    feature_lists_path = output_dir / "feature_lists.json"
    with open(feature_lists_path, "w") as f:
        json.dump(feature_lists, f, indent=2)
    print(f"  ✓ Feature lists saved: {feature_lists_path}")

    # 5. Save model metadata (using shared builder)
    dataset_info = {
        "training_samples": results["dataset_info"]["train_val_samples"],
        "training_sources": {
            "original_train": results["dataset_info"]["train_samples"],
            "original_val": results["dataset_info"]["val_samples"],
            "combined_total": results["dataset_info"]["train_val_samples"]
        },
        "test_samples": results["dataset_info"]["test_samples"],
        "num_features": results["dataset_info"]["num_features"],
        "fraud_rate_train": results["dataset_info"]["fraud_rate_train_val"],
        "fraud_rate_test": results["dataset_info"]["fraud_rate_test"],
        "class_imbalance_ratio": results["dataset_info"]["class_imbalance_ratio"]
    }
    feature_lists = {
        "continuous_numeric": results["continuous_numeric"],
        "categorical": results["categorical_features"],
        "binary": results["binary"]
    }
    metadata = build_model_metadata(
        hyperparameters=results["optimal_params"],
        test_metrics=results["test_metrics"],
        dataset_info=dataset_info,
        feature_lists=feature_lists,
        note="Production model trained on train+val combined with pre-optimized hyperparameters"
    )

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Model metadata saved: {metadata_path}")

    # 6. Save training report
    report_path = output_dir / "training_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("XGBoost FRAUD DETECTION MODEL - TRAINING REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Version: 1.0\n")
        f.write(f"Random Seed: {random_seed}\n\n")
        f.write("=" * 100 + "\n")
        f.write("TEST SET PERFORMANCE\n")
        f.write("=" * 100 + "\n")
        for metric, value in results["test_metrics"].items():
            f.write(f"{metric.upper():12s}: {value:.4f}\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write("TOP 20 FEATURES (SHAP Importance)\n")
        f.write("=" * 100 + "\n")
        for i, (_, row) in enumerate(results["feature_importance"].head(20).iterrows(), 1):
            direction = "↑ fraud" if row['mean_shap'] > 0 else "↓ fraud"
            f.write(f"{i:2d}. {row['feature']:40s} - {row['shap_importance']:.6f}  ({direction})\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write("THRESHOLD CONFIGURATIONS\n")
        f.write("=" * 100 + "\n")
        for name, config in results["optimized_thresholds"].items():
            f.write(f"\n{name}:\n")
            for key, value in config.items():
                f.write(f"  {key}: {value}\n")

    print(f"  ✓ Training report saved: {report_path}")
    print("=" * 100)

    return {
        "model_path": model_path,
        "transformer_config_path": transformer_config_path,
        "threshold_path": threshold_path,
        "feature_lists_path": feature_lists_path,
        "metadata_path": metadata_path,
        "report_path": report_path,
    }


def main():
    """Main training pipeline."""
    args = parse_args()

    print("=" * 100)
    print("E-COMMERCE FRAUD DETECTION - MODEL TRAINING")
    print("=" * 100)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.random_seed}")
    print("=" * 100)

    try:
        # Load raw data and split using shared function
        train_raw, val_raw, test_raw = load_and_split_data(
            data_path=str(Path(args.data_dir) / "transactions.csv"),
            random_seed=args.random_seed,
            verbose=True
        )

        # Apply feature engineering pipeline using FraudFeatureTransformer
        print("\n" + "=" * 100)
        print("FEATURE ENGINEERING")
        print("=" * 100)
        print("Applying FraudFeatureTransformer...")

        transformer = FraudFeatureTransformer()
        transformer.fit(train_raw)

        print(f"  ✓ Transformer fitted on {len(train_raw):,} training samples")

        # Transform all datasets
        train_df = transformer.transform(train_raw)
        val_df = transformer.transform(val_raw)
        test_df = transformer.transform(test_raw)

        print(f"  ✓ Engineered features: {train_df.shape[1]} features")
        print(f"  ✓ Training set: {len(train_df):,} samples")
        print(f"  ✓ Validation set: {len(val_df):,} samples")
        print(f"  ✓ Test set: {len(test_df):,} samples")

        # Add target column back (transformer removes it)
        train_df['is_fraud'] = train_raw['is_fraud'].values
        val_df['is_fraud'] = val_raw['is_fraud'].values
        test_df['is_fraud'] = test_raw['is_fraud'].values

        # Train model on engineered features
        results = train_model(
            train_df,
            val_df,
            test_df,
            random_seed=args.random_seed,
            verbose=args.verbose,
        )

        # Add transformer to results for saving
        results['transformer'] = transformer

        # Save artifacts
        output_dir = Path(args.output_dir)
        saved_files = save_artifacts(results, output_dir, args.random_seed)

        # Final summary
        print("\n" + "=" * 100)
        print("TRAINING COMPLETE")
        print("=" * 100)
        print(f"\nTest Set Performance:")
        print(f"  PR-AUC:    {results['test_metrics']['pr_auc']:.4f}")
        print(f"  Precision: {results['test_metrics']['precision']:.4f}")
        print(f"  Recall:    {results['test_metrics']['recall']:.4f}")
        print(f"  F1 Score:  {results['test_metrics']['f1']:.4f}")
        print(f"\nModel ready for deployment!")
        print(f"Model location: {saved_files['model_path']}")
        print("=" * 100)

        return 0

    except Exception as e:
        print(f"\n❌ Error during training: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
