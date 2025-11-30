#!/usr/bin/env python3
"""
E-Commerce Fraud Detection Model Training Script

This script trains an XGBoost classifier for fraud detection using engineered features.
It loads preprocessed data, trains the model with optimized hyperparameters, evaluates
performance, and saves all deployment artifacts.

Usage:
    python train.py --data-dir data --output-dir models --random-seed 1
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Import production feature engineering pipeline and shared modules
from src.preprocessing.transformer import FraudFeatureTransformer
from src.preprocessing import PreprocessingPipelineFactory
from src.config import FeatureListsConfig, ModelConfig, TrainingConfig
from src.data import load_and_split_data
from src.evaluation import evaluate_model, optimize_thresholds


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost fraud detection model",
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
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning and use optimal params directly",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output during training"
    )

    return parser.parse_args()


# NOTE: Data loading, evaluate_model, optimize_thresholds, and preprocessing
# pipeline creation are now imported from shared modules
# (src/data, src/evaluation, and src/preprocessing)


def train_model(
    train_df, val_df, test_df, target_col="is_fraud", random_seed=1, skip_tuning=False, verbose=False
):
    """Train XGBoost model with hyperparameter tuning.

    Methodology:
    1. Combine train+val datasets (80% of total data)
    2. Use GridSearchCV with 4-fold CV on combined data for hyperparameter tuning
    3. GridSearchCV automatically retrains on full train+val with best params (refit=True)
    4. Evaluate final model on held-out test set (20%)

    This follows ML best practices and maximizes data available for hyperparameter selection.
    """

    # Load feature categorization from shared config
    feature_config = FeatureListsConfig.load()
    categorical_features = feature_config['categorical']
    continuous_numeric = feature_config['continuous_numeric']
    binary = feature_config['binary']

    # Combine train and validation for hyperparameter tuning
    # This gives GridSearchCV access to 80% of data instead of just 60%
    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    X_train_val = train_val_df.drop(columns=[target_col])
    y_train_val = train_val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print("\n" + "=" * 100)
    print("HYPERPARAMETER TUNING & MODEL TRAINING")
    print("=" * 100)
    print(f"Combined training data: {len(train_val_df):,} samples (train + validation)")
    print(f"  - Training set:   {len(train_df):,} samples")
    print(f"  - Validation set: {len(val_df):,} samples")
    print(f"Test data: {len(test_df):,} samples")
    print("=" * 100)

    # Create preprocessing pipeline using shared factory
    preprocessor = PreprocessingPipelineFactory.create_tree_pipeline(
        categorical_features, continuous_numeric, binary
    )

    # Feature names after preprocessing (order matches notebook)
    feature_names = categorical_features + continuous_numeric + binary

    # Load optimal hyperparameters using shared ModelConfig
    optimal_params = ModelConfig.load_hyperparameters(
        model_type='xgboost',
        source='metadata',
        random_seed=random_seed
    )

    if not skip_tuning:
        print("\nPerforming hyperparameter tuning with GridSearchCV...")
        print("Using 4-fold cross-validation on combined train+validation data")
        print("This may take several minutes...")

        # Create pipeline
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier(random_state=random_seed))])

        # Load parameter grid from shared config
        param_grid = ModelConfig.get_param_grid(model_type='xgboost')

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=TrainingConfig.get_cv_strategy(random_seed=random_seed),
            scoring="average_precision",  # Equivalent to XGBoost's eval_metric="aucpr"
            n_jobs=-1,
            verbose=2 if verbose else 0,
            refit=True,  # Automatically retrain on full train+val with best params
        )

        # Fit on combined train+val (80% of total data)
        # GridSearchCV will:
        # 1. Use 4-fold CV to find best parameters
        # 2. Automatically retrain on ALL of X_train_val with best params
        # 3. Store final model in best_estimator_
        grid_search.fit(X_train_val, y_train_val)

        print(f"\nBest parameters found (via 4-fold CV on {len(X_train_val):,} samples):")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV score (PR-AUC): {grid_search.best_score_:.4f}")

        # Use the best estimator (already trained on full train+val)
        final_pipeline = grid_search.best_estimator_

        # Update optimal params with tuned values
        for param, value in grid_search.best_params_.items():
            param_name = param.replace("classifier__", "")
            optimal_params[param_name] = value

    else:
        # skip_tuning=True: Train model with optimal params from config
        print("\n" + "=" * 100)
        print("TRAINING MODEL WITH OPTIMAL PARAMETERS (SKIPPING TUNING)")
        print("=" * 100)
        print(f"Training on {len(train_val_df):,} samples (train + validation)")

        # Create final production model with optimal params
        final_pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", xgb.XGBClassifier(**optimal_params)),
            ]
        )

        print("\nTraining final model...")
        final_pipeline.fit(X_train_val, y_train_val)

    # Evaluate on test set
    print("\n" + "=" * 100)
    print("TEST SET EVALUATION")
    print("=" * 100)
    test_metrics = evaluate_model(final_pipeline, X_test, y_test, "XGBoost (Final)", "Test")

    # Optimize thresholds using cross-validation predictions on train+val
    # This provides unbiased threshold estimates without needing a separate validation set
    print("\n" + "=" * 100)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 100)
    print("Using cross-validation predictions on train+val for threshold optimization...")

    from sklearn.model_selection import cross_val_predict

    # Get CV predictions on train+val
    cv_predictions = cross_val_predict(
        final_pipeline,
        X_train_val,
        y_train_val,
        cv=TrainingConfig.get_cv_strategy(random_seed=random_seed),
        method='predict_proba'
    )

    # Optimize thresholds using CV predictions
    # Note: We need to create a temporary object that has predict_proba method
    class PredictionWrapper:
        def __init__(self, predictions):
            self.predictions = predictions

        def predict_proba(self, X):
            return self.predictions

    wrapper = PredictionWrapper(cv_predictions)
    threshold_config = optimize_thresholds(wrapper, X_train_val, y_train_val)

    # Extract feature importance
    xgb_model = final_pipeline.named_steps["classifier"]
    importance_scores = xgb_model.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_scores}
    ).sort_values("importance", ascending=False)

    print("\n" + "=" * 100)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 100)
    for i, (idx, row) in enumerate(feature_importance_df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:40s} - Importance: {row['importance']:.6f}")
    print("=" * 100)

    return {
        "model": final_pipeline,
        "test_metrics": test_metrics,
        "threshold_config": threshold_config,
        "feature_importance": feature_importance_df,
        "feature_names": feature_names,
        "categorical_features": categorical_features,
        "continuous_numeric": continuous_numeric,
        "binary": binary,
        "optimal_params": optimal_params,
        # Additional metadata for comprehensive model_metadata.json
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
        "cv_best_score": grid_search.best_score_ if not skip_tuning else None,
        "tuning_performed": not skip_tuning,
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

    # 3. Save threshold configuration
    threshold_path = output_dir / "threshold_config.json"
    with open(threshold_path, "w") as f:
        json.dump(results["threshold_config"], f, indent=2)
    print(f"  ✓ Threshold config saved: {threshold_path}")

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

    # 5. Save model metadata (matching notebook structure)
    metadata = {
        "model_info": {
            "model_name": "XGBoost Fraud Detector",
            "model_type": "XGBClassifier",
            "version": "1.0",
            "training_date": datetime.now().strftime('%Y-%m-%d'),
            "framework": "xgboost + scikit-learn",
            "python_version": "3.12+",
            "note": "Final production model trained on train+val combined, evaluated on test set"
        },
        "hyperparameters": {
            param: value for param, value in results["optimal_params"].items()
        },
        "dataset_info": {
            "training_samples": results["dataset_info"]["train_val_samples"],
            "training_sources": {
                "original_train": results["dataset_info"]["train_samples"],
                "original_val": results["dataset_info"]["val_samples"],
                "combined_total": results["dataset_info"]["train_val_samples"]
            },
            "test_samples": results["dataset_info"]["test_samples"],
            "num_features": results["dataset_info"]["num_features"],
            "fraud_rate_train_val": results["dataset_info"]["fraud_rate_train_val"],
            "fraud_rate_test": results["dataset_info"]["fraud_rate_test"],
            "class_imbalance_ratio": results["dataset_info"]["class_imbalance_ratio"]
        },
        "performance": {
            "test_set_final": {
                "note": "Final performance from model trained on train+val combined",
                "roc_auc": float(results["test_metrics"]["roc_auc"]),
                "pr_auc": float(results["test_metrics"]["pr_auc"]),
                "f1_score": float(results["test_metrics"]["f1"]),
                "precision": float(results["test_metrics"]["precision"]),
                "recall": float(results["test_metrics"]["recall"]),
                "accuracy": float(results["test_metrics"]["accuracy"])
            },
            "cross_validation": {
                "cv_folds": 4,
                "cv_strategy": "StratifiedKFold",
                "best_cv_pr_auc": float(results["cv_best_score"]) if results["cv_best_score"] is not None else None,
                "note": "CV performed on train+val combined for hyperparameter selection" if results["tuning_performed"] else "No CV performed (skip-tuning mode)"
            }
        },
        "features": {
            "continuous_numeric": results["continuous_numeric"],
            "categorical": results["categorical_features"],
            "binary": results["binary"],
            "total_count": results["dataset_info"]["num_features"]
        },
        "preprocessing": {
            "categorical_encoding": "OrdinalEncoder (handle_unknown=use_encoded_value)",
            "numeric_scaling": "None (tree-based model)",
            "binary_features": "Passthrough (no transformation)"
        },
        "optimization": {
            "optimization_metric": "PR-AUC (Precision-Recall Area Under Curve)",
            "search_method": "GridSearchCV" if results["tuning_performed"] else "Pre-optimized parameters",
            "num_combinations_tested": 108 if results["tuning_performed"] else 0,
            "tuned_parameters": list(results["optimal_params"].keys()),
            "final_model_training": "Retrained on train+val combined with optimal hyperparameters"
        }
    }

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
        f.write("TOP 20 FEATURES\n")
        f.write("=" * 100 + "\n")
        for i, (idx, row) in enumerate(results["feature_importance"].head(20).iterrows(), 1):
            f.write(f"{i:2d}. {row['feature']:40s} - {row['importance']:.6f}\n")
        f.write("\n" + "=" * 100 + "\n")
        f.write("THRESHOLD CONFIGURATIONS\n")
        f.write("=" * 100 + "\n")
        for name, config in results["threshold_config"].items():
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
    print(f"Skip tuning: {args.skip_tuning}")
    print("=" * 100)

    try:
        # Load raw data and split using shared function from src.data
        train_raw, val_raw, test_raw = load_and_split_data(
            data_path=str(Path(args.data_dir) / "transactions.csv"),
            random_seed=args.random_seed,
            verbose=True
        )

        # Apply feature engineering pipeline using FraudFeatureTransformer
        print("\n" + "=" * 100)
        print("FEATURE ENGINEERING - USING PRODUCTION PIPELINE")
        print("=" * 100)
        print("Applying FraudFeatureTransformer from src/preprocessing/")

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
            skip_tuning=args.skip_tuning,
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
