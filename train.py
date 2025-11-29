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
    """Train XGBoost model with hyperparameter tuning."""

    # Load feature categorization from shared config
    feature_config = FeatureListsConfig.load()
    categorical_features = feature_config['categorical']
    continuous_numeric = feature_config['continuous_numeric']
    binary = feature_config['binary']

    # Split features and target
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col]
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col]

    print("\n" + "=" * 100)
    print("FEATURE ENGINEERING & MODEL TRAINING")
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
        print("This may take several minutes...")

        # Create pipeline
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier(random_state=random_seed))])

        # Reduced parameter grid for faster tuning
        param_grid = {
            "classifier__n_estimators": [70, 90, 110],
            "classifier__max_depth": [4, 5, 6],
            "classifier__learning_rate": [0.06, 0.08, 0.10],
            "classifier__scale_pos_weight": [6, 8, 10],
            "classifier__gamma": [0.4, 0.6, 0.8],
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=TrainingConfig.get_cv_strategy(random_seed=random_seed),
            scoring="average_precision",
            n_jobs=-1,
            verbose=2 if verbose else 0,
        )

        grid_search.fit(X_train, y_train)

        print(f"\nBest parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        # Evaluate tuned model on validation set
        val_metrics = evaluate_model(grid_search.best_estimator_, X_val, y_val, "XGBoost (Tuned)", "Validation")

        # Update optimal params with tuned values
        for param, value in grid_search.best_params_.items():
            param_name = param.replace("classifier__", "")
            optimal_params[param_name] = value

    # Retrain on train+val combined with optimal hyperparameters
    print("\n" + "=" * 100)
    print("RETRAINING FINAL MODEL ON TRAIN+VALIDATION COMBINED")
    print("=" * 100)

    train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
    X_train_val = train_val_df.drop(columns=[target_col])
    y_train_val = train_val_df[target_col]

    print(f"Combined training samples: {len(train_val_df):,}")

    # Create final production model
    final_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", xgb.XGBClassifier(**optimal_params)),
        ]
    )

    print("\nTraining final model...")
    final_pipeline.fit(X_train_val, y_train_val)

    # Evaluate on test set
    test_metrics = evaluate_model(final_pipeline, X_test, y_test, "XGBoost (Final - Retrained)", "Test")

    # Optimize thresholds on validation set
    threshold_config = optimize_thresholds(final_pipeline, X_val, y_val)

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

    # 5. Save model metadata
    metadata = {
        "model_info": {
            "name": "XGBoost Fraud Detector",
            "version": "1.0",
            "training_date": datetime.now().isoformat(),
            "algorithm": "XGBoost Gradient Boosting",
            "training_methodology": "GridSearchCV tuning on train → Retrain on train+val → Evaluate on test",
        },
        "hyperparameters": results["optimal_params"],
        "performance": {
            "test_set": {
                "pr_auc": results["test_metrics"]["pr_auc"],
                "roc_auc": results["test_metrics"]["roc_auc"],
                "precision": results["test_metrics"]["precision"],
                "recall": results["test_metrics"]["recall"],
                "f1": results["test_metrics"]["f1"],
            }
        },
        "feature_importance_top10": results["feature_importance"].head(10)[["feature", "importance"]].to_dict(
            orient="records"
        ),
        "threshold_strategies": results["threshold_config"],
        "random_seed": random_seed,
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
