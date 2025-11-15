#!/usr/bin/env python3
"""
E-Commerce Fraud Detection Model Training Script

This script trains an XGBoost classifier for fraud detection using engineered features.
It loads preprocessed data, trains the model with optimized hyperparameters, evaluates
performance, and saves all deployment artifacts.

Usage:
    python train.py --data-dir data --output-dir models --random-seed 42
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
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# Import production feature engineering pipeline
from src.preprocessing.transformer import FraudFeatureTransformer


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
        help="Directory containing train/val/test pickle files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Directory to save model artifacts"
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
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


def load_data(data_dir: Path, random_seed: int = 1):
    """Load raw transaction data and split into train/val/test sets."""
    print("Loading raw transaction data...")
    print(f"  Data directory: {data_dir}")

    csv_path = data_dir / "transactions.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {csv_path}\n"
            "Please ensure data/transactions.csv exists.\n"
            "Download from Kaggle: umuttuygurr/e-commerce-fraud-detection-dataset"
        )

    # Load raw CSV data
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Total samples: {len(df):,}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")

    # Split into train/val/test (60/20/20)
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df['is_fraud'], random_state=random_seed
    )

    # Second split: separate train and validation
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.25, stratify=train_val_df['is_fraud'], random_state=random_seed
    )

    print(f"\n  Training set: {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df


def create_preprocessing_pipeline(categorical_features: list, continuous_numeric: list, binary: list) -> ColumnTransformer:
    """Create preprocessing pipeline for tree-based models.

    Matches the notebook's preprocessing approach for exact reproducibility.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
                categorical_features,
            ),
            ("rest", "passthrough", continuous_numeric + binary)
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate_model(model, X, y, model_name="Model", dataset_name="Dataset"):
    """Evaluate model performance and return metrics dictionary."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "pr_auc": average_precision_score(y, y_proba),
        "roc_auc": roc_auc_score(y, y_proba),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }

    # Print evaluation results
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


def optimize_thresholds(model, X_val, y_val):
    """Find optimal thresholds for different recall targets."""
    print("\n" + "=" * 100)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 100)

    y_val_proba = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_proba)

    # Target recall levels
    recall_targets = {"conservative_90pct_recall": 0.90, "balanced_85pct_recall": 0.85, "aggressive_80pct_recall": 0.80}

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

        print(f"\n{name}:")
        print(f"  Target Recall: {target_recall:.1%}")
        print(f"  Actual Recall: {recall:.4f}")
        print(f"  Precision:     {precision:.4f}")
        print(f"  Threshold:     {threshold:.6f}")

    print("=" * 100)

    return threshold_config


def train_model(
    train_df, val_df, test_df, target_col="is_fraud", random_seed=1, skip_tuning=False, verbose=False
):
    """Train XGBoost model with hyperparameter tuning."""

    # Define feature categories to match notebook preprocessing
    # After FraudFeatureTransformer, these features remain categorical or should be treated as such
    categorical_features = ['channel', 'promo_used', 'avs_match', 'cvv_result', 'three_ds_flag']

    # Continuous numeric features
    continuous_numeric = [
        'account_age_days', 'total_transactions_user', 'avg_amount_user',
        'amount', 'shipping_distance_km', 'hour_local', 'day_of_week_local',
        'month_local', 'amount_deviation', 'amount_vs_avg_ratio',
        'transaction_velocity', 'security_score'
    ]

    # Binary features (already 0/1, no preprocessing needed but grouped for consistency)
    binary = [
        'is_weekend_local', 'is_late_night_local', 'is_business_hours_local',
        'is_micro_transaction', 'is_large_transaction', 'is_new_account',
        'is_high_frequency_user', 'country_mismatch', 'high_risk_distance',
        'zero_distance', 'new_account_with_promo', 'late_night_micro_transaction',
        'high_value_long_distance'
    ]

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

    # Create preprocessing pipeline (matches notebook approach)
    preprocessor = create_preprocessing_pipeline(categorical_features, continuous_numeric, binary)

    # Feature names after preprocessing (order matches notebook)
    feature_names = categorical_features + continuous_numeric + binary

    # Load optimal hyperparameters from previous training if available
    metadata_path = Path("models/model_metadata.json")
    if metadata_path.exists():
        print("\nLoading optimal hyperparameters from previous training...")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            loaded_params = metadata["hyperparameters"]
            # Extract only the model parameters (exclude random_state which may differ)
            optimal_params = {
                "n_estimators": loaded_params["n_estimators"],
                "max_depth": loaded_params["max_depth"],
                "learning_rate": loaded_params["learning_rate"],
                "subsample": loaded_params["subsample"],
                "colsample_bytree": loaded_params["colsample_bytree"],
                "min_child_weight": loaded_params["min_child_weight"],
                "gamma": loaded_params["gamma"],
                "scale_pos_weight": loaded_params["scale_pos_weight"],
                "eval_metric": "aucpr",
                "random_state": random_seed,
                "n_jobs": -1,
            }
        print(f"  ✓ Loaded hyperparameters from {metadata_path}")
    else:
        # Fallback to optimal hyperparameters from initial notebook tuning
        print("\nUsing optimal hyperparameters from initial notebook tuning...")
        optimal_params = {
            "n_estimators": 90,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 5,
            "gamma": 0.6,
            "scale_pos_weight": 8,
            "eval_metric": "aucpr",
            "random_state": random_seed,
            "n_jobs": -1,
        }

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
            cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=random_seed),
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
        "all_features": results["feature_names"],
        "categorical_features": results["categorical_features"],
        "continuous_features": [
            f for f in results["feature_names"] if f not in results["categorical_features"]
        ],
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
        # Load raw data and split
        data_dir = Path(args.data_dir)
        train_raw, val_raw, test_raw = load_data(data_dir, random_seed=args.random_seed)

        # Apply feature engineering pipeline
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
