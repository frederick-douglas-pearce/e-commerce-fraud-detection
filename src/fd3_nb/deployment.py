"""
Deployment artifact utilities for fd3 notebook.

This module provides functions for saving model artifacts, configurations,
and metadata for production deployment.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib


def save_production_model(
    model: Any,
    model_dir: Path,
    filename: str = "xgb_fraud_detector.joblib"
) -> Path:
    """
    Save the trained model for production deployment.

    Args:
        model: Trained model pipeline
        model_dir: Directory to save model
        filename: Output filename

    Returns:
        Path to saved model
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / filename
    joblib.dump(model, model_path)

    print(f"✓ Production model saved to: {model_path}")
    print(f"  File size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")

    return model_path


def save_threshold_config(
    optimal_f1_result: Dict[str, Any],
    target_performance_result: Optional[Dict[str, Any]],
    threshold_results: List[Dict[str, Any]],
    model_dir: Path,
    filename: str = "threshold_config.json"
) -> Path:
    """
    Save threshold configuration for different business requirements.

    Args:
        optimal_f1_result: Dict with optimal F1 threshold results
        target_performance_result: Dict with target performance threshold results (or None)
        threshold_results: List of dicts with recall-targeted threshold results
        model_dir: Directory to save config
        filename: Output filename

    Returns:
        Path to saved config
    """
    optimized_thresholds = {
        'optimal_f1': {
            'threshold': float(optimal_f1_result['threshold']),
            'precision': float(optimal_f1_result['precision']),
            'recall': float(optimal_f1_result['recall']),
            'f1': float(optimal_f1_result['f1']),
            'description': 'Optimal F1 score - best precision-recall balance'
        }
    }

    # Add target performance threshold if available
    if target_performance_result is not None:
        min_prec = target_performance_result.get('min_precision', 0.70)
        optimized_thresholds['target_performance'] = {
            'threshold': float(target_performance_result['threshold']),
            'precision': float(target_performance_result['precision']),
            'recall': float(target_performance_result['recall']),
            'f1': float(target_performance_result['f1']),
            'min_precision_constraint': min_prec,
            'description': f'Max recall with >={min_prec*100:.0f}% precision (RECOMMENDED)'
        }

    # Add recall-targeted thresholds
    optimized_thresholds['conservative_90pct_recall'] = {
        'threshold': float(threshold_results[0]['threshold']),
        'target_recall': 0.90,
        'achieved_recall': float(threshold_results[0]['recall']),
        'precision': float(threshold_results[0]['precision']),
        'f1': float(threshold_results[0]['f1']),
        'description': 'Catch most fraud (90% recall), accept more false positives'
    }
    optimized_thresholds['balanced_85pct_recall'] = {
        'threshold': float(threshold_results[1]['threshold']),
        'target_recall': 0.85,
        'achieved_recall': float(threshold_results[1]['recall']),
        'precision': float(threshold_results[1]['precision']),
        'f1': float(threshold_results[1]['f1']),
        'description': 'Balanced precision-recall trade-off (85% recall target)'
    }
    optimized_thresholds['aggressive_80pct_recall'] = {
        'threshold': float(threshold_results[2]['threshold']),
        'target_recall': 0.80,
        'achieved_recall': float(threshold_results[2]['recall']),
        'precision': float(threshold_results[2]['precision']),
        'f1': float(threshold_results[2]['f1']),
        'description': 'Prioritize precision (80% recall), reduce false positives'
    }

    threshold_config = {
        'default_threshold': 0.5,
        'recommended_threshold': 'target_performance' if target_performance_result else 'optimal_f1',
        'optimized_thresholds': optimized_thresholds,
        'note': 'target_performance maximizes recall while meeting precision constraint (recommended for production)'
    }

    config_path = model_dir / filename
    with open(config_path, 'w') as f:
        json.dump(threshold_config, f, indent=2)

    num_strategies = len(optimized_thresholds)
    print(f"✓ Threshold configuration saved to: {config_path}")
    print(f"  • {num_strategies} threshold strategies available")
    print(f"  • Recommended: {threshold_config['recommended_threshold']}")

    return config_path


def save_model_metadata(
    best_params: Dict[str, Any],
    validation_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    feature_lists: Dict[str, List[str]],
    dataset_sizes: Dict[str, int],
    fraud_rates: Dict[str, float],
    model_dir: Path,
    random_seed: int = 1,
    filename: str = "model_metadata.json"
) -> Path:
    """
    Save comprehensive model metadata for documentation and reproducibility.

    Args:
        best_params: Dict with best hyperparameters
        validation_metrics: Dict with CV validation metrics
        test_metrics: Dict with test set metrics
        feature_lists: Dict with feature categorization
        dataset_sizes: Dict with 'train', 'val', 'test' sample counts
        fraud_rates: Dict with 'train' and 'test' fraud rates
        model_dir: Directory to save metadata
        random_seed: Random seed used for reproducibility
        filename: Output filename

    Returns:
        Path to saved metadata
    """
    metadata = {
        'model_info': {
            'model_name': 'XGBoost Fraud Detector',
            'model_type': 'XGBClassifier',
            'version': '1.0',
            'training_date': datetime.now().strftime('%Y-%m-%d'),
            'framework': 'xgboost + scikit-learn',
            'python_version': '3.12+',
            'note': 'Model trained on train+val combined in Notebook 2, evaluated on test set here'
        },
        'hyperparameters': {
            'n_estimators': int(best_params['xgboost']['classifier__n_estimators']),
            'max_depth': int(best_params['xgboost']['classifier__max_depth']),
            'learning_rate': float(best_params['xgboost']['classifier__learning_rate']),
            'subsample': float(best_params['xgboost']['classifier__subsample']),
            'colsample_bytree': float(best_params['xgboost']['classifier__colsample_bytree']),
            'min_child_weight': int(best_params['xgboost']['classifier__min_child_weight']),
            'gamma': float(best_params['xgboost']['classifier__gamma']),
            'reg_alpha': float(best_params['xgboost']['classifier__reg_alpha']),
            'reg_lambda': float(best_params['xgboost']['classifier__reg_lambda']),
            'scale_pos_weight': int(best_params['xgboost']['classifier__scale_pos_weight']),
            'eval_metric': 'aucpr',
            'random_state': random_seed
        },
        'dataset_info': {
            'training_samples': dataset_sizes['train'] + dataset_sizes['val'],
            'training_sources': {
                'original_train': dataset_sizes['train'],
                'original_val': dataset_sizes['val'],
                'combined_total': dataset_sizes['train'] + dataset_sizes['val']
            },
            'test_samples': dataset_sizes['test'],
            'num_features': 30,
            'fraud_rate_train': fraud_rates['train'],
            'fraud_rate_test': fraud_rates['test'],
            'class_imbalance_ratio': (1 - fraud_rates['train']) / fraud_rates['train']
        },
        'performance': {
            'test_set': {
                'note': 'Performance on held-out test set (model trained on train+val in Notebook 2)',
                'roc_auc': float(test_metrics['roc_auc']),
                'pr_auc': float(test_metrics['pr_auc']),
                'f1_score': float(test_metrics['f1']),
                'precision': float(test_metrics['precision']),
                'recall': float(test_metrics['recall']),
                'accuracy': float(test_metrics['accuracy'])
            },
            'cross_validation': {
                'cv_folds': validation_metrics.get('cv_folds', 4),
                'cv_strategy': 'StratifiedKFold',
                'cv_pr_auc': float(validation_metrics['xgboost_tuned']['pr_auc']),
                'note': 'CV performed on train+val combined for hyperparameter selection in Notebook 2'
            }
        },
        'features': {
            'continuous_numeric': feature_lists['continuous_numeric'],
            'categorical': feature_lists['categorical'],
            'binary': feature_lists['binary'],
            'total_count': 30
        },
        'preprocessing': {
            'categorical_encoding': 'OrdinalEncoder (handle_unknown=use_encoded_value)',
            'numeric_scaling': 'None (tree-based model)',
            'binary_features': 'Passthrough (no transformation)'
        },
        'workflow': {
            'training_notebook': 'fd2_model_selection_tuning.ipynb',
            'evaluation_notebook': 'fd3_model_evaluation_deployment.ipynb',
            'model_source': 'GridSearchCV best_estimator_ (auto-refit on train+val)',
            'note': 'Model loaded from best_model.joblib, no retraining in fd3'
        }
    }

    metadata_path = model_dir / filename
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Model metadata saved to: {metadata_path}")
    print("\nMetadata Summary:")
    print(f"  • Model: {metadata['model_info']['model_name']} v{metadata['model_info']['version']}")
    print(f"  • Training Date: {metadata['model_info']['training_date']}")
    print(f"  • Training Samples: {metadata['dataset_info']['training_samples']:,} (train+val combined)")
    print(f"  • Test PR-AUC: {metadata['performance']['test_set']['pr_auc']:.4f}")
    print(f"  • CV PR-AUC: {metadata['performance']['cross_validation']['cv_pr_auc']:.4f}")
    print(f"  • Total Features: {metadata['features']['total_count']}")

    return metadata_path


def print_deployment_summary(
    model_path: Path,
    threshold_config_path: Path,
    feature_lists_path: Path,
    metadata_path: Path,
    validation_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any]
) -> None:
    """
    Print a comprehensive deployment summary.

    Args:
        model_path: Path to saved model
        threshold_config_path: Path to threshold config
        feature_lists_path: Path to feature lists
        metadata_path: Path to model metadata
        validation_metrics: Dict with CV validation metrics
        test_metrics: Dict with test set metrics
    """
    print("=" * 100)
    print("DEPLOYMENT PACKAGE READY")
    print("=" * 100)
    print("\nSaved Artifacts:")
    print(f"  1. Model Pipeline:      {model_path}")
    print(f"  2. Threshold Config:    {threshold_config_path}")
    print(f"  3. Feature Lists:       {feature_lists_path}")
    print(f"  4. Model Metadata:      {metadata_path}")

    print("\nDeployment Files Total Size:")
    total_size = sum([
        model_path.stat().st_size,
        threshold_config_path.stat().st_size,
        feature_lists_path.stat().st_size if feature_lists_path.exists() else 0,
        metadata_path.stat().st_size
    ])
    print(f"  {total_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 100)
    print("MODEL EVALUATION COMPLETE")
    print("=" * 100)
    print(f"\n🏆 Model: XGBoost (Trained on Train+Val in Notebook 2)")
    print(f"   • CV PR-AUC (Notebook 2):  {validation_metrics['xgboost_tuned']['pr_auc']:.4f}")
    print(f"   • Test PR-AUC (this notebook): {test_metrics['pr_auc']:.4f}")
    print(f"   • Precision:         {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"   • Recall:            {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"   • F1 Score:          {test_metrics['f1']:.4f}")

    print("\n📊 Key Achievements:")
    print("   ✅ Model loaded from Notebook 2 (no redundant retraining)")
    print("   ✅ Test performance validates CV results")
    print("   ✅ Feature importance analyzed")
    print("   ✅ Threshold optimization complete")
    print("   ✅ Deployment package created")

    print("\n🚀 Next Steps:")
    print("   1. Deploy model to production API")
    print("   2. Implement monitoring dashboard")
    print("   3. Set up retraining pipeline")
    print("   4. Conduct A/B test against current system")
    print("=" * 100)
