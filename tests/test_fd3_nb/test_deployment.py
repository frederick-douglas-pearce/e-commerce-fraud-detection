"""Tests for deployment module."""

import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fd3_nb.deployment import (
    save_production_model,
    save_threshold_config,
    save_model_metadata,
)


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)
    n_samples = 100

    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
    })
    y = pd.Series(np.random.randint(0, 2, n_samples))

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    model.fit(X, y)

    return model


class TestSaveProductionModel:
    """Tests for save_production_model function."""

    def test_returns_path(self, sample_model, tmp_path):
        """Test that save_production_model returns a Path."""
        result = save_production_model(sample_model, tmp_path)
        assert isinstance(result, Path)

    def test_creates_file(self, sample_model, tmp_path):
        """Test that model file is created."""
        result = save_production_model(sample_model, tmp_path)
        assert result.exists()

    def test_file_has_correct_name(self, sample_model, tmp_path):
        """Test that file has correct default name."""
        result = save_production_model(sample_model, tmp_path)
        assert result.name == 'xgb_fraud_detector.joblib'

    def test_custom_filename(self, sample_model, tmp_path):
        """Test with custom filename."""
        result = save_production_model(
            sample_model, tmp_path, filename='custom_model.joblib'
        )
        assert result.name == 'custom_model.joblib'

    def test_creates_parent_directory(self, sample_model, tmp_path):
        """Test that parent directories are created."""
        nested_path = tmp_path / 'nested' / 'model' / 'dir'
        result = save_production_model(sample_model, nested_path)
        assert result.exists()
        assert result.parent.exists()

    def test_model_can_be_loaded(self, sample_model, tmp_path):
        """Test that saved model can be loaded and used."""
        import joblib
        model_path = save_production_model(sample_model, tmp_path)
        loaded_model = joblib.load(model_path)

        # Create test data
        X_test = pd.DataFrame({
            'feature1': [0.5, -0.5],
            'feature2': [0.3, -0.3]
        })

        # Should be able to predict
        predictions = loaded_model.predict(X_test)
        assert len(predictions) == 2

    def test_file_size_positive(self, sample_model, tmp_path):
        """Test that saved file has positive size."""
        result = save_production_model(sample_model, tmp_path)
        assert result.stat().st_size > 0


class TestSaveThresholdConfig:
    """Tests for save_threshold_config function."""

    @pytest.fixture
    def sample_threshold_results(self):
        """Create sample threshold results."""
        optimal_f1_result = {
            'threshold': 0.45,
            'precision': 0.72,
            'recall': 0.85,
            'f1': 0.78,
        }

        target_performance_result = {
            'threshold': 0.35,
            'precision': 0.70,
            'recall': 0.88,
            'f1': 0.78,
            'min_precision': 0.70,
        }

        threshold_results = [
            {'threshold': 0.30, 'precision': 0.65, 'recall': 0.90, 'f1': 0.75},
            {'threshold': 0.40, 'precision': 0.70, 'recall': 0.86, 'f1': 0.77},
            {'threshold': 0.50, 'precision': 0.75, 'recall': 0.81, 'f1': 0.78},
        ]

        return optimal_f1_result, target_performance_result, threshold_results

    def test_returns_path(self, sample_threshold_results, tmp_path):
        """Test that save_threshold_config returns a Path."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        result = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )
        assert isinstance(result, Path)

    def test_creates_file(self, sample_threshold_results, tmp_path):
        """Test that config file is created."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        result = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )
        assert result.exists()

    def test_file_is_valid_json(self, sample_threshold_results, tmp_path):
        """Test that saved file is valid JSON."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        config_path = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )

        with open(config_path, 'r') as f:
            config = json.load(f)

        assert isinstance(config, dict)

    def test_contains_required_keys(self, sample_threshold_results, tmp_path):
        """Test that config contains required keys."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        config_path = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )

        with open(config_path, 'r') as f:
            config = json.load(f)

        assert 'default_threshold' in config
        assert 'recommended_threshold' in config
        assert 'optimized_thresholds' in config

    def test_contains_optimal_f1_threshold(self, sample_threshold_results, tmp_path):
        """Test that config contains optimal_f1 threshold."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        config_path = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )

        with open(config_path, 'r') as f:
            config = json.load(f)

        assert 'optimal_f1' in config['optimized_thresholds']
        assert config['optimized_thresholds']['optimal_f1']['threshold'] == 0.45

    def test_contains_target_performance_threshold(self, sample_threshold_results, tmp_path):
        """Test that config contains target_performance threshold when provided."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        config_path = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )

        with open(config_path, 'r') as f:
            config = json.load(f)

        assert 'target_performance' in config['optimized_thresholds']
        assert config['optimized_thresholds']['target_performance']['threshold'] == 0.35

    def test_handles_none_target_performance(self, sample_threshold_results, tmp_path):
        """Test that config handles None target_performance_result."""
        optimal_f1, _, threshold_results = sample_threshold_results
        config_path = save_threshold_config(
            optimal_f1, None, threshold_results, tmp_path
        )

        with open(config_path, 'r') as f:
            config = json.load(f)

        assert 'target_performance' not in config['optimized_thresholds']
        assert config['recommended_threshold'] == 'optimal_f1'

    def test_contains_recall_targeted_thresholds(self, sample_threshold_results, tmp_path):
        """Test that config contains recall-targeted thresholds."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        config_path = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path
        )

        with open(config_path, 'r') as f:
            config = json.load(f)

        thresholds = config['optimized_thresholds']
        assert 'conservative_90pct_recall' in thresholds
        assert 'balanced_85pct_recall' in thresholds
        assert 'aggressive_80pct_recall' in thresholds

    def test_custom_filename(self, sample_threshold_results, tmp_path):
        """Test with custom filename."""
        optimal_f1, target_perf, threshold_results = sample_threshold_results
        result = save_threshold_config(
            optimal_f1, target_perf, threshold_results, tmp_path,
            filename='custom_thresholds.json'
        )
        assert result.name == 'custom_thresholds.json'


class TestSaveModelMetadata:
    """Tests for save_model_metadata function."""

    @pytest.fixture
    def sample_metadata_inputs(self):
        """Create sample inputs for save_model_metadata."""
        best_params = {
            'xgboost': {
                'classifier__n_estimators': 100,
                'classifier__max_depth': 4,
                'classifier__learning_rate': 0.1,
                'classifier__subsample': 0.9,
                'classifier__colsample_bytree': 0.9,
                'classifier__min_child_weight': 7,
                'classifier__gamma': 0.7,
                'classifier__reg_alpha': 0.1,
                'classifier__reg_lambda': 1.0,
                'classifier__scale_pos_weight': 8,
            }
        }

        validation_metrics = {
            'xgboost_tuned': {
                'pr_auc': 0.87,
                'roc_auc': 0.96,
                'f1': 0.75,
            },
            'cv_folds': 4
        }

        test_metrics = {
            'roc_auc': 0.95,
            'pr_auc': 0.86,
            'f1': 0.74,
            'precision': 0.72,
            'recall': 0.76,
            'accuracy': 0.98,
        }

        feature_lists = {
            'continuous_numeric': ['feature1', 'feature2', 'feature3'],
            'categorical': ['cat1'],
            'binary': ['bin1', 'bin2'],
        }

        dataset_sizes = {
            'train': 100000,
            'val': 30000,
            'test': 50000,
        }

        fraud_rates = {
            'train': 0.022,
            'test': 0.021,
        }

        return (best_params, validation_metrics, test_metrics,
                feature_lists, dataset_sizes, fraud_rates)

    def test_returns_path(self, sample_metadata_inputs, tmp_path):
        """Test that save_model_metadata returns a Path."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        result = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )
        assert isinstance(result, Path)

    def test_creates_file(self, sample_metadata_inputs, tmp_path):
        """Test that metadata file is created."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        result = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )
        assert result.exists()

    def test_file_is_valid_json(self, sample_metadata_inputs, tmp_path):
        """Test that saved file is valid JSON."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert isinstance(metadata, dict)

    def test_contains_model_info(self, sample_metadata_inputs, tmp_path):
        """Test that metadata contains model_info section."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'model_info' in metadata
        assert 'model_name' in metadata['model_info']
        assert 'version' in metadata['model_info']
        assert 'training_date' in metadata['model_info']

    def test_contains_hyperparameters(self, sample_metadata_inputs, tmp_path):
        """Test that metadata contains hyperparameters section."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'hyperparameters' in metadata
        assert metadata['hyperparameters']['n_estimators'] == 100
        assert metadata['hyperparameters']['max_depth'] == 4

    def test_contains_dataset_info(self, sample_metadata_inputs, tmp_path):
        """Test that metadata contains dataset_info section."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'dataset_info' in metadata
        assert metadata['dataset_info']['test_samples'] == 50000

    def test_contains_performance_section(self, sample_metadata_inputs, tmp_path):
        """Test that metadata contains performance section."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'performance' in metadata
        assert 'test_set' in metadata['performance']
        assert 'cross_validation' in metadata['performance']

    def test_contains_features_section(self, sample_metadata_inputs, tmp_path):
        """Test that metadata contains features section."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert 'features' in metadata
        assert 'continuous_numeric' in metadata['features']
        assert 'categorical' in metadata['features']
        assert 'binary' in metadata['features']

    def test_custom_random_seed(self, sample_metadata_inputs, tmp_path):
        """Test with custom random seed."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        metadata_path = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path,
            random_seed=42
        )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        assert metadata['hyperparameters']['random_state'] == 42

    def test_custom_filename(self, sample_metadata_inputs, tmp_path):
        """Test with custom filename."""
        (best_params, validation_metrics, test_metrics,
         feature_lists, dataset_sizes, fraud_rates) = sample_metadata_inputs

        result = save_model_metadata(
            best_params, validation_metrics, test_metrics,
            feature_lists, dataset_sizes, fraud_rates, tmp_path,
            filename='custom_metadata.json'
        )
        assert result.name == 'custom_metadata.json'
