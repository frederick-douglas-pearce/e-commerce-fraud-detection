"""Tests for evaluation module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fd3_nb.evaluation import evaluate_model, compare_val_test_performance


@pytest.fixture
def sample_model():
    """Create a simple trained model for testing."""
    np.random.seed(42)
    n_samples = 500

    # Create imbalanced dataset (10% positive)
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    })
    y = pd.Series(np.zeros(n_samples))
    y.iloc[:50] = 1
    y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    # Make features predictive of target
    X.loc[y == 1, 'feature1'] += 1.5
    X.loc[y == 1, 'feature2'] += 1.0

    # Create and train pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    model.fit(X, y)

    return model, X, y


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_dict(self, sample_model):
        """Test that evaluate_model returns a dictionary."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, verbose=False)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_model):
        """Test that result contains required keys."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, verbose=False)
        required_keys = ['model', 'dataset', 'roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'accuracy']
        for key in required_keys:
            assert key in result

    def test_metrics_in_valid_range(self, sample_model):
        """Test that all metrics are in [0, 1] range."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, verbose=False)
        metric_keys = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall', 'accuracy']
        for key in metric_keys:
            assert 0 <= result[key] <= 1, f"{key} should be in [0, 1]"

    def test_model_name_stored(self, sample_model):
        """Test that model name is stored correctly."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, model_name='Test Model', verbose=False)
        assert result['model'] == 'Test Model'

    def test_dataset_name_stored(self, sample_model):
        """Test that dataset name is stored correctly."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, dataset_name='Test Set', verbose=False)
        assert result['dataset'] == 'Test Set'

    def test_verbose_mode_runs(self, sample_model):
        """Test that verbose mode runs without error."""
        model, X, y = sample_model
        # Should not raise an exception
        evaluate_model(model, X, y, verbose=True)

    def test_roc_auc_reasonable(self, sample_model):
        """Test that ROC-AUC is reasonable for the predictive model."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, verbose=False)
        # Model should perform better than random (0.5)
        assert result['roc_auc'] > 0.5

    def test_accuracy_calculation(self, sample_model):
        """Test that accuracy is calculated correctly."""
        model, X, y = sample_model
        result = evaluate_model(model, X, y, verbose=False)
        # Verify accuracy matches manual calculation
        y_pred = model.predict(X)
        expected_accuracy = (y_pred == y).mean()
        assert abs(result['accuracy'] - expected_accuracy) < 0.001


class TestCompareValTestPerformance:
    """Tests for compare_val_test_performance function."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample validation and test metrics."""
        validation_metrics = {
            'xgboost_tuned': {
                'roc_auc': 0.95,
                'pr_auc': 0.85,
                'f1': 0.75,
                'precision': 0.72,
                'recall': 0.78
            }
        }
        test_metrics = {
            'model': 'XGBoost',
            'dataset': 'Test',
            'roc_auc': 0.94,
            'pr_auc': 0.84,
            'f1': 0.74,
            'precision': 0.71,
            'recall': 0.77
        }
        return validation_metrics, test_metrics

    def test_returns_dataframe(self, sample_metrics):
        """Test that compare_val_test_performance returns a DataFrame."""
        validation_metrics, test_metrics = sample_metrics
        result = compare_val_test_performance(
            validation_metrics, test_metrics, verbose=False
        )
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_two_rows(self, sample_metrics):
        """Test that DataFrame has rows for both val and test."""
        validation_metrics, test_metrics = sample_metrics
        result = compare_val_test_performance(
            validation_metrics, test_metrics, verbose=False
        )
        assert len(result) == 2

    def test_index_contains_datasets(self, sample_metrics):
        """Test that index contains dataset names."""
        validation_metrics, test_metrics = sample_metrics
        result = compare_val_test_performance(
            validation_metrics, test_metrics, verbose=False
        )
        assert 'CV Validation' in result.index
        assert 'Test' in result.index

    def test_contains_metric_columns(self, sample_metrics):
        """Test that DataFrame contains metric columns."""
        validation_metrics, test_metrics = sample_metrics
        result = compare_val_test_performance(
            validation_metrics, test_metrics, verbose=False
        )
        metric_cols = ['roc_auc', 'pr_auc', 'f1', 'precision', 'recall']
        for col in metric_cols:
            assert col in result.columns

    def test_custom_val_key(self, sample_metrics):
        """Test with custom validation key."""
        validation_metrics = {
            'custom_model': {
                'roc_auc': 0.92,
                'pr_auc': 0.82,
                'f1': 0.72,
                'precision': 0.70,
                'recall': 0.75
            }
        }
        _, test_metrics = sample_metrics
        result = compare_val_test_performance(
            validation_metrics, test_metrics, val_key='custom_model', verbose=False
        )
        assert len(result) == 2
        assert result.loc['CV Validation', 'roc_auc'] == 0.92

    def test_verbose_mode_runs(self, sample_metrics):
        """Test that verbose mode runs without error."""
        validation_metrics, test_metrics = sample_metrics
        # Should not raise an exception
        compare_val_test_performance(validation_metrics, test_metrics, verbose=True)

    def test_model_column_dropped(self, sample_metrics):
        """Test that model column is dropped from result."""
        validation_metrics, test_metrics = sample_metrics
        result = compare_val_test_performance(
            validation_metrics, test_metrics, verbose=False
        )
        assert 'model' not in result.columns
