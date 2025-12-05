"""
Tests for src/deployment/evaluation/metrics.py

Tests calculate_metrics and evaluate_model functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.deployment.evaluation.metrics import calculate_metrics, evaluate_model


@pytest.fixture
def sample_binary_classification_data():
    """Create sample binary classification data for testing"""
    np.random.seed(42)

    # Generate synthetic data: 100 samples, 20% positive class
    n_samples = 100
    y_true = np.array([0] * 80 + [1] * 20)

    # Generate realistic predictions
    # For true positives: higher probabilities
    # For true negatives: lower probabilities
    y_proba = np.zeros(n_samples)
    y_proba[:80] = np.random.beta(2, 5, 80)  # Negatives: lower probs
    y_proba[80:] = np.random.beta(5, 2, 20)  # Positives: higher probs

    # Binary predictions at 0.5 threshold
    y_pred = (y_proba >= 0.5).astype(int)

    return y_true, y_pred, y_proba


@pytest.fixture
def mock_model(sample_binary_classification_data):
    """Create a mock model for testing"""
    y_true, y_pred, y_proba = sample_binary_classification_data

    model = Mock()
    model.predict.return_value = y_pred
    model.predict_proba.return_value = np.column_stack([1 - y_proba, y_proba])

    return model


class TestCalculateMetrics:
    """Tests for calculate_metrics function"""

    def test_calculate_metrics_returns_dict(self, sample_binary_classification_data):
        """Test that calculate_metrics returns a dictionary"""
        y_true, y_pred, y_proba = sample_binary_classification_data
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        assert isinstance(metrics, dict)

    def test_calculate_metrics_has_expected_keys(self, sample_binary_classification_data):
        """Test that metrics dict has expected keys"""
        y_true, y_pred, y_proba = sample_binary_classification_data
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        expected_keys = ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy']
        assert set(metrics.keys()) == set(expected_keys)

    def test_calculate_metrics_values_in_valid_range(self, sample_binary_classification_data):
        """Test that all metric values are in valid range [0, 1]"""
        y_true, y_pred, y_proba = sample_binary_classification_data
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"{metric_name} = {value} is out of range [0, 1]"

    def test_calculate_metrics_values_are_floats(self, sample_binary_classification_data):
        """Test that all metric values are floats"""
        y_true, y_pred, y_proba = sample_binary_classification_data
        metrics = calculate_metrics(y_true, y_pred, y_proba)

        for metric_name, value in metrics.items():
            assert isinstance(value, (float, np.floating)), \
                f"{metric_name} = {value} is not a float"

    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # Perfect predictions should give perfect scores
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1'] == 1.0
        assert metrics['roc_auc'] == 1.0
        assert metrics['pr_auc'] == 1.0

    def test_calculate_metrics_all_negative_predictions(self):
        """Test metrics when predicting all negative"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])

        metrics = calculate_metrics(y_true, y_pred, y_proba)

        # All negative predictions: recall should be 0
        assert metrics['recall'] == 0.0


class TestEvaluateModel:
    """Tests for evaluate_model function"""

    def test_evaluate_model_returns_dict(self, mock_model, sample_binary_classification_data):
        """Test that evaluate_model returns a dictionary"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)  # Dummy features

        metrics = evaluate_model(mock_model, X, y_true, verbose=False)

        assert isinstance(metrics, dict)

    def test_evaluate_model_has_expected_keys(self, mock_model, sample_binary_classification_data):
        """Test that metrics dict has expected keys"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        metrics = evaluate_model(mock_model, X, y_true, verbose=False)

        expected_keys = ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1', 'accuracy']
        assert set(metrics.keys()) == set(expected_keys)

    def test_evaluate_model_calls_predict(self, mock_model, sample_binary_classification_data):
        """Test that evaluate_model calls model.predict()"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        evaluate_model(mock_model, X, y_true, verbose=False)

        mock_model.predict.assert_called_once()

    def test_evaluate_model_calls_predict_proba(self, mock_model, sample_binary_classification_data):
        """Test that evaluate_model calls model.predict_proba()"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        evaluate_model(mock_model, X, y_true, verbose=False)

        mock_model.predict_proba.assert_called_once()

    def test_evaluate_model_verbose_false_no_print(self, mock_model,
                                                     sample_binary_classification_data,
                                                     capsys):
        """Test that verbose=False doesn't print output"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        evaluate_model(mock_model, X, y_true, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_evaluate_model_verbose_true_prints_info(self, mock_model,
                                                      sample_binary_classification_data,
                                                      capsys):
        """Test that verbose=True prints evaluation information"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        evaluate_model(mock_model, X, y_true, model_name="TestModel",
                      dataset_name="TestData", verbose=True)

        captured = capsys.readouterr()
        assert "TestModel" in captured.out
        assert "TestData" in captured.out
        assert "PR-AUC:" in captured.out
        assert "ROC-AUC:" in captured.out
        assert "Precision:" in captured.out
        assert "Recall:" in captured.out
        assert "F1 Score:" in captured.out
        assert "Confusion Matrix:" in captured.out

    def test_evaluate_model_custom_names(self, mock_model,
                                         sample_binary_classification_data,
                                         capsys):
        """Test that custom model_name and dataset_name are used in output"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        evaluate_model(
            mock_model, X, y_true,
            model_name="CustomModel",
            dataset_name="CustomDataset",
            verbose=True
        )

        captured = capsys.readouterr()
        assert "CustomModel" in captured.out
        assert "CustomDataset" in captured.out

    def test_evaluate_model_default_names(self, mock_model,
                                          sample_binary_classification_data,
                                          capsys):
        """Test that default names are used when not provided"""
        y_true, _, _ = sample_binary_classification_data
        X = np.random.randn(len(y_true), 5)

        evaluate_model(mock_model, X, y_true, verbose=True)

        captured = capsys.readouterr()
        assert "Model" in captured.out
        assert "Dataset" in captured.out
