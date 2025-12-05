"""Tests for model_comparison module."""

import pytest
import pandas as pd
import numpy as np

from src.fd2_nb.model_comparison import compare_models, get_best_model


class TestCompareModels:
    """Tests for compare_models function."""

    @pytest.fixture
    def sample_metrics(self):
        """Create sample model metrics for testing."""
        return [
            {
                'model': 'Logistic Regression',
                'pr_auc': 0.75,
                'roc_auc': 0.85,
                'precision': 0.70,
                'recall': 0.65,
                'f1': 0.67,
                'accuracy': 0.90
            },
            {
                'model': 'Random Forest',
                'pr_auc': 0.82,
                'roc_auc': 0.90,
                'precision': 0.78,
                'recall': 0.75,
                'f1': 0.76,
                'accuracy': 0.92
            },
            {
                'model': 'XGBoost',
                'pr_auc': 0.85,
                'roc_auc': 0.92,
                'precision': 0.80,
                'recall': 0.78,
                'f1': 0.79,
                'accuracy': 0.93
            }
        ]

    def test_compare_models_returns_dataframe(self, sample_metrics):
        """Test that compare_models returns a DataFrame."""
        result = compare_models(sample_metrics, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_compare_models_correct_shape(self, sample_metrics):
        """Test that result has correct number of rows."""
        result = compare_models(sample_metrics, verbose=False)
        assert len(result) == 3

    def test_compare_models_model_as_index(self, sample_metrics):
        """Test that model names are used as index."""
        result = compare_models(sample_metrics, verbose=False)
        assert 'Logistic Regression' in result.index
        assert 'Random Forest' in result.index
        assert 'XGBoost' in result.index

    def test_compare_models_drops_dataset_column(self, sample_metrics):
        """Test that dataset column is dropped if present."""
        metrics_with_dataset = [
            {**m, 'dataset': 'Validation'} for m in sample_metrics
        ]
        result = compare_models(metrics_with_dataset, verbose=False)
        assert 'dataset' not in result.columns

    def test_compare_models_filters_metrics(self, sample_metrics):
        """Test that metrics_to_display filters columns."""
        result = compare_models(
            sample_metrics,
            metrics_to_display=['pr_auc', 'roc_auc'],
            verbose=False
        )
        assert list(result.columns) == ['pr_auc', 'roc_auc']

    def test_compare_models_metric_values(self, sample_metrics):
        """Test that metric values are correct."""
        result = compare_models(sample_metrics, verbose=False)
        assert result.loc['XGBoost', 'pr_auc'] == 0.85
        assert result.loc['Logistic Regression', 'roc_auc'] == 0.85


class TestGetBestModel:
    """Tests for get_best_model function."""

    @pytest.fixture
    def comparison_df(self):
        """Create sample comparison DataFrame."""
        return pd.DataFrame({
            'pr_auc': [0.75, 0.82, 0.85],
            'roc_auc': [0.85, 0.90, 0.92],
            'f1': [0.67, 0.76, 0.79]
        }, index=['Logistic Regression', 'Random Forest', 'XGBoost'])

    def test_get_best_model_returns_tuple(self, comparison_df):
        """Test that get_best_model returns a tuple."""
        result = get_best_model(comparison_df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_best_model_correct_name(self, comparison_df):
        """Test that best model name is correct."""
        name, _ = get_best_model(comparison_df, 'pr_auc')
        assert name == 'XGBoost'

    def test_get_best_model_returns_metrics_dict(self, comparison_df):
        """Test that metrics dict contains all metrics."""
        _, metrics = get_best_model(comparison_df)
        assert 'pr_auc' in metrics
        assert 'roc_auc' in metrics
        assert 'f1' in metrics

    def test_get_best_model_different_metric(self, comparison_df):
        """Test with different primary metric."""
        name, metrics = get_best_model(comparison_df, 'roc_auc')
        assert name == 'XGBoost'
        assert metrics['roc_auc'] == 0.92

    def test_get_best_model_invalid_metric_raises(self, comparison_df):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            get_best_model(comparison_df, 'invalid_metric')
