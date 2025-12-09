"""Tests for visualization module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.fd3_nb.visualization import plot_shap_beeswarm


@pytest.fixture
def sample_shap_data():
    """Create sample SHAP values and feature data for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Create feature names
    feature_names = [f'feature_{i}' for i in range(n_features)]

    # Create SHAP values matrix
    shap_values = np.random.randn(n_samples, n_features)

    # Create corresponding feature DataFrame with numeric values
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )

    return shap_values, X, feature_names


@pytest.fixture
def sample_shap_data_with_categorical():
    """Create sample SHAP values with categorical feature for testing."""
    np.random.seed(42)
    n_samples = 100
    n_numeric_features = 8

    # Create feature names
    numeric_cols = [f'num_feature_{i}' for i in range(n_numeric_features)]
    feature_names = ['category'] + numeric_cols

    # Create SHAP values matrix (9 features total)
    shap_values = np.random.randn(n_samples, len(feature_names))

    # Create corresponding feature DataFrame
    X_numeric = pd.DataFrame(
        np.random.randn(n_samples, n_numeric_features),
        columns=numeric_cols
    )
    X_cat = pd.DataFrame({
        'category': np.random.choice(['A', 'B', 'C'], n_samples)
    })
    X = pd.concat([X_cat, X_numeric], axis=1)

    return shap_values, X, feature_names


class TestPlotShapBeeswarm:
    """Tests for plot_shap_beeswarm function."""

    def test_runs_without_error(self, sample_shap_data):
        """Test that plot_shap_beeswarm runs without error."""
        shap_values, X, feature_names = sample_shap_data
        # Should not raise an exception
        plot_shap_beeswarm(shap_values, X, feature_names, top_n=5)

    def test_handles_categorical_features(self, sample_shap_data_with_categorical):
        """Test that function correctly handles string/categorical columns."""
        shap_values, X, feature_names = sample_shap_data_with_categorical
        # Should not raise an exception (previously failed on string columns)
        plot_shap_beeswarm(shap_values, X, feature_names, top_n=5)

    def test_handles_numeric_features(self, sample_shap_data):
        """Test that function works with continuous numeric values."""
        shap_values, X, feature_names = sample_shap_data
        # Should not raise an exception
        plot_shap_beeswarm(shap_values, X, feature_names, top_n=5)

    def test_respects_top_n_parameter(self, sample_shap_data, capsys):
        """Test that top_n parameter limits features shown."""
        shap_values, X, feature_names = sample_shap_data

        # Test with top_n=3
        plot_shap_beeswarm(shap_values, X, feature_names, top_n=3)
        captured = capsys.readouterr()
        assert "3 features shown" in captured.out

        # Test with top_n=7
        plot_shap_beeswarm(shap_values, X, feature_names, top_n=7)
        captured = capsys.readouterr()
        assert "7 features shown" in captured.out

    def test_saves_figure_when_path_provided(self, sample_shap_data):
        """Test that figure is saved when save_path is provided."""
        shap_values, X, feature_names = sample_shap_data

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_beeswarm.png"
            plot_shap_beeswarm(
                shap_values, X, feature_names,
                top_n=5,
                save_path=str(save_path)
            )
            assert save_path.exists()
            assert save_path.stat().st_size > 0

    def test_handles_missing_feature_in_dataframe(self, sample_shap_data):
        """Test that function handles features not present in DataFrame."""
        shap_values, X, feature_names = sample_shap_data

        # Modify feature names to include one not in X
        modified_feature_names = feature_names.copy()
        modified_feature_names[0] = 'nonexistent_feature'

        # Should not raise an exception (falls back to zeros)
        plot_shap_beeswarm(shap_values, X, modified_feature_names, top_n=5)

    def test_handles_constant_feature_values(self, sample_shap_data):
        """Test that function handles features with constant values."""
        shap_values, X, feature_names = sample_shap_data

        # Make one feature constant
        X['feature_0'] = 1.0

        # Should not raise an exception (division by zero edge case)
        plot_shap_beeswarm(shap_values, X, feature_names, top_n=5)
