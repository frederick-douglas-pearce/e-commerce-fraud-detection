"""Tests for feature_importance module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.fd3_nb.feature_importance import (
    extract_feature_importance,
    print_feature_importance_summary,
)


@pytest.fixture
def sample_xgb_pipeline():
    """Create a trained XGBoost pipeline for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Create features
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )

    # Create target (10% positive)
    y = pd.Series(np.zeros(n_samples))
    y.iloc[:50] = 1
    y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    # Make first few features predictive
    X.loc[y == 1, 'feature_0'] += 2.0
    X.loc[y == 1, 'feature_1'] += 1.5
    X.loc[y == 1, 'feature_2'] += 1.0

    # Create and train pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    model.fit(X, y)

    return model, feature_names


class TestExtractFeatureImportance:
    """Tests for extract_feature_importance function."""

    def test_returns_dataframe(self, sample_xgb_pipeline):
        """Test that extract_feature_importance returns a DataFrame."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, sample_xgb_pipeline):
        """Test that result contains required columns."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert 'feature' in result.columns
        assert 'importance' in result.columns

    def test_correct_number_of_rows(self, sample_xgb_pipeline):
        """Test that result has correct number of rows."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert len(result) == len(feature_names)

    def test_sorted_by_importance_descending(self, sample_xgb_pipeline):
        """Test that results are sorted by importance descending."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        importance_values = result['importance'].values
        assert all(importance_values[i] >= importance_values[i+1]
                   for i in range(len(importance_values)-1))

    def test_importance_values_non_negative(self, sample_xgb_pipeline):
        """Test that all importance values are non-negative."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert (result['importance'] >= 0).all()

    def test_importance_values_sum_to_one(self, sample_xgb_pipeline):
        """Test that importance values sum to approximately 1."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        total_importance = result['importance'].sum()
        assert abs(total_importance - 1.0) < 0.01

    def test_top_features_are_predictive(self, sample_xgb_pipeline):
        """Test that most predictive features have high importance."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        top_3_features = result.head(3)['feature'].tolist()
        # The predictive features (feature_0, feature_1, feature_2) should be in top 3
        predictive_features = ['feature_0', 'feature_1', 'feature_2']
        overlap = len(set(top_3_features) & set(predictive_features))
        assert overlap >= 2, "At least 2 of top 3 should be predictive features"

    def test_verbose_mode_runs(self, sample_xgb_pipeline):
        """Test that verbose mode runs without error."""
        model, feature_names = sample_xgb_pipeline
        # Should not raise an exception
        extract_feature_importance(model, feature_names, verbose=True)


class TestPrintFeatureImportanceSummary:
    """Tests for print_feature_importance_summary function."""

    @pytest.fixture
    def sample_importance_df(self):
        """Create sample feature importance DataFrame."""
        np.random.seed(42)
        n_features = 30

        # Create importance values that sum to 1
        importance_values = np.random.exponential(scale=0.1, size=n_features)
        importance_values = importance_values / importance_values.sum()
        importance_values = np.sort(importance_values)[::-1]  # Sort descending

        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(n_features)],
            'importance': importance_values
        })

    def test_runs_without_error(self, sample_importance_df, capsys):
        """Test that print_feature_importance_summary runs without error."""
        # Should not raise an exception
        print_feature_importance_summary(sample_importance_df)

        # Verify something was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_prints_top_n_features(self, sample_importance_df, capsys):
        """Test that function prints top N features."""
        print_feature_importance_summary(sample_importance_df, top_n=10)
        captured = capsys.readouterr()

        # Should mention "Top 10"
        assert "Top 10" in captured.out

    def test_prints_cumulative_importance(self, sample_importance_df, capsys):
        """Test that function prints cumulative importance info."""
        print_feature_importance_summary(sample_importance_df)
        captured = capsys.readouterr()

        # Should mention model concentration
        assert "Top 5 features" in captured.out
        assert "Top 10 features" in captured.out

    def test_custom_top_n(self, sample_importance_df, capsys):
        """Test with custom top_n value."""
        print_feature_importance_summary(sample_importance_df, top_n=5)
        captured = capsys.readouterr()

        # Should mention "Top 5"
        assert "Top 5" in captured.out
