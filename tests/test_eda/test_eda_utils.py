"""
Tests for src/fd1_nb/eda_utils.py

Tests EDA analysis and visualization utilities.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
from src.fd1_nb.eda_utils import (
    calculate_mi_scores,
    calculate_numeric_correlations,
    calculate_vif,
    analyze_vif,
    analyze_correlations,
    analyze_mutual_information
)


@pytest.fixture
def sample_classification_df():
    """Create a sample classification DataFrame for testing."""
    np.random.seed(42)
    n = 1000

    # Create correlated features
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.5  # Correlated with x1
    x3 = np.random.randn(n)  # Independent

    # Create target with some correlation to features
    target = ((x1 + x2 > 0) & (np.random.rand(n) > 0.3)).astype(int)

    return pd.DataFrame({
        'amount': x1 * 100 + 300,
        'avg_amount': x2 * 100 + 300,
        'distance': x3 * 50 + 100,
        'category': np.random.choice(['A', 'B', 'C'], n),
        'channel': np.random.choice(['web', 'mobile'], n),
        'target': target
    })


class TestCalculateMIScores:
    """Tests for calculate_mi_scores function."""

    def test_returns_dataframe(self, sample_classification_df):
        """Test that function returns a DataFrame."""
        result = calculate_mi_scores(
            sample_classification_df,
            categorical_features=['category', 'channel'],
            target_col='target'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'mi_score' in result.columns
        assert len(result) == 2

    def test_sorted_by_score(self, sample_classification_df):
        """Test that results are sorted by MI score descending."""
        result = calculate_mi_scores(
            sample_classification_df,
            categorical_features=['category', 'channel'],
            target_col='target'
        )

        # Check if sorted
        mi_scores = result['mi_score'].values
        assert all(mi_scores[i] >= mi_scores[i+1] for i in range(len(mi_scores)-1))

    def test_mi_score_range(self, sample_classification_df):
        """Test that MI scores are non-negative."""
        result = calculate_mi_scores(
            sample_classification_df,
            categorical_features=['category', 'channel'],
            target_col='target'
        )

        assert all(result['mi_score'] >= 0)


class TestCalculateNumericCorrelations:
    """Tests for calculate_numeric_correlations function."""

    def test_returns_dataframe(self, sample_classification_df):
        """Test that function returns a DataFrame."""
        result = calculate_numeric_correlations(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            target_col='target'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'correlation' in result.columns
        assert len(result) == 3

    def test_correlation_range(self, sample_classification_df):
        """Test that correlations are in [-1, 1] range."""
        result = calculate_numeric_correlations(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            target_col='target'
        )

        assert all(result['correlation'] >= -1)
        assert all(result['correlation'] <= 1)

    def test_sorted_by_absolute_correlation(self, sample_classification_df):
        """Test that results are sorted by absolute correlation."""
        result = calculate_numeric_correlations(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            target_col='target'
        )

        # First feature should have highest absolute correlation
        abs_corrs = result['correlation'].abs().values
        assert all(abs_corrs[i] >= abs_corrs[i+1] for i in range(len(abs_corrs)-1))


class TestCalculateVIF:
    """Tests for calculate_vif function."""

    def test_returns_dataframe(self, sample_classification_df):
        """Test that function returns a DataFrame."""
        result = calculate_vif(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance']
        )

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'VIF' in result.columns
        assert len(result) == 3

    def test_vif_positive(self, sample_classification_df):
        """Test that VIF values are positive."""
        result = calculate_vif(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance']
        )

        assert all(result['VIF'] > 0)

    def test_correlated_features_high_vif(self):
        """Test that highly correlated features have high VIF."""
        # Create highly correlated features
        np.random.seed(42)
        x = np.random.randn(500)
        df = pd.DataFrame({
            'x1': x,
            'x2': x + np.random.randn(500) * 0.01,  # Almost identical to x1
            'x3': np.random.randn(500)  # Independent
        })

        result = calculate_vif(df, ['x1', 'x2', 'x3'])

        # VIF for x1 or x2 should be much higher than x3
        x3_vif = result[result['feature'] == 'x3']['VIF'].values[0]
        max_vif = result['VIF'].max()

        assert max_vif > x3_vif * 2  # Correlated features should have >2x higher VIF


class TestAnalyzeVIF:
    """Tests for analyze_vif function."""

    def test_returns_dataframe(self, sample_classification_df):
        """Test that function returns VIF DataFrame."""
        result = analyze_vif(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            verbose=False
        )

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'VIF' in result.columns

    def test_verbose_output(self, sample_classification_df, capsys):
        """Test that verbose mode prints analysis."""
        analyze_vif(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            verbose=True
        )

        captured = capsys.readouterr()
        assert 'Variance Inflation Factor' in captured.out
        assert 'Interpretation' in captured.out


class TestAnalyzeCorrelations:
    """Tests for analyze_correlations function."""

    def test_returns_dataframe(self, sample_classification_df):
        """Test that function returns correlation DataFrame."""
        result = analyze_correlations(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            target_col='target',
            verbose=False
        )

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'correlation' in result.columns

    def test_verbose_output(self, sample_classification_df, capsys):
        """Test that verbose mode prints analysis."""
        analyze_correlations(
            sample_classification_df,
            numeric_features=['amount', 'avg_amount', 'distance'],
            target_col='target',
            verbose=True
        )

        captured = capsys.readouterr()
        assert 'Pearson Correlation' in captured.out
        assert 'Key Insights' in captured.out


class TestAnalyzeMutualInformation:
    """Tests for analyze_mutual_information function."""

    def test_returns_dataframe(self, sample_classification_df):
        """Test that function returns MI DataFrame."""
        result = analyze_mutual_information(
            sample_classification_df,
            categorical_features=['category', 'channel'],
            target_col='target',
            mi_threshold=0.05
        )

        assert isinstance(result, pd.DataFrame)
        assert 'feature' in result.columns
        assert 'mi_score' in result.columns

    def test_prints_interpretation(self, sample_classification_df, capsys):
        """Test that function prints interpretation."""
        analyze_mutual_information(
            sample_classification_df,
            categorical_features=['category', 'channel'],
            target_col='target',
            mi_threshold=0.1
        )

        captured = capsys.readouterr()
        assert 'Mutual Information' in captured.out
        assert 'Interpretation' in captured.out
        assert '0.1' in captured.out  # Should mention the threshold
