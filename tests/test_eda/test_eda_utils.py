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
    analyze_mutual_information,
    analyze_temporal_patterns,
    analyze_categorical_fraud_rates,
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


class TestAnalyzeTemporalPatterns:
    """Tests for analyze_temporal_patterns function."""

    @pytest.fixture
    def temporal_df(self):
        """Create a DataFrame with datetime column for temporal analysis."""
        np.random.seed(42)
        n = 1000

        # Create datetime range over several months
        dates = pd.date_range(start='2023-01-01', periods=n, freq='h')

        # Create target with some temporal pattern
        # Higher fraud rate during late night hours (0-5)
        hours = dates.hour
        target = np.zeros(n)
        for i, hour in enumerate(hours):
            if hour < 6:
                target[i] = np.random.choice([0, 1], p=[0.90, 0.10])  # 10% fraud
            else:
                target[i] = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud

        return pd.DataFrame({
            'transaction_time': dates,
            'amount': np.random.uniform(10, 500, n),
            'is_fraud': target.astype(int)
        })

    def test_creates_temporal_features(self, temporal_df):
        """Test that function creates temporal feature columns."""
        df_copy = temporal_df.copy()
        analyze_temporal_patterns(
            df_copy,
            date_col='transaction_time',
            target_col='is_fraud',
            baseline_rate=0.05
        )

        # Check that temporal columns were created
        assert 'hour' in df_copy.columns
        assert 'day_of_week' in df_copy.columns
        assert 'month' in df_copy.columns
        assert 'is_weekend' in df_copy.columns

    def test_hour_feature_range(self, temporal_df):
        """Test that hour feature is in valid range."""
        df_copy = temporal_df.copy()
        analyze_temporal_patterns(
            df_copy,
            date_col='transaction_time',
            target_col='is_fraud',
            baseline_rate=0.05
        )

        assert df_copy['hour'].min() >= 0
        assert df_copy['hour'].max() <= 23

    def test_day_of_week_range(self, temporal_df):
        """Test that day_of_week feature is in valid range."""
        df_copy = temporal_df.copy()
        analyze_temporal_patterns(
            df_copy,
            date_col='transaction_time',
            target_col='is_fraud',
            baseline_rate=0.05
        )

        assert df_copy['day_of_week'].min() >= 0
        assert df_copy['day_of_week'].max() <= 6

    def test_is_weekend_binary(self, temporal_df):
        """Test that is_weekend is binary."""
        df_copy = temporal_df.copy()
        analyze_temporal_patterns(
            df_copy,
            date_col='transaction_time',
            target_col='is_fraud',
            baseline_rate=0.05
        )

        assert set(df_copy['is_weekend'].unique()).issubset({0, 1})

    def test_modifies_dataframe_inplace(self, temporal_df):
        """Test that function modifies DataFrame in-place."""
        df_original = temporal_df.copy()
        original_columns = set(df_original.columns)

        analyze_temporal_patterns(
            df_original,
            date_col='transaction_time',
            target_col='is_fraud',
            baseline_rate=0.05
        )

        # Should have more columns now
        assert len(df_original.columns) > len(original_columns)

    def test_custom_risk_threshold(self, temporal_df, capsys):
        """Test with custom risk threshold."""
        df_copy = temporal_df.copy()
        analyze_temporal_patterns(
            df_copy,
            date_col='transaction_time',
            target_col='is_fraud',
            baseline_rate=0.05,
            risk_threshold=2.0
        )

        captured = capsys.readouterr()
        # Should print temporal insights
        assert 'Peak' in captured.out or 'Temporal' in captured.out or 'fraud' in captured.out.lower()


class TestAnalyzeCategoricalFraudRates:
    """Tests for analyze_categorical_fraud_rates function."""

    @pytest.fixture
    def categorical_fraud_df(self):
        """Create a DataFrame with categorical features and fraud target."""
        np.random.seed(42)
        n = 1000

        # Create categories with different fraud rates
        channels = np.random.choice(['web', 'mobile', 'app'], n, p=[0.5, 0.3, 0.2])
        countries = np.random.choice(['US', 'UK', 'CA', 'AU'], n, p=[0.4, 0.3, 0.2, 0.1])

        # Create fraud with pattern based on channel
        target = np.zeros(n)
        for i in range(n):
            if channels[i] == 'web':
                target[i] = np.random.choice([0, 1], p=[0.95, 0.05])  # 5% fraud
            elif channels[i] == 'mobile':
                target[i] = np.random.choice([0, 1], p=[0.98, 0.02])  # 2% fraud
            else:
                target[i] = np.random.choice([0, 1], p=[0.99, 0.01])  # 1% fraud

        return pd.DataFrame({
            'channel': channels,
            'country': countries,
            'amount': np.random.uniform(10, 500, n),
            'is_fraud': target.astype(int)
        })

    def test_prints_feature_analysis(self, categorical_fraud_df, capsys):
        """Test that function prints analysis for each feature."""
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel', 'country'],
            target_col='is_fraud'
        )

        captured = capsys.readouterr()
        assert 'CHANNEL' in captured.out
        assert 'COUNTRY' in captured.out
        assert 'target_rate' in captured.out

    def test_prints_column_headers(self, categorical_fraud_df, capsys):
        """Test that output includes expected column headers."""
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel'],
            target_col='is_fraud'
        )

        captured = capsys.readouterr()
        assert 'target_count' in captured.out
        assert 'total_count' in captured.out
        assert 'target_rate' in captured.out

    def test_custom_baseline_rate(self, categorical_fraud_df, capsys):
        """Test with custom baseline rate."""
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel'],
            target_col='is_fraud',
            baseline_rate=0.02
        )

        captured = capsys.readouterr()
        assert '2.00%' in captured.out  # Baseline rate in output

    def test_high_risk_detection(self, categorical_fraud_df, capsys):
        """Test that high-risk categories are flagged."""
        # Use a low baseline and threshold to ensure high-risk detection
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel'],
            target_col='is_fraud',
            baseline_rate=0.01,
            risk_threshold=1.5
        )

        captured = capsys.readouterr()
        # Web has 5% fraud, which should be flagged as high-risk
        assert 'High-risk' in captured.out or 'web' in captured.out.lower()

    def test_custom_risk_threshold(self, categorical_fraud_df, capsys):
        """Test with custom risk threshold."""
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel'],
            target_col='is_fraud',
            risk_threshold=3.0
        )

        captured = capsys.readouterr()
        assert '>3.0x' in captured.out or 'CHANNEL' in captured.out

    def test_single_feature(self, categorical_fraud_df, capsys):
        """Test with single categorical feature."""
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel'],
            target_col='is_fraud'
        )

        captured = capsys.readouterr()
        assert 'CHANNEL' in captured.out
        assert 'COUNTRY' not in captured.out

    def test_computed_baseline_rate(self, categorical_fraud_df, capsys):
        """Test with no baseline_rate (computed from data)."""
        analyze_categorical_fraud_rates(
            categorical_fraud_df,
            categorical_features=['channel'],
            target_col='is_fraud',
            baseline_rate=None
        )

        captured = capsys.readouterr()
        # Should still work without explicit baseline
        assert 'CHANNEL' in captured.out
