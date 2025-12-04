"""
Tests for src/eda/feature_engineering.py

Tests feature engineering utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from src.eda.feature_engineering import (
    convert_utc_to_local_time,
    create_temporal_features,
    create_interaction_features,
    create_percentile_based_features
)


@pytest.fixture
def sample_datetime_df():
    """Create a sample DataFrame with UTC datetimes."""
    np.random.seed(42)
    n = 100

    # Create UTC-aware datetimes
    base_time = datetime(2024, 1, 1, tzinfo=pytz.UTC)
    timestamps = [base_time + timedelta(hours=i) for i in range(n)]

    return pd.DataFrame({
        'id': range(n),
        'transaction_time': timestamps,
        'country': np.random.choice(['US', 'GB', 'FR'], n),
        'amount': np.random.uniform(10, 500, n)
    })


@pytest.fixture
def timezone_mapping():
    """Provide sample timezone mapping."""
    return {
        'US': 'America/New_York',
        'GB': 'Europe/London',
        'FR': 'Europe/Paris'
    }


class TestConvertUTCToLocalTime:
    """Tests for convert_utc_to_local_time function."""

    def test_creates_local_time_column(self, sample_datetime_df, timezone_mapping):
        """Test that function creates local_time column."""
        result = convert_utc_to_local_time(
            sample_datetime_df,
            date_col='transaction_time',
            country_col='country',
            timezone_mapping=timezone_mapping,
            verbose=False
        )

        assert 'local_time' in result.columns
        assert len(result) == len(sample_datetime_df)

    def test_preserves_original_columns(self, sample_datetime_df, timezone_mapping):
        """Test that original columns are preserved."""
        original_cols = set(sample_datetime_df.columns)

        result = convert_utc_to_local_time(
            sample_datetime_df,
            date_col='transaction_time',
            country_col='country',
            timezone_mapping=timezone_mapping,
            verbose=False
        )

        # All original columns should still exist
        assert original_cols.issubset(set(result.columns))

    def test_timezone_conversion_accuracy(self, timezone_mapping):
        """Test that timezone conversion is accurate."""
        # Create specific test case
        utc_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=pytz.UTC)  # Noon UTC
        df = pd.DataFrame({
            'time': [utc_time, utc_time, utc_time],
            'country': ['US', 'GB', 'FR']
        })

        result = convert_utc_to_local_time(
            df,
            date_col='time',
            country_col='country',
            timezone_mapping=timezone_mapping,
            verbose=False
        )

        # US (EST) should be 5 hours behind (7am)
        us_local = result[result['country'] == 'US']['local_time'].iloc[0]
        assert us_local.hour == 7

        # GB (GMT) should be same as UTC (12pm)
        gb_local = result[result['country'] == 'GB']['local_time'].iloc[0]
        assert gb_local.hour == 12

        # FR (CET) should be 1 hour ahead (1pm)
        fr_local = result[result['country'] == 'FR']['local_time'].iloc[0]
        assert fr_local.hour == 13

    def test_raises_error_without_timezone(self, timezone_mapping):
        """Test that function raises error if datetime is not timezone-aware."""
        df = pd.DataFrame({
            'time': [datetime(2024, 1, 1, 12, 0, 0)],  # No timezone
            'country': ['US']
        })

        with pytest.raises(ValueError, match="must be timezone-aware"):
            convert_utc_to_local_time(
                df,
                date_col='time',
                country_col='country',
                timezone_mapping=timezone_mapping,
                verbose=False
            )

    def test_custom_output_column(self, sample_datetime_df, timezone_mapping):
        """Test custom output column name."""
        result = convert_utc_to_local_time(
            sample_datetime_df,
            date_col='transaction_time',
            country_col='country',
            timezone_mapping=timezone_mapping,
            output_col='local_timestamp',
            verbose=False
        )

        assert 'local_timestamp' in result.columns


class TestCreateTemporalFeatures:
    """Tests for create_temporal_features function."""

    def test_creates_all_features(self, sample_datetime_df):
        """Test that function creates all temporal features."""
        result, features = create_temporal_features(
            sample_datetime_df,
            date_col='transaction_time',
            suffix='_utc'
        )

        expected_features = [
            'hour_utc',
            'day_of_week_utc',
            'month_utc',
            'is_weekend_utc',
            'is_late_night_utc',
            'is_business_hours_utc'
        ]

        assert set(expected_features) == set(features)
        for feat in expected_features:
            assert feat in result.columns

    def test_hour_range(self, sample_datetime_df):
        """Test that hour values are in [0, 23] range."""
        result, features = create_temporal_features(
            sample_datetime_df,
            date_col='transaction_time'
        )

        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23

    def test_day_of_week_range(self, sample_datetime_df):
        """Test that day_of_week values are in [0, 6] range."""
        result, features = create_temporal_features(
            sample_datetime_df,
            date_col='transaction_time'
        )

        assert result['day_of_week'].min() >= 0
        assert result['day_of_week'].max() <= 6

    def test_month_range(self, sample_datetime_df):
        """Test that month values are in [1, 12] range."""
        result, features = create_temporal_features(
            sample_datetime_df,
            date_col='transaction_time'
        )

        assert result['month'].min() >= 1
        assert result['month'].max() <= 12

    def test_binary_features_are_binary(self, sample_datetime_df):
        """Test that binary features only have 0 and 1 values."""
        result, features = create_temporal_features(
            sample_datetime_df,
            date_col='transaction_time'
        )

        binary_features = ['is_weekend', 'is_late_night', 'is_business_hours']
        for feat in binary_features:
            assert set(result[feat].unique()).issubset({0, 1})

    def test_custom_late_night_hours(self):
        """Test custom late night hours configuration."""
        # Create specific times
        times = [
            datetime(2024, 1, 1, 22, 0, 0),  # 10pm - not late night
            datetime(2024, 1, 1, 23, 0, 0),  # 11pm - late night
            datetime(2024, 1, 1, 3, 0, 0),   # 3am - late night
            datetime(2024, 1, 1, 6, 0, 0),   # 6am - not late night
        ]

        df = pd.DataFrame({'time': times})

        result, _ = create_temporal_features(
            df,
            date_col='time',
            late_night_hours=(23, 5)  # 11pm - 5am
        )

        expected = [0, 1, 1, 0]
        assert list(result['is_late_night']) == expected

    def test_custom_suffix(self, sample_datetime_df):
        """Test custom suffix for feature names."""
        result, features = create_temporal_features(
            sample_datetime_df,
            date_col='transaction_time',
            suffix='_local'
        )

        assert all('_local' in feat for feat in features)


class TestCreateInteractionFeatures:
    """Tests for create_interaction_features function."""

    def test_creates_interaction_features(self):
        """Test that function creates interaction features."""
        df = pd.DataFrame({
            'is_new_account': [1, 1, 0, 0],
            'promo_used': [1, 0, 1, 0],
            'is_late_night': [1, 1, 0, 0],
            'is_micro': [1, 0, 1, 0]
        })

        config = [
            {
                'name': 'new_with_promo',
                'conditions': ['is_new_account == 1', 'promo_used == 1'],
                'operator': 'and'
            },
            {
                'name': 'late_or_micro',
                'conditions': ['is_late_night == 1', 'is_micro == 1'],
                'operator': 'or'
            }
        ]

        result, features = create_interaction_features(df, config)

        assert 'new_with_promo' in result.columns
        assert 'late_or_micro' in result.columns
        assert set(features) == {'new_with_promo', 'late_or_micro'}

    def test_and_operator(self):
        """Test AND operator for interactions."""
        df = pd.DataFrame({
            'a': [1, 1, 0, 0],
            'b': [1, 0, 1, 0]
        })

        config = [{
            'name': 'a_and_b',
            'conditions': ['a == 1', 'b == 1'],
            'operator': 'and'
        }]

        result, _ = create_interaction_features(df, config)

        # Only first row has both a=1 and b=1
        assert list(result['a_and_b']) == [1, 0, 0, 0]

    def test_or_operator(self):
        """Test OR operator for interactions."""
        df = pd.DataFrame({
            'a': [1, 1, 0, 0],
            'b': [1, 0, 1, 0]
        })

        config = [{
            'name': 'a_or_b',
            'conditions': ['a == 1', 'b == 1'],
            'operator': 'or'
        }]

        result, _ = create_interaction_features(df, config)

        # First three rows have at least one of a=1 or b=1
        assert list(result['a_or_b']) == [1, 1, 1, 0]

    def test_invalid_operator(self):
        """Test that invalid operator raises ValueError."""
        df = pd.DataFrame({'a': [1, 0], 'b': [1, 0]})

        config = [{
            'name': 'invalid',
            'conditions': ['a == 1', 'b == 1'],
            'operator': 'xor'  # Invalid operator
        }]

        with pytest.raises(ValueError, match="Unsupported operator"):
            create_interaction_features(df, config)


class TestCreatePercentileBasedFeatures:
    """Tests for create_percentile_based_features function."""

    def test_creates_percentile_features(self):
        """Test that function creates percentile-based features."""
        df = pd.DataFrame({
            'amount': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })

        config = [{
            'source_col': 'amount',
            'feature_name': 'is_high_amount',
            'operator': '>=',
            'percentile': 0.9
        }]

        result, features = create_percentile_based_features(df, config, None, None)

        assert 'is_high_amount' in result.columns
        assert 'is_high_amount' in features

        # 90th percentile of [10..100] is 91, so values >= 91 should be flagged (only 100)
        assert result['is_high_amount'].sum() == 1
        assert result[result['amount'] == 100]['is_high_amount'].all()

    def test_different_operators(self):
        """Test different comparison operators."""
        df = pd.DataFrame({'value': range(100)})

        # Test >=
        config_gte = [{
            'source_col': 'value',
            'feature_name': 'high',
            'operator': '>=',
            'percentile': 0.75
        }]
        result_gte, _ = create_percentile_based_features(df, config_gte, None, None)
        assert result_gte['high'].sum() == 25  # Top 25%

        # Test <=
        config_lte = [{
            'source_col': 'value',
            'feature_name': 'low',
            'operator': '<=',
            'percentile': 0.25
        }]
        result_lte, _ = create_percentile_based_features(df, config_lte, None, None)
        assert result_lte['low'].sum() == 25  # Bottom 25%

    def test_multiple_features(self):
        """Test creating multiple percentile-based features."""
        df = pd.DataFrame({
            'amount': range(100),
            'distance': range(100, 200)
        })

        config = [
            {
                'source_col': 'amount',
                'feature_name': 'high_amount',
                'operator': '>=',
                'percentile': 0.9
            },
            {
                'source_col': 'distance',
                'feature_name': 'long_distance',
                'operator': '>=',
                'percentile': 0.75
            }
        ]

        result, features = create_percentile_based_features(df, config, None, None)

        assert len(features) == 2
        assert 'high_amount' in result.columns
        assert 'long_distance' in result.columns
