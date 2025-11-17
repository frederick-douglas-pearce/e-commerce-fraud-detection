"""Shared pytest fixtures for testing fraud detection system."""

import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.preprocessing import FeatureConfig, FraudFeatureTransformer
from src.preprocessing.features import get_country_timezone_mapping, get_final_feature_names


@pytest.fixture
def sample_raw_df():
    """Create a small sample raw DataFrame for testing.

    Returns:
        DataFrame with 10 rows containing all required raw features
    """
    np.random.seed(42)

    # Create sample data
    n_samples = 10
    base_date = datetime(2024, 1, 15, 10, 0, 0)

    data = {
        'transaction_id': [f'TXN{i:04d}' for i in range(n_samples)],
        'user_id': [f'USER{i:03d}' for i in range(n_samples)],
        'transaction_time': [
            (base_date + timedelta(hours=i * 2)).strftime('%Y-%m-%d %H:%M:%S')
            for i in range(n_samples)
        ],
        'amount': [10.5, 1.0, 150.0, 75.0, 5.0, 200.0, 25.0, 3.0, 100.0, 50.0],
        'merchant_category': ['electronics'] * n_samples,
        'country': ['US', 'GB', 'FR', 'US', 'DE', 'US', 'IT', 'US', 'CA', 'US'],
        'bin_country': ['US', 'US', 'FR', 'US', 'GB', 'US', 'IT', 'US', 'CA', 'US'],
        'shipping_distance_km': [0, 500, 100, 0, 1500, 250, 75, 0, 5000, 125],
        'channel': ['web', 'app', 'web', 'app', 'web', 'web', 'app', 'web', 'web', 'app'],
        'promo_used': [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        'avs_match': [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        'cvv_result': [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        'three_ds_flag': [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
        'account_age_days': [150, 25, 365, 90, 15, 500, 30, 200, 60, 180],
        'total_transactions_user': [20, 3, 100, 45, 2, 150, 5, 75, 30, 50],
        'avg_amount_user': [50.0, 10.0, 125.0, 60.0, 5.0, 175.0, 20.0, 75.0, 90.0, 55.0]
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_raw_df_utc(sample_raw_df):
    """Create sample raw DataFrame with UTC timezone-aware datetimes.

    This fixture preprocesses the sample data to have proper UTC timestamps,
    as required by the feature engineering pipeline.

    Returns:
        DataFrame with transaction_time as timezone-aware UTC datetime
    """
    df = sample_raw_df.copy()
    df['transaction_time'] = pd.to_datetime(df['transaction_time'], utc=True)
    return df


@pytest.fixture
def sample_config():
    """Create a pre-configured FeatureConfig for testing.

    Returns:
        FeatureConfig with reasonable test values
    """
    return FeatureConfig(
        amount_95th_percentile=180.0,
        total_transactions_75th_percentile=75,
        shipping_distance_75th_percentile=500.0,
        timezone_mapping=get_country_timezone_mapping(),
        final_features=get_final_feature_names(),
        date_col='transaction_time',
        country_col='country'
    )


@pytest.fixture
def fitted_transformer(sample_raw_df_utc):
    """Create a fitted FraudFeatureTransformer for testing.

    Returns:
        FraudFeatureTransformer fitted on sample data
    """
    transformer = FraudFeatureTransformer()
    transformer.fit(sample_raw_df_utc)
    return transformer


@pytest.fixture
def sample_engineered_df(fitted_transformer, sample_raw_df_utc):
    """Create a sample engineered DataFrame with all features.

    Returns:
        DataFrame with 30 engineered features
    """
    return fitted_transformer.transform(sample_raw_df_utc)
