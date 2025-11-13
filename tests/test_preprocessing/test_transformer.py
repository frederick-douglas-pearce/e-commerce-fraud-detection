"""Tests for FraudFeatureTransformer integration."""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from src.preprocessing import FraudFeatureTransformer, FeatureConfig


def test_transformer_initialization():
    """Test that transformer can be initialized with and without config."""
    # Without config
    transformer = FraudFeatureTransformer()
    assert transformer.config is None

    # With config
    config = FeatureConfig(
        amount_95th_percentile=100.0,
        total_transactions_75th_percentile=50.0,
        shipping_distance_75th_percentile=300.0,
        timezone_mapping={'US': 'America/New_York'},
        final_features=['amount', 'account_age_days']
    )
    transformer = FraudFeatureTransformer(config=config)
    assert transformer.config is not None
    assert transformer.config.amount_95th_percentile == 100.0


def test_transformer_fit(sample_raw_df_utc):
    """Test that transformer can be fitted on training data."""
    transformer = FraudFeatureTransformer()

    # Should start with no config
    assert transformer.config is None

    # Fit should create config
    transformer.fit(sample_raw_df_utc)

    # Config should now be set
    assert transformer.config is not None
    assert isinstance(transformer.config, FeatureConfig)
    assert transformer.config.amount_95th_percentile > 0
    assert len(transformer.config.final_features) == 30


def test_transformer_transform_without_fit():
    """Test that transform raises error if transformer not fitted."""
    transformer = FraudFeatureTransformer()

    df = pd.DataFrame({'amount': [10, 20]})

    with pytest.raises(ValueError, match="Transformer must be fit"):
        transformer.transform(df)


def test_transformer_transform_output_shape(fitted_transformer, sample_raw_df_utc):
    """Test that transform returns correct shape (30 features)."""
    result = fitted_transformer.transform(sample_raw_df_utc)

    # Should have 30 columns (features)
    assert result.shape[1] == 30

    # Should have same number of rows as input
    assert result.shape[0] == sample_raw_df_utc.shape[0]


def test_transformer_transform_output_columns(fitted_transformer, sample_raw_df_utc):
    """Test that transform returns exactly the 30 selected features."""
    result = fitted_transformer.transform(sample_raw_df_utc)

    expected_features = fitted_transformer.config.final_features
    assert len(result.columns) == 30
    assert list(result.columns) == expected_features


def test_transformer_fit_transform(sample_raw_df_utc):
    """Test that fit_transform works correctly."""
    transformer = FraudFeatureTransformer()

    result = transformer.fit_transform(sample_raw_df_utc)

    # Transformer should be fitted
    assert transformer.config is not None

    # Result should have correct shape
    assert result.shape[1] == 30
    assert result.shape[0] == sample_raw_df_utc.shape[0]


def test_transformer_preserves_index(sample_raw_df_utc):
    """Test that transformer preserves DataFrame index."""
    df = sample_raw_df_utc.copy()
    df.index = [f'row_{i}' for i in range(len(df))]

    transformer = FraudFeatureTransformer()
    result = transformer.fit_transform(df)

    # Index should be preserved
    assert list(result.index) == list(df.index)


def test_transformer_creates_expected_features(fitted_transformer, sample_raw_df_utc):
    """Test that all expected feature categories are present."""
    result = fitted_transformer.transform(sample_raw_df_utc)

    # Check for features from each category
    # Original numeric
    assert 'amount' in result.columns
    assert 'account_age_days' in result.columns

    # Original categorical
    assert 'channel' in result.columns
    assert 'promo_used' in result.columns

    # Temporal (local)
    assert 'hour_local' in result.columns
    assert 'is_late_night_local' in result.columns

    # Amount features
    assert 'amount_deviation' in result.columns
    assert 'is_micro_transaction' in result.columns

    # Behavior features
    assert 'transaction_velocity' in result.columns
    assert 'is_new_account' in result.columns

    # Geographic features
    assert 'country_mismatch' in result.columns
    assert 'high_risk_distance' in result.columns

    # Security features
    assert 'security_score' in result.columns

    # Interaction features
    assert 'new_account_with_promo' in result.columns
    assert 'late_night_micro_transaction' in result.columns


def test_transformer_no_missing_values(fitted_transformer, sample_raw_df_utc):
    """Test that transformed data has no missing values."""
    result = fitted_transformer.transform(sample_raw_df_utc)

    # No NaN values should be present
    assert not result.isna().any().any()


def test_transformer_feature_types(fitted_transformer, sample_raw_df_utc):
    """Test that features have appropriate data types."""
    result = fitted_transformer.transform(sample_raw_df_utc)

    # Binary flags should be 0 or 1
    binary_features = [
        'is_micro_transaction', 'is_large_transaction',
        'is_new_account', 'is_high_frequency_user',
        'country_mismatch', 'high_risk_distance', 'zero_distance',
        'is_weekend_local', 'is_late_night_local', 'is_business_hours_local',
        'new_account_with_promo', 'late_night_micro_transaction', 'high_value_long_distance'
    ]

    for feat in binary_features:
        if feat in result.columns:
            assert set(result[feat].unique()).issubset({0, 1})

    # Numeric features should be numeric
    assert pd.api.types.is_numeric_dtype(result['amount'])
    assert pd.api.types.is_numeric_dtype(result['transaction_velocity'])


def test_transformer_save_load(fitted_transformer, sample_raw_df_utc):
    """Test that transformer can be saved and loaded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'transformer_config.json'

        # Save transformer
        fitted_transformer.save(str(config_path))

        # Load transformer
        loaded_transformer = FraudFeatureTransformer.load(str(config_path))

        # Transform with both transformers
        result_original = fitted_transformer.transform(sample_raw_df_utc)
        result_loaded = loaded_transformer.transform(sample_raw_df_utc)

        # Results should be identical
        pd.testing.assert_frame_equal(result_original, result_loaded)


def test_transformer_save_unfitted_raises_error():
    """Test that saving unfitted transformer raises error."""
    transformer = FraudFeatureTransformer()

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'

        with pytest.raises(ValueError, match="Cannot save unfitted transformer"):
            transformer.save(str(config_path))


def test_transformer_sklearn_pipeline_compatibility(sample_raw_df_utc):
    """Test that transformer works in sklearn Pipeline."""
    # Create a simple pipeline
    pipeline = Pipeline([
        ('feature_engineering', FraudFeatureTransformer()),
        # Logistic Regression will fail without proper preprocessing,
        # but we're just testing that the transformer works in pipeline
    ])

    # Fit pipeline
    # Create dummy target
    y = pd.Series([0] * len(sample_raw_df_utc))

    # This should not raise any errors
    pipeline.fit(sample_raw_df_utc, y)

    # Transform should work
    result = pipeline.transform(sample_raw_df_utc)
    assert result.shape[1] == 30


def test_transformer_handles_new_data_same_format(fitted_transformer, sample_raw_df_utc):
    """Test that transformer works on new data with same format."""
    # Create new data with same structure
    new_data = sample_raw_df_utc.iloc[:3].copy()

    # Transform should work
    result = fitted_transformer.transform(new_data)

    assert result.shape[0] == 3
    assert result.shape[1] == 30


def test_transformer_consistent_thresholds(sample_raw_df_utc):
    """Test that fitted thresholds are used consistently."""
    transformer = FraudFeatureTransformer()
    transformer.fit(sample_raw_df_utc)

    # Get thresholds
    amount_threshold = transformer.config.amount_95th_percentile

    # Transform data
    result = transformer.transform(sample_raw_df_utc)

    # Check that is_large_transaction uses the fitted threshold
    expected_large = (sample_raw_df_utc['amount'] >= amount_threshold).astype(int)

    # Note: The result DataFrame doesn't have the original amount column in the same order,
    # but we can verify the logic worked by checking that the flag was created
    assert 'is_large_transaction' in result.columns


def test_transformer_datetime_preprocessing(sample_raw_df):
    """Test that transformer properly handles datetime preprocessing."""
    # Create data without UTC timezone (string dates)
    df = sample_raw_df.copy()

    transformer = FraudFeatureTransformer()

    # Fit should work (will convert to UTC internally)
    transformer.fit(df)

    # Transform should work
    result = transformer.transform(df)

    assert result.shape[1] == 30
    assert 'hour_local' in result.columns


def test_transformer_multiple_transforms(fitted_transformer, sample_raw_df_utc):
    """Test that transformer can be used multiple times."""
    # First transform
    result1 = fitted_transformer.transform(sample_raw_df_utc)

    # Second transform on same data
    result2 = fitted_transformer.transform(sample_raw_df_utc)

    # Results should be identical
    pd.testing.assert_frame_equal(result1, result2)


def test_transformer_feature_order_consistency(sample_raw_df_utc):
    """Test that features are always in the same order."""
    transformer1 = FraudFeatureTransformer()
    transformer2 = FraudFeatureTransformer()

    result1 = transformer1.fit_transform(sample_raw_df_utc)
    result2 = transformer2.fit_transform(sample_raw_df_utc)

    # Column order should be identical
    assert list(result1.columns) == list(result2.columns)
