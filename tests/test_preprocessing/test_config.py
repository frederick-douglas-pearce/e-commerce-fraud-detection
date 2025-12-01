"""Tests for FeatureConfig dataclass."""

import pytest
import json
import tempfile
from pathlib import Path

from src.preprocessing import FeatureConfig
from src.preprocessing.features import get_country_timezone_mapping, get_final_feature_names


def test_feature_config_creation(sample_config):
    """Test that FeatureConfig can be created with all required fields."""
    assert sample_config.amount_95th_percentile == 180.0
    assert sample_config.total_transactions_75th_percentile == 75
    assert sample_config.shipping_distance_75th_percentile == 500.0
    assert sample_config.date_col == 'transaction_time'
    assert sample_config.country_col == 'country'
    assert isinstance(sample_config.timezone_mapping, dict)
    assert isinstance(sample_config.final_features, list)
    assert len(sample_config.final_features) == 30


def test_feature_config_from_training_data(sample_raw_df_utc):
    """Test that FeatureConfig can be created from training data."""
    config = FeatureConfig.from_training_data(sample_raw_df_utc)

    # Check that quantiles are calculated
    assert isinstance(config.amount_95th_percentile, float)
    assert isinstance(config.total_transactions_75th_percentile, int)
    assert isinstance(config.shipping_distance_75th_percentile, float)

    # Check that quantiles are reasonable
    assert config.amount_95th_percentile > 0
    assert config.total_transactions_75th_percentile > 0
    assert config.shipping_distance_75th_percentile > 0

    # Check that mappings are populated
    assert len(config.timezone_mapping) == 10
    assert len(config.final_features) == 30

    # Verify specific quantile values for sample data
    expected_amount_95th = sample_raw_df_utc['amount'].quantile(0.95)
    assert abs(config.amount_95th_percentile - expected_amount_95th) < 0.01


def test_feature_config_save_load_roundtrip(sample_config):
    """Test that FeatureConfig can be saved and loaded correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'test_config.json'

        # Save config
        sample_config.save(str(config_path))

        # Verify file was created
        assert config_path.exists()

        # Load config
        loaded_config = FeatureConfig.load(str(config_path))

        # Verify all fields match
        assert loaded_config.amount_95th_percentile == sample_config.amount_95th_percentile
        assert loaded_config.total_transactions_75th_percentile == sample_config.total_transactions_75th_percentile
        assert loaded_config.shipping_distance_75th_percentile == sample_config.shipping_distance_75th_percentile
        assert loaded_config.date_col == sample_config.date_col
        assert loaded_config.country_col == sample_config.country_col
        assert loaded_config.timezone_mapping == sample_config.timezone_mapping
        assert loaded_config.final_features == sample_config.final_features


def test_feature_config_save_creates_directory(sample_config):
    """Test that save() creates parent directories if they don't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'nested' / 'dir' / 'config.json'

        # Directory doesn't exist yet
        assert not config_path.parent.exists()

        # Save should create it
        sample_config.save(str(config_path))

        # Verify directory and file were created
        assert config_path.parent.exists()
        assert config_path.exists()


def test_feature_config_json_structure(sample_config):
    """Test that saved JSON has correct structure and is human-readable."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / 'config.json'
        sample_config.save(str(config_path))

        # Load raw JSON
        with open(config_path, 'r') as f:
            json_data = json.load(f)

        # Verify all expected keys are present
        expected_keys = {
            'amount_95th_percentile',
            'total_transactions_75th_percentile',
            'shipping_distance_75th_percentile',
            'timezone_mapping',
            'final_features',
            'date_col',
            'country_col'
        }
        assert set(json_data.keys()) == expected_keys

        # Verify types
        assert isinstance(json_data['amount_95th_percentile'], (int, float))
        assert isinstance(json_data['timezone_mapping'], dict)
        assert isinstance(json_data['final_features'], list)


def test_feature_config_timezone_mapping_content(sample_config):
    """Test that timezone mapping contains expected countries."""
    expected_countries = {'US', 'GB', 'FR', 'DE', 'IT', 'ES', 'CA', 'AU', 'JP', 'BR'}
    assert set(sample_config.timezone_mapping.keys()) == expected_countries

    # Verify timezone format
    for country, tz in sample_config.timezone_mapping.items():
        assert isinstance(tz, str)
        assert '/' in tz  # Timezone format like 'America/New_York'


def test_feature_config_final_features_content(sample_config):
    """Test that final_features list contains expected feature categories."""
    features = sample_config.final_features

    # Expected features by category
    original_numeric = {'account_age_days', 'total_transactions_user', 'avg_amount_user', 'amount', 'shipping_distance_km'}
    original_categorical = {'channel'}
    original_binary = {'promo_used', 'avs_match', 'cvv_result', 'three_ds_flag'}
    temporal_local = {'hour_local', 'day_of_week_local', 'month_local', 'is_weekend_local', 'is_late_night_local', 'is_business_hours_local'}
    amount_features = {'amount_deviation', 'amount_vs_avg_ratio', 'is_micro_transaction', 'is_large_transaction'}
    behavior_features = {'transaction_velocity', 'is_new_account', 'is_high_frequency_user'}
    geographic_features = {'country_mismatch', 'high_risk_distance', 'zero_distance'}
    security_features = {'security_score'}
    interaction_features = {'new_account_with_promo', 'late_night_micro_transaction', 'high_value_long_distance'}

    all_expected = (
        original_numeric | original_categorical | original_binary | temporal_local |
        amount_features | behavior_features | geographic_features |
        security_features | interaction_features
    )

    assert set(features) == all_expected
    assert len(features) == 30


def test_feature_config_load_nonexistent_file():
    """Test that loading a nonexistent file raises appropriate error."""
    with pytest.raises(FileNotFoundError):
        FeatureConfig.load('nonexistent_config.json')
