"""Tests for feature engineering functions."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.preprocessing.features import (
    get_country_timezone_mapping,
    get_final_feature_names,
    convert_to_local_time,
    create_temporal_features,
    create_amount_features,
    create_user_behavior_features,
    create_geographic_features,
    create_security_features,
    create_interaction_features
)


# Test helper functions

def test_get_country_timezone_mapping():
    """Test that timezone mapping returns expected countries."""
    mapping = get_country_timezone_mapping()

    assert isinstance(mapping, dict)
    assert len(mapping) == 10
    assert 'US' in mapping
    assert mapping['US'] == 'America/New_York'
    assert 'GB' in mapping
    assert mapping['GB'] == 'Europe/London'


def test_get_final_feature_names():
    """Test that final feature names list is correct."""
    features = get_final_feature_names()

    assert isinstance(features, list)
    assert len(features) == 30

    # Check for specific expected features
    assert 'amount' in features
    assert 'is_late_night_local' in features
    assert 'new_account_with_promo' in features
    assert 'security_score' in features

    # Verify no UTC temporal features (should use local only)
    assert 'hour' not in features
    assert 'hour_local' in features


# Test timezone conversion

def test_convert_to_local_time_basic(sample_raw_df_utc):
    """Test basic timezone conversion functionality."""
    df = sample_raw_df_utc.copy()
    mapping = get_country_timezone_mapping()

    result = convert_to_local_time(df, 'transaction_time', 'country', mapping)

    # Verify local_time column is created
    assert 'local_time' in result.columns

    # Verify local_time is timezone-naive
    assert result['local_time'].dt.tz is None

    # Verify times have been converted (should differ from UTC for most countries)
    us_mask = result['country'] == 'US'
    if us_mask.sum() > 0:
        # US Eastern Time is UTC-5 or UTC-4 depending on DST
        # Just verify that local_time exists and is different or same as UTC
        assert result.loc[us_mask, 'local_time'].notna().all()


def test_convert_to_local_time_validation_no_timezone():
    """Test that conversion raises error if input is not timezone-aware."""
    df = pd.DataFrame({
        'transaction_time': pd.to_datetime(['2024-01-15 10:00:00']),
        'country': ['US']
    })
    mapping = get_country_timezone_mapping()

    with pytest.raises(ValueError, match="must be timezone-aware in UTC"):
        convert_to_local_time(df, 'transaction_time', 'country', mapping)


def test_convert_to_local_time_validation_wrong_timezone():
    """Test that conversion raises error if input is not in UTC."""
    df = pd.DataFrame({
        'transaction_time': pd.to_datetime(['2024-01-15 10:00:00']).tz_localize('America/New_York'),
        'country': ['US']
    })
    mapping = get_country_timezone_mapping()

    with pytest.raises(ValueError, match="must be in UTC timezone"):
        convert_to_local_time(df, 'transaction_time', 'country', mapping)


# Test temporal features

def test_create_temporal_features_utc(sample_raw_df_utc):
    """Test temporal feature creation from UTC timestamps."""
    df = sample_raw_df_utc.copy()

    result, features = create_temporal_features(df, 'transaction_time', use_local_time=False)

    # Verify all features are created
    assert len(features) == 6
    expected_features = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_late_night', 'is_business_hours']
    assert features == expected_features

    for feat in features:
        assert feat in result.columns

    # Verify feature values are reasonable
    assert result['hour'].min() >= 0
    assert result['hour'].max() <= 23
    assert result['day_of_week'].min() >= 0
    assert result['day_of_week'].max() <= 6
    assert result['month'].min() >= 1
    assert result['month'].max() <= 12

    # Verify binary flags
    assert set(result['is_weekend'].unique()).issubset({0, 1})
    assert set(result['is_late_night'].unique()).issubset({0, 1})
    assert set(result['is_business_hours'].unique()).issubset({0, 1})


def test_create_temporal_features_local(sample_raw_df_utc):
    """Test temporal feature creation from local timestamps."""
    df = sample_raw_df_utc.copy()
    mapping = get_country_timezone_mapping()
    df = convert_to_local_time(df, 'transaction_time', 'country', mapping)

    result, features = create_temporal_features(df, 'local_time', use_local_time=True)

    # Verify all features have '_local' suffix
    assert len(features) == 6
    for feat in features:
        assert feat.endswith('_local')
        assert feat in result.columns


# Test amount features

def test_create_amount_features(sample_raw_df_utc):
    """Test amount feature creation."""
    df = sample_raw_df_utc.copy()
    threshold = 150.0

    result, features = create_amount_features(df, threshold)

    # Verify all features are created
    assert len(features) == 4
    expected_features = ['amount_deviation', 'amount_vs_avg_ratio', 'is_micro_transaction', 'is_large_transaction']
    assert features == expected_features

    # Verify amount_deviation is non-negative
    assert (result['amount_deviation'] >= 0).all()

    # Verify ratio is calculated correctly
    for idx in result.index:
        if result.loc[idx, 'avg_amount_user'] > 0:
            expected_ratio = result.loc[idx, 'amount'] / result.loc[idx, 'avg_amount_user']
            assert abs(result.loc[idx, 'amount_vs_avg_ratio'] - expected_ratio) < 0.01

    # Verify micro transaction flag (amount <= 5)
    assert (result[result['amount'] <= 5]['is_micro_transaction'] == 1).all()
    assert (result[result['amount'] > 5]['is_micro_transaction'] == 0).all()

    # Verify large transaction flag
    assert (result[result['amount'] >= threshold]['is_large_transaction'] == 1).all()


def test_create_amount_features_zero_avg():
    """Test that amount features handle zero average correctly."""
    df = pd.DataFrame({
        'amount': [10.0, 20.0],
        'avg_amount_user': [0.0, 50.0]
    })

    result, _ = create_amount_features(df, 100.0)

    # When avg is 0, ratio should be 0 (not division by zero error)
    assert result.loc[0, 'amount_vs_avg_ratio'] == 0.0
    assert result.loc[1, 'amount_vs_avg_ratio'] == 0.4  # 20 / 50


# Test user behavior features

def test_create_user_behavior_features(sample_raw_df_utc):
    """Test user behavior feature creation."""
    df = sample_raw_df_utc.copy()
    threshold = 50.0

    result, features = create_user_behavior_features(df, threshold)

    # Verify all features are created
    assert len(features) == 3
    expected_features = ['transaction_velocity', 'is_new_account', 'is_high_frequency_user']
    assert features == expected_features

    # Verify velocity is non-negative
    assert (result['transaction_velocity'] >= 0).all()

    # Verify new account flag (account_age_days <= 30)
    assert (result[result['account_age_days'] <= 30]['is_new_account'] == 1).all()
    assert (result[result['account_age_days'] > 30]['is_new_account'] == 0).all()

    # Verify high frequency flag
    assert (result[result['total_transactions_user'] >= threshold]['is_high_frequency_user'] == 1).all()


def test_create_user_behavior_features_zero_account_age():
    """Test that user behavior features handle zero account age correctly."""
    df = pd.DataFrame({
        'account_age_days': [0, 100],
        'total_transactions_user': [5, 50]
    })

    result, _ = create_user_behavior_features(df, 75.0)

    # When account_age is 0, velocity should be 0 (not division by zero error)
    assert result.loc[0, 'transaction_velocity'] == 0.0
    assert result.loc[1, 'transaction_velocity'] == 0.5  # 50 / 100


# Test geographic features

def test_create_geographic_features(sample_raw_df_utc):
    """Test geographic feature creation."""
    df = sample_raw_df_utc.copy()
    threshold = 500.0

    result, features = create_geographic_features(df, threshold)

    # Verify all features are created
    assert len(features) == 3
    expected_features = ['country_mismatch', 'high_risk_distance', 'zero_distance']
    assert features == expected_features

    # Verify country mismatch
    for idx in result.index:
        expected = 1 if result.loc[idx, 'country'] != result.loc[idx, 'bin_country'] else 0
        assert result.loc[idx, 'country_mismatch'] == expected

    # Verify high risk distance
    assert (result[result['shipping_distance_km'] >= threshold]['high_risk_distance'] == 1).all()

    # Verify zero distance
    assert (result[result['shipping_distance_km'] == 0]['zero_distance'] == 1).all()
    assert (result[result['shipping_distance_km'] > 0]['zero_distance'] == 0).all()


# Test security features

def test_create_security_features(sample_raw_df_utc):
    """Test security feature creation."""
    df = sample_raw_df_utc.copy()

    result, features = create_security_features(df)

    # Verify all features are created
    assert len(features) == 4
    expected_features = ['security_score', 'verification_failures', 'all_verifications_passed', 'all_verifications_failed']
    assert features == expected_features

    # Verify security score range (0-3)
    assert result['security_score'].min() >= 0
    assert result['security_score'].max() <= 3

    # Verify security score calculation
    for idx in result.index:
        expected_score = (result.loc[idx, 'avs_match'] +
                         result.loc[idx, 'cvv_result'] +
                         result.loc[idx, 'three_ds_flag'])
        assert result.loc[idx, 'security_score'] == expected_score

    # Verify verification failures
    assert (result['verification_failures'] == 3 - result['security_score']).all()

    # Verify all passed/failed flags
    assert (result[result['security_score'] == 3]['all_verifications_passed'] == 1).all()
    assert (result[result['security_score'] == 0]['all_verifications_failed'] == 1).all()


# Test interaction features

def test_create_interaction_features():
    """Test interaction feature creation."""
    # Create a DataFrame with prerequisite features
    df = pd.DataFrame({
        'is_new_account': [1, 0, 1, 0],
        'promo_used': [1, 1, 0, 0],
        'is_late_night_local': [1, 1, 0, 0],
        'is_micro_transaction': [1, 0, 1, 0],
        'is_large_transaction': [1, 0, 1, 0],
        'high_risk_distance': [1, 1, 0, 0],
        'country_mismatch': [1, 0, 1, 0],
        'verification_failures': [2, 0, 1, 0],
        'is_high_frequency_user': [1, 1, 0, 0]
    })

    result, features = create_interaction_features(df)

    # Verify all features are created
    assert len(features) == 6

    # Test new_account_with_promo (scenario #3)
    assert result.loc[0, 'new_account_with_promo'] == 1  # is_new_account=1, promo_used=1
    assert result.loc[1, 'new_account_with_promo'] == 0  # is_new_account=0
    assert result.loc[2, 'new_account_with_promo'] == 0  # promo_used=0

    # Test late_night_micro_transaction (scenario #1)
    assert result.loc[0, 'late_night_micro_transaction'] == 1  # both are 1
    assert result.loc[1, 'late_night_micro_transaction'] == 0  # is_micro_transaction=0

    # Test high_value_long_distance (scenario #2)
    assert result.loc[0, 'high_value_long_distance'] == 1  # both are 1
    assert result.loc[1, 'high_value_long_distance'] == 0  # high_risk_distance=1 but is_large=0

    # Test foreign_card_failed_verification
    assert result.loc[0, 'foreign_card_failed_verification'] == 1  # country_mismatch=1, failures>0
    assert result.loc[1, 'foreign_card_failed_verification'] == 0  # country_mismatch=0

    # Test triple_risk_combo
    assert result.loc[0, 'triple_risk_combo'] == 1  # all three conditions met
    assert result.loc[1, 'triple_risk_combo'] == 0  # is_new_account=0


def test_interaction_features_binary_output():
    """Test that all interaction features are binary (0 or 1)."""
    df = pd.DataFrame({
        'is_new_account': [1, 0],
        'promo_used': [1, 0],
        'is_late_night_local': [1, 0],
        'is_micro_transaction': [1, 0],
        'is_large_transaction': [1, 0],
        'high_risk_distance': [1, 0],
        'country_mismatch': [1, 0],
        'verification_failures': [1, 0],
        'is_high_frequency_user': [1, 0]
    })

    result, features = create_interaction_features(df)

    # All interaction features should be 0 or 1
    for feat in features:
        assert set(result[feat].unique()).issubset({0, 1})
