"""Feature engineering functions for fraud detection.

This module contains all feature engineering logic extracted from the EDA notebook.
Functions are designed to be called by FraudFeatureTransformer with appropriate configuration.
"""

import pandas as pd
import numpy as np
import pytz
from typing import Dict, List, Tuple


def get_country_timezone_mapping() -> Dict[str, str]:
    """Create mapping of country codes to capital city timezones.

    Returns:
        Dictionary mapping country codes to timezone strings
    """
    return {
        'US': 'America/New_York',
        'GB': 'Europe/London',
        'FR': 'Europe/Paris',
        'DE': 'Europe/Berlin',
        'IT': 'Europe/Rome',
        'ES': 'Europe/Madrid',
        'CA': 'America/Toronto',
        'AU': 'Australia/Sydney',
        'JP': 'Asia/Tokyo',
        'BR': 'America/Sao_Paulo'
    }


def get_final_feature_names() -> List[str]:
    """Return list of 30 final selected features for model input.

    This represents the feature selection decisions from EDA:
    - 5 original numeric features
    - 5 original categorical features
    - 6 temporal features (local time only)
    - 4 amount features
    - 3 user behavior features
    - 3 geographic features
    - 1 security feature
    - 3 interaction features (fraud scenario-specific)

    Returns:
        List of 30 feature names
    """
    # Original numeric (5)
    original_numeric = [
        'account_age_days',
        'total_transactions_user',
        'avg_amount_user',
        'amount',
        'shipping_distance_km'
    ]

    # Original categorical (1) - only channel is truly categorical (web/app)
    original_categorical = [
        'channel'
    ]

    # Original binary (4) - these are 0/1 flags, not categorical
    original_binary = [
        'promo_used',
        'avs_match',
        'cvv_result',
        'three_ds_flag'
    ]

    # Temporal - local time only (6)
    temporal_local = [
        'hour_local',
        'day_of_week_local',
        'month_local',
        'is_weekend_local',
        'is_late_night_local',
        'is_business_hours_local'
    ]

    # Amount features (4)
    amount_features = [
        'amount_deviation',
        'amount_vs_avg_ratio',
        'is_micro_transaction',
        'is_large_transaction'
    ]

    # User behavior (3)
    behavior_features = [
        'transaction_velocity',
        'is_new_account',
        'is_high_frequency_user'
    ]

    # Geographic (3)
    geographic_features = [
        'country_mismatch',
        'high_risk_distance',
        'zero_distance'
    ]

    # Security (1)
    security_features = [
        'security_score'
    ]

    # Interaction features - fraud scenario specific (3)
    interaction_features = [
        'new_account_with_promo',
        'late_night_micro_transaction',
        'high_value_long_distance'
    ]

    return (original_numeric + original_categorical + original_binary + temporal_local +
            amount_features + behavior_features + geographic_features +
            security_features + interaction_features)


def convert_to_local_time(df: pd.DataFrame, date_col: str, country_col: str,
                          timezone_mapping: Dict[str, str]) -> pd.DataFrame:
    """Convert UTC timestamps to local time based on country capital timezone.

    This function performs strict validation to ensure input timestamps are
    timezone-aware in UTC. It approximates local time using the timezone of
    each country's capital city.

    Args:
        df: Input DataFrame
        date_col: Name of UTC datetime column
        country_col: Name of country column
        timezone_mapping: Dict mapping country codes to timezone strings

    Returns:
        DataFrame with new 'local_time' column (timezone-naive)

    Raises:
        ValueError: If date_col is not timezone-aware or not in UTC
    """
    df = df.copy()

    # Initialize as timezone-naive (will hold local times)
    df['local_time'] = df[date_col].dt.tz_localize(None)

    for country, tz_str in timezone_mapping.items():
        mask = df[country_col] == country
        if mask.sum() > 0:
            dt_series = df.loc[mask, date_col]

            # Strict validation - require timezone-aware UTC input
            if dt_series.dt.tz is None:
                raise ValueError(
                    f"Column '{date_col}' must be timezone-aware in UTC. "
                    f"Ensure timestamps are parsed with utc=True during data loading."
                )
            if str(dt_series.dt.tz) != 'UTC':
                raise ValueError(
                    f"Column '{date_col}' must be in UTC timezone, but found: {dt_series.dt.tz}"
                )

            # Convert UTC to local timezone, then remove timezone info
            utc_times = dt_series
            local_times = utc_times.dt.tz_convert(tz_str).dt.tz_localize(None)
            df.loc[mask, 'local_time'] = local_times

    return df


def create_temporal_features(df: pd.DataFrame, date_col: str,
                            use_local_time: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    """Create temporal features from datetime column.

    Generates 6 temporal features:
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - month: Month (1-12)
    - is_weekend: 1 if Saturday/Sunday, else 0
    - is_late_night: 1 if hour >= 23 or hour <= 4, else 0
    - is_business_hours: 1 if 9 <= hour <= 17, else 0

    Args:
        df: Input DataFrame
        date_col: Name of datetime column to use
        use_local_time: If True, adds '_local' suffix to feature names

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    df = df.copy()
    suffix = '_local' if use_local_time else ''

    # Extract temporal components
    df[f'hour{suffix}'] = df[date_col].dt.hour
    df[f'day_of_week{suffix}'] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6
    df[f'month{suffix}'] = df[date_col].dt.month

    # Derived flags
    df[f'is_weekend{suffix}'] = (df[f'day_of_week{suffix}'] >= 5).astype(int)
    df[f'is_late_night{suffix}'] = ((df[f'hour{suffix}'] >= 23) | (df[f'hour{suffix}'] <= 4)).astype(int)
    df[f'is_business_hours{suffix}'] = ((df[f'hour{suffix}'] >= 9) & (df[f'hour{suffix}'] <= 17)).astype(int)

    features = [
        f'hour{suffix}',
        f'day_of_week{suffix}',
        f'month{suffix}',
        f'is_weekend{suffix}',
        f'is_late_night{suffix}',
        f'is_business_hours{suffix}'
    ]

    return df, features


def create_amount_features(df: pd.DataFrame, amount_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Create transaction amount-based features.

    Generates 4 amount features:
    - amount_deviation: Absolute deviation from user's average
    - amount_vs_avg_ratio: Ratio of transaction to user average
    - is_micro_transaction: 1 if amount <= $5, else 0 (card testing pattern)
    - is_large_transaction: 1 if amount >= threshold (95th percentile), else 0

    Args:
        df: Input DataFrame
        amount_threshold: 95th percentile threshold from training data

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    df = df.copy()

    # Deviation from user's average
    df['amount_deviation'] = np.abs(df['amount'] - df['avg_amount_user'])

    # Ratio (handle division by zero)
    df['amount_vs_avg_ratio'] = np.where(
        df['avg_amount_user'] > 0,
        df['amount'] / df['avg_amount_user'],
        0
    )

    # Micro transaction flag (card testing)
    df['is_micro_transaction'] = (df['amount'] <= 5).astype(int)

    # Large transaction flag (using training set threshold)
    df['is_large_transaction'] = (df['amount'] >= amount_threshold).astype(int)

    features = [
        'amount_deviation',
        'amount_vs_avg_ratio',
        'is_micro_transaction',
        'is_large_transaction'
    ]

    return df, features


def create_user_behavior_features(df: pd.DataFrame,
                                  transaction_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Create user behavior features.

    Generates 3 behavior features:
    - transaction_velocity: Transactions per day of account age
    - is_new_account: 1 if account_age_days <= 30, else 0 (promo abuse pattern)
    - is_high_frequency_user: 1 if total_transactions >= threshold (75th percentile), else 0

    Args:
        df: Input DataFrame
        transaction_threshold: 75th percentile threshold from training data

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    df = df.copy()

    # Transaction velocity (handle division by zero)
    df['transaction_velocity'] = np.where(
        df['account_age_days'] > 0,
        df['total_transactions_user'] / df['account_age_days'],
        0
    )

    # New account flag
    df['is_new_account'] = (df['account_age_days'] <= 30).astype(int)

    # High frequency user flag (using training set threshold)
    df['is_high_frequency_user'] = (df['total_transactions_user'] >= transaction_threshold).astype(int)

    features = [
        'transaction_velocity',
        'is_new_account',
        'is_high_frequency_user'
    ]

    return df, features


def create_geographic_features(df: pd.DataFrame,
                               distance_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Create geographic features.

    Generates 3 geographic features:
    - country_mismatch: 1 if user country != card issuing country, else 0
    - high_risk_distance: 1 if shipping_distance >= threshold (75th percentile), else 0
    - zero_distance: 1 if shipping_distance == 0 (billing = shipping), else 0

    Args:
        df: Input DataFrame
        distance_threshold: 75th percentile threshold from training data

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    df = df.copy()

    # Country mismatch flag
    df['country_mismatch'] = (df['country'] != df['bin_country']).astype(int)

    # High risk distance flag (using training set threshold)
    df['high_risk_distance'] = (df['shipping_distance_km'] >= distance_threshold).astype(int)

    # Zero distance flag (lower risk - billing = shipping)
    df['zero_distance'] = (df['shipping_distance_km'] == 0).astype(int)

    features = [
        'country_mismatch',
        'high_risk_distance',
        'zero_distance'
    ]

    return df, features


def create_security_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create security verification features.

    Generates 4 security features (only security_score used in final 30):
    - security_score: Sum of avs_match + cvv_result + three_ds_flag (0-3)
    - verification_failures: 3 - security_score
    - all_verifications_passed: 1 if security_score == 3, else 0
    - all_verifications_failed: 1 if security_score == 0, else 0

    Args:
        df: Input DataFrame

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    df = df.copy()

    # Composite security score
    df['security_score'] = df['avs_match'] + df['cvv_result'] + df['three_ds_flag']

    # Verification failures (inverse of security score)
    df['verification_failures'] = 3 - df['security_score']

    # All verifications passed
    df['all_verifications_passed'] = (df['security_score'] == 3).astype(int)

    # All verifications failed
    df['all_verifications_failed'] = (df['security_score'] == 0).astype(int)

    features = [
        'security_score',
        'verification_failures',
        'all_verifications_passed',
        'all_verifications_failed'
    ]

    return df, features


def create_interaction_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Create fraud scenario-specific interaction features.

    Generates 6 interaction features (only 3 used in final 30):
    - new_account_with_promo: Fraud scenario #3 (promo abuse from new accounts)
    - late_night_micro_transaction: Fraud scenario #1 (card testing at midnight)
    - high_value_long_distance: Fraud scenario #2 variant (large amounts shipped far)
    - foreign_card_failed_verification: Country mismatch + failed verification
    - new_high_velocity_account: New account with high transaction velocity
    - triple_risk_combo: New account + promo + verification failures

    Args:
        df: Input DataFrame (must have prerequisite features)

    Returns:
        Tuple of (DataFrame with new features, list of feature names)
    """
    df = df.copy()

    # Scenario #3: Promo abuse from new accounts
    df['new_account_with_promo'] = (
        (df['is_new_account'] == 1) & (df['promo_used'] == 1)
    ).astype(int)

    # Scenario #1: Card testing at midnight
    df['late_night_micro_transaction'] = (
        (df['is_late_night_local'] == 1) & (df['is_micro_transaction'] == 1)
    ).astype(int)

    # Scenario #2 variant: Large amounts shipped far
    df['high_value_long_distance'] = (
        (df['is_large_transaction'] == 1) & (df['high_risk_distance'] == 1)
    ).astype(int)

    # Additional high-risk combinations
    df['foreign_card_failed_verification'] = (
        (df['country_mismatch'] == 1) & (df['verification_failures'] > 0)
    ).astype(int)

    df['new_high_velocity_account'] = (
        (df['is_new_account'] == 1) & (df['is_high_frequency_user'] == 1)
    ).astype(int)

    df['triple_risk_combo'] = (
        (df['is_new_account'] == 1) &
        (df['promo_used'] == 1) &
        (df['verification_failures'] > 0)
    ).astype(int)

    features = [
        'new_account_with_promo',
        'late_night_micro_transaction',
        'foreign_card_failed_verification',
        'new_high_velocity_account',
        'high_value_long_distance',
        'triple_risk_combo'
    ]

    return df, features
