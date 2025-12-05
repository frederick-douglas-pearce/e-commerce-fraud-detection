"""
Feature engineering utilities for creating temporal and interaction features.

This module provides general-purpose functions for timezone conversion, temporal feature
extraction, and interaction feature creation. All functions are designed to be reusable
across different datasets and projects.
"""

from typing import Dict, List, Tuple, Optional

import pandas as pd
import pytz


def convert_utc_to_local_time(
    df: pd.DataFrame,
    date_col: str,
    country_col: str,
    timezone_mapping: Dict[str, str],
    output_col: str = 'local_time',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Convert UTC timestamps to local time based on country-to-timezone mapping.

    Creates a new column with timezone-naive local timestamps based on each record's
    country value and the provided timezone mapping.

    Args:
        df: DataFrame containing the date and country columns
        date_col: Name of the datetime column (must be timezone-aware in UTC)
        country_col: Name of the country column
        timezone_mapping: Dictionary mapping country codes to timezone strings
                          (e.g., {'US': 'America/New_York', 'GB': 'Europe/London'})
        output_col: Name of the output column for local timestamps (default: 'local_time')
        verbose: If True, print conversion progress

    Returns:
        pd.DataFrame: Copy of input DataFrame with additional local_time column

    Raises:
        ValueError: If date_col is not timezone-aware or not in UTC

    Example:
        >>> tz_map = {'US': 'America/New_York', 'GB': 'Europe/London'}
        >>> df_local = convert_utc_to_local_time(
        ...     df,
        ...     date_col='transaction_time',
        ...     country_col='country',
        ...     timezone_mapping=tz_map
        ... )
        Converting UTC to local time by country...
          ✓ Converted 10000 transactions to local time
    """
    df = df.copy()

    # Initialize output column (timezone-naive, will represent local times)
    df[output_col] = df[date_col].dt.tz_localize(None)

    if verbose:
        print(f"Converting UTC to local time by country...")

    for country, tz_str in timezone_mapping.items():
        mask = df[country_col] == country
        if mask.sum() > 0:
            # Get the datetime series
            dt_series = df.loc[mask, date_col]

            # Validate that input is timezone-aware and in UTC
            if dt_series.dt.tz is None:
                raise ValueError(
                    f"Column '{date_col}' must be timezone-aware in UTC. "
                    f"Ensure timestamps are parsed with utc=True during data loading."
                )

            # Validate it's in UTC
            if str(dt_series.dt.tz) != 'UTC':
                raise ValueError(
                    f"Column '{date_col}' must be in UTC timezone, "
                    f"but found: {dt_series.dt.tz}"
                )

            # Convert UTC to target local timezone
            utc_times = dt_series
            local_times = utc_times.dt.tz_convert(tz_str).dt.tz_localize(None)
            df.loc[mask, output_col] = local_times

    if verbose:
        print(f"  ✓ Converted {len(df)} records to local time")

    return df


def create_temporal_features(
    df: pd.DataFrame,
    date_col: str,
    suffix: str = '',
    late_night_hours: Tuple[int, int] = (23, 4),
    business_hours: Tuple[int, int] = (9, 17),
    weekend_threshold: int = 5
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create temporal features from datetime column.

    Extracts hour, day of week, month, and creates binary indicators for
    weekend, late night, and business hours.

    Args:
        df: DataFrame containing the datetime column
        date_col: Name of the datetime column to extract features from
        suffix: Optional suffix to append to feature names (e.g., '_local', '_utc')
        late_night_hours: Tuple of (start_hour, end_hour) for late night period
                          (default: (23, 4) means 11pm-4am)
        business_hours: Tuple of (start_hour, end_hour) for business hours
                        (default: (9, 17) means 9am-5pm)
        weekend_threshold: Day of week value >= this is weekend (default: 5 for Saturday/Sunday)

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Copy of DataFrame with new temporal features
            - List of created feature column names

    Example:
        >>> df_temporal, features = create_temporal_features(
        ...     df,
        ...     date_col='transaction_time',
        ...     suffix='_utc',
        ...     late_night_hours=(23, 5),
        ...     business_hours=(8, 18)
        ... )
        >>> print(features)
        ['hour_utc', 'day_of_week_utc', 'month_utc', 'is_weekend_utc',
         'is_late_night_utc', 'is_business_hours_utc']
    """
    df = df.copy()

    # Extract basic time components
    hour_col = f'hour{suffix}'
    dow_col = f'day_of_week{suffix}'
    month_col = f'month{suffix}'

    df[hour_col] = df[date_col].dt.hour
    df[dow_col] = df[date_col].dt.dayofweek  # 0=Monday, 6=Sunday
    df[month_col] = df[date_col].dt.month

    # Derived temporal features
    weekend_col = f'is_weekend{suffix}'
    late_night_col = f'is_late_night{suffix}'
    business_col = f'is_business_hours{suffix}'

    df[weekend_col] = (df[dow_col] >= weekend_threshold).astype(int)

    # Late night hours (e.g., 23-4 means >= 23 OR <= 4)
    late_start, late_end = late_night_hours
    if late_start > late_end:  # Wraps around midnight
        df[late_night_col] = ((df[hour_col] >= late_start) | (df[hour_col] <= late_end)).astype(int)
    else:
        df[late_night_col] = ((df[hour_col] >= late_start) & (df[hour_col] <= late_end)).astype(int)

    # Business hours
    biz_start, biz_end = business_hours
    df[business_col] = ((df[hour_col] >= biz_start) & (df[hour_col] <= biz_end)).astype(int)

    features_created = [hour_col, dow_col, month_col, weekend_col, late_night_col, business_col]

    return df, features_created


def create_interaction_features(
    df: pd.DataFrame,
    interaction_config: List[Dict[str, any]]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create interaction features from combinations of existing features.

    Interaction features capture combinations of conditions that may be more
    predictive than individual features alone.

    Args:
        df: DataFrame containing the base features
        interaction_config: List of dictionaries, each defining an interaction:
            - 'name': str - Name of the new feature
            - 'conditions': List[str] - List of boolean expressions to AND together
            - 'operator': str - 'and' or 'or' (default: 'and')

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Copy of DataFrame with new interaction features
            - List of created feature column names

    Example:
        >>> config = [
        ...     {
        ...         'name': 'new_account_with_promo',
        ...         'conditions': ['is_new_account == 1', 'promo_used == 1'],
        ...         'operator': 'and'
        ...     },
        ...     {
        ...         'name': 'late_night_micro_transaction',
        ...         'conditions': ['is_late_night_local == 1', 'is_micro_transaction == 1'],
        ...         'operator': 'and'
        ...     }
        ... ]
        >>> df_int, features = create_interaction_features(df, config)
        >>> print(features)
        ['new_account_with_promo', 'late_night_micro_transaction']

    Note:
        - All conditions must evaluate to boolean Series
        - The 'operator' determines how conditions are combined ('and' or 'or')
        - For 'and', all conditions must be True
        - For 'or', at least one condition must be True
    """
    df = df.copy()
    features_created = []

    for interaction in interaction_config:
        feature_name = interaction['name']
        conditions = interaction['conditions']
        operator = interaction.get('operator', 'and')

        # Evaluate each condition
        condition_results = []
        for condition in conditions:
            # Evaluate the condition string in the context of the DataFrame
            result = df.eval(condition)
            condition_results.append(result)

        # Combine conditions based on operator
        if operator == 'and':
            # All conditions must be True
            combined = condition_results[0]
            for cond in condition_results[1:]:
                combined = combined & cond
        elif operator == 'or':
            # At least one condition must be True
            combined = condition_results[0]
            for cond in condition_results[1:]:
                combined = combined | cond
        else:
            raise ValueError(f"Unsupported operator: {operator}. Use 'and' or 'or'.")

        df[feature_name] = combined.astype(int)
        features_created.append(feature_name)

    return df, features_created


def create_percentile_based_features(
    df: pd.DataFrame,
    feature_config: List[Dict[str, any]],
    percentile_column: str,
    percentile_value: float
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create binary features based on percentile thresholds.

    Creates indicator variables flagging values above or below a percentile threshold.

    Args:
        df: DataFrame containing the numeric columns
        feature_config: List of dictionaries, each defining a feature:
            - 'source_col': str - Source column name
            - 'feature_name': str - New feature name
            - 'operator': str - Comparison operator ('>=', '<=', '>', '<')
            - 'percentile': float - Percentile threshold (0-1)
        percentile_column: Not used (kept for backward compatibility)
        percentile_value: Not used (kept for backward compatibility)

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            - Copy of DataFrame with new percentile-based features
            - List of created feature column names

    Example:
        >>> config = [
        ...     {
        ...         'source_col': 'amount',
        ...         'feature_name': 'is_large_transaction',
        ...         'operator': '>=',
        ...         'percentile': 0.95
        ...     },
        ...     {
        ...         'source_col': 'distance',
        ...         'feature_name': 'high_risk_distance',
        ...         'operator': '>=',
        ...         'percentile': 0.75
        ...     }
        ... ]
        >>> df_pct, features = create_percentile_based_features(df, config, None, None)
        >>> print(features)
        ['is_large_transaction', 'high_risk_distance']
    """
    df = df.copy()
    features_created = []

    for feature_def in feature_config:
        source_col = feature_def['source_col']
        feature_name = feature_def['feature_name']
        operator = feature_def['operator']
        percentile = feature_def['percentile']

        # Calculate percentile threshold
        threshold = df[source_col].quantile(percentile)

        # Apply operator
        if operator == '>=':
            df[feature_name] = (df[source_col] >= threshold).astype(int)
        elif operator == '>':
            df[feature_name] = (df[source_col] > threshold).astype(int)
        elif operator == '<=':
            df[feature_name] = (df[source_col] <= threshold).astype(int)
        elif operator == '<':
            df[feature_name] = (df[source_col] < threshold).astype(int)
        elif operator == '==':
            df[feature_name] = (df[source_col] == threshold).astype(int)
        else:
            raise ValueError(f"Unsupported operator: {operator}")

        features_created.append(feature_name)

    return df, features_created
