"""Configuration for fraud detection feature engineering pipeline."""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class FeatureConfig:
    """Type-safe configuration for feature engineering.

    Stores training-time statistics (quantile thresholds) and feature lists
    for consistent feature engineering during inference.

    Attributes:
        amount_95th_percentile: 95th percentile of transaction amounts (for is_large_transaction)
        total_transactions_75th_percentile: 75th percentile of user transaction counts (for is_high_frequency_user)
        shipping_distance_75th_percentile: 75th percentile of shipping distances (for high_risk_distance)
        timezone_mapping: Mapping of country codes to capital city timezones
        final_features: List of 30 selected features for model input
        date_col: Name of datetime column (default: 'transaction_time')
        country_col: Name of country column (default: 'country')
    """

    amount_95th_percentile: float
    total_transactions_75th_percentile: float
    shipping_distance_75th_percentile: float
    timezone_mapping: Dict[str, str]
    final_features: List[str]
    date_col: str = 'transaction_time'
    country_col: str = 'country'

    @classmethod
    def from_training_data(cls, train_df):
        """Create configuration from training dataset.

        Calculates quantile thresholds from training data and sets up
        timezone mappings and final feature list.

        Args:
            train_df: Training DataFrame with engineered features

        Returns:
            FeatureConfig instance with calculated thresholds
        """
        from .features import get_country_timezone_mapping, get_final_feature_names

        return cls(
            amount_95th_percentile=round(float(train_df['amount'].quantile(0.95)), 2),
            total_transactions_75th_percentile=float(train_df['total_transactions_user'].quantile(0.75)),
            shipping_distance_75th_percentile=round(float(train_df['shipping_distance_km'].quantile(0.75)), 2),
            timezone_mapping=get_country_timezone_mapping(),
            final_features=get_final_feature_names()
        )

    def save(self, path: str):
        """Save configuration to JSON file.

        Args:
            path: File path to save configuration (e.g., 'models/feature_config.json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file.

        Args:
            path: File path to load configuration from

        Returns:
            FeatureConfig instance loaded from file
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)
