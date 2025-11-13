"""Scikit-learn compatible transformer for fraud detection feature engineering."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import FeatureConfig
from .features import (
    convert_to_local_time,
    create_temporal_features,
    create_amount_features,
    create_user_behavior_features,
    create_geographic_features,
    create_security_features,
    create_interaction_features
)


class FraudFeatureTransformer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for fraud detection features.

    This transformer applies the complete feature engineering pipeline:
    1. Datetime preprocessing (UTC timezone validation)
    2. Timezone conversion (UTC → local time by country)
    3. Temporal feature creation (UTC and local)
    4. Amount, behavior, geographic, security features
    5. Interaction features (fraud scenario-specific)
    6. Feature selection (30 final features)

    The transformer follows sklearn's fit/transform pattern:
    - fit(): Calculates quantile thresholds from training data
    - transform(): Applies feature engineering with stored thresholds

    Can be used in sklearn Pipeline for seamless integration with models.

    Attributes:
        config: FeatureConfig with quantile thresholds and feature lists
    """

    def __init__(self, config: FeatureConfig = None):
        """Initialize transformer with optional configuration.

        Args:
            config: FeatureConfig instance. If None, must call fit() before transform()
        """
        self.config = config

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer by calculating configuration from training data.

        Calculates quantile thresholds (95th percentile for amount, 75th for
        transaction count and distance) from training set. These thresholds
        are stored in config and used during transform().

        Args:
            X: Training DataFrame with raw features
            y: Target variable (ignored, included for sklearn compatibility)

        Returns:
            self
        """
        self.config = FeatureConfig.from_training_data(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply full feature engineering pipeline.

        Transforms raw transaction data into 30 engineered features ready
        for model input. Uses quantile thresholds calculated during fit().

        Args:
            X: Input DataFrame with raw features

        Returns:
            DataFrame with 30 engineered features

        Raises:
            ValueError: If transformer has not been fitted
        """
        if self.config is None:
            raise ValueError(
                "Transformer must be fit before transform. "
                "Call fit() or load() first to set configuration."
            )

        # Step 1: Preprocess datetime
        X = self._preprocess(X)

        # Step 2: Engineer features
        X = self._engineer_features(X)

        # Step 3: Return only final selected features
        return X[self.config.final_features]

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform in one step.

        Convenience method that calls fit() then transform().

        Args:
            X: Training DataFrame with raw features
            y: Target variable (ignored, included for sklearn compatibility)

        Returns:
            DataFrame with 30 engineered features
        """
        return self.fit(X, y).transform(X)

    def _preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """Preprocess datetime column with strict UTC validation.

        Converts transaction_time to timezone-aware datetime in UTC.
        This is required for timezone conversion in feature engineering.

        Args:
            X: Input DataFrame

        Returns:
            DataFrame with processed datetime column
        """
        X = X.copy()

        # Convert to timezone-aware datetime in UTC
        # This will fail if timezone info is missing or incorrect
        X[self.config.date_col] = pd.to_datetime(
            X[self.config.date_col],
            utc=True,
            errors='coerce'
        )

        return X

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps in correct order.

        This method orchestrates the complete feature engineering pipeline,
        ensuring features are created in the correct dependency order.

        Order of operations:
        1. Timezone conversion (UTC → local)
        2. Temporal features (UTC)
        3. Temporal features (local)
        4. Amount features
        5. User behavior features
        6. Geographic features
        7. Security features
        8. Interaction features (requires features from steps 3-7)

        Args:
            X: DataFrame with preprocessed datetime

        Returns:
            DataFrame with all engineered features
        """
        # Step 1: Convert UTC to local time
        X = convert_to_local_time(
            X,
            self.config.date_col,
            self.config.country_col,
            self.config.timezone_mapping
        )

        # Step 2: Create temporal features (UTC)
        X, _ = create_temporal_features(X, self.config.date_col, use_local_time=False)

        # Step 3: Create temporal features (local)
        X, _ = create_temporal_features(X, 'local_time', use_local_time=True)

        # Step 4: Create amount features
        X, _ = create_amount_features(X, self.config.amount_95th_percentile)

        # Step 5: Create user behavior features
        X, _ = create_user_behavior_features(X, self.config.total_transactions_75th_percentile)

        # Step 6: Create geographic features
        X, _ = create_geographic_features(X, self.config.shipping_distance_75th_percentile)

        # Step 7: Create security features
        X, _ = create_security_features(X)

        # Step 8: Create interaction features (requires features from steps 3-7)
        X, _ = create_interaction_features(X)

        return X

    def save(self, path: str):
        """Save transformer configuration to JSON file.

        Only the configuration (quantile thresholds, feature lists) is saved,
        not the transformer object itself. This makes the saved file lightweight
        and version-control friendly.

        Args:
            path: File path to save configuration (e.g., 'models/transformer_config.json')

        Raises:
            ValueError: If transformer has not been fitted
        """
        if self.config is None:
            raise ValueError(
                "Cannot save unfitted transformer. Call fit() first."
            )
        self.config.save(path)

    @classmethod
    def load(cls, path: str):
        """Load transformer from saved configuration file.

        Creates a new transformer instance with configuration loaded from JSON.

        Args:
            path: File path to load configuration from

        Returns:
            FraudFeatureTransformer instance with loaded configuration
        """
        config = FeatureConfig.load(path)
        return cls(config=config)
