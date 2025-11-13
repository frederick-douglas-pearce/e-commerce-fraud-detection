"""Fraud detection feature engineering pipeline.

This package provides production-ready feature engineering for fraud detection,
including scikit-learn compatible transformers and configuration management.

Main exports:
    FraudFeatureTransformer: Sklearn-compatible transformer for feature engineering
    FeatureConfig: Configuration dataclass for storing training-time parameters
"""

from .transformer import FraudFeatureTransformer
from .config import FeatureConfig

__all__ = ['FraudFeatureTransformer', 'FeatureConfig']
