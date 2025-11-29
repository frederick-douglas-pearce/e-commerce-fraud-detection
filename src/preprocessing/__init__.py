"""Fraud detection feature engineering pipeline.

This package provides production-ready feature engineering for fraud detection,
including scikit-learn compatible transformers and configuration management.

Main exports:
    FraudFeatureTransformer: Sklearn-compatible transformer for feature engineering
    FeatureConfig: Configuration dataclass for storing training-time parameters
    PreprocessingPipelineFactory: Factory for creating preprocessing pipelines
"""

from .transformer import FraudFeatureTransformer
from .config import FeatureConfig
from .pipelines import PreprocessingPipelineFactory

__all__ = ['FraudFeatureTransformer', 'FeatureConfig', 'PreprocessingPipelineFactory']
