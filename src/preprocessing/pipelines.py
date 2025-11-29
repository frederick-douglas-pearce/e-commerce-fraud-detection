"""
Preprocessing pipeline factories.

Provides standardized preprocessing pipelines for different model types,
ensuring consistency across all scripts (bias_variance_analysis.py, train.py, notebook).
"""

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from ..config import FeatureListsConfig


class PreprocessingPipelineFactory:
    """Factory for creating preprocessing pipelines."""

    @classmethod
    def create_tree_pipeline(
        cls,
        categorical_features: List[str] = None,
        continuous_numeric: List[str] = None,
        binary: List[str] = None
    ) -> ColumnTransformer:
        """Create preprocessing pipeline for tree-based models (Random Forest, XGBoost).

        Tree-based models don't need feature scaling, so we only apply:
        - OrdinalEncoder for categorical features
        - Passthrough for continuous numeric and binary features

        Args:
            categorical_features: List of categorical feature names (default: from config)
            continuous_numeric: List of continuous numeric feature names (default: from config)
            binary: List of binary feature names (default: from config)

        Returns:
            ColumnTransformer configured for tree-based models

        Examples:
            >>> preprocessor = PreprocessingPipelineFactory.create_tree_pipeline()
            >>> X_transformed = preprocessor.fit_transform(X_train)
        """
        # Load from config if not provided
        if categorical_features is None or continuous_numeric is None or binary is None:
            feature_config = FeatureListsConfig.load()
            categorical_features = categorical_features or feature_config['categorical']
            continuous_numeric = continuous_numeric or feature_config['continuous_numeric']
            binary = binary or feature_config['binary']

        return ColumnTransformer(
            transformers=[
                (
                    'cat',
                    OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1
                    ),
                    categorical_features
                ),
                (
                    'rest',
                    'passthrough',
                    continuous_numeric + binary
                )
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

    @classmethod
    def create_logistic_pipeline(
        cls,
        categorical_features: List[str] = None,
        continuous_numeric: List[str] = None,
        binary: List[str] = None
    ) -> ColumnTransformer:
        """Create preprocessing pipeline for logistic regression.

        Logistic regression benefits from:
        - StandardScaler for continuous numeric features
        - OneHotEncoder for categorical features
        - Binary features are scaled like continuous features

        Args:
            categorical_features: List of categorical feature names (default: from config)
            continuous_numeric: List of continuous numeric feature names (default: from config)
            binary: List of binary feature names (default: from config)

        Returns:
            ColumnTransformer configured for logistic regression

        Examples:
            >>> preprocessor = PreprocessingPipelineFactory.create_logistic_pipeline()
            >>> X_transformed = preprocessor.fit_transform(X_train)
        """
        # Load from config if not provided
        if categorical_features is None or continuous_numeric is None or binary is None:
            feature_config = FeatureListsConfig.load()
            categorical_features = categorical_features or feature_config['categorical']
            continuous_numeric = continuous_numeric or feature_config['continuous_numeric']
            binary = binary or feature_config['binary']

        return ColumnTransformer(
            transformers=[
                (
                    'num',
                    StandardScaler(),
                    continuous_numeric + binary  # Scale both continuous and binary
                ),
                (
                    'cat',
                    OneHotEncoder(drop='first', handle_unknown='ignore'),
                    categorical_features
                )
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )

    @classmethod
    def create_pipeline(
        cls,
        model_type: str,
        categorical_features: List[str] = None,
        continuous_numeric: List[str] = None,
        binary: List[str] = None
    ) -> ColumnTransformer:
        """Create preprocessing pipeline for specified model type.

        Args:
            model_type: Type of model ('tree', 'logistic')
            categorical_features: List of categorical feature names (default: from config)
            continuous_numeric: List of continuous numeric feature names (default: from config)
            binary: List of binary feature names (default: from config)

        Returns:
            ColumnTransformer configured for the specified model type

        Raises:
            ValueError: If model_type is not supported

        Examples:
            >>> preprocessor = PreprocessingPipelineFactory.create_pipeline('tree')
            >>> preprocessor = PreprocessingPipelineFactory.create_pipeline('logistic')
        """
        if model_type.lower() in ['tree', 'random_forest', 'xgboost', 'rf', 'xgb']:
            return cls.create_tree_pipeline(categorical_features, continuous_numeric, binary)
        elif model_type.lower() in ['logistic', 'logistic_regression', 'lr']:
            return cls.create_logistic_pipeline(categorical_features, continuous_numeric, binary)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Use 'tree' or 'logistic' (or aliases like 'xgboost', 'random_forest', 'lr')"
            )
