"""
Tests for src/deployment/preprocessing/pipelines.py

Tests PreprocessingPipelineFactory for creating preprocessing pipelines.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from src.deployment.preprocessing.pipelines import PreprocessingPipelineFactory
from src.deployment.config.model_config import FeatureListsConfig


@pytest.fixture
def sample_features():
    """Get sample feature lists from config"""
    config = FeatureListsConfig.load()
    return {
        'categorical': config['categorical'],
        'continuous_numeric': config['continuous_numeric'],
        'binary': config['binary']
    }


@pytest.fixture
def sample_data(sample_features):
    """Create sample data for testing pipelines"""
    np.random.seed(42)
    n_samples = 100

    # Create data with all feature types
    data = {}

    # Categorical features
    for feat in sample_features['categorical']:
        data[feat] = np.random.choice(['A', 'B', 'C'], n_samples)

    # Continuous numeric features
    for feat in sample_features['continuous_numeric']:
        data[feat] = np.random.randn(n_samples)

    # Binary features
    for feat in sample_features['binary']:
        data[feat] = np.random.choice([0, 1], n_samples)

    return pd.DataFrame(data)


class TestPreprocessingPipelineFactory:
    """Tests for PreprocessingPipelineFactory class"""

    def test_create_tree_pipeline_returns_column_transformer(self):
        """Test that create_tree_pipeline returns ColumnTransformer"""
        pipeline = PreprocessingPipelineFactory.create_tree_pipeline()

        assert isinstance(pipeline, ColumnTransformer)

    def test_create_logistic_pipeline_returns_column_transformer(self):
        """Test that create_logistic_pipeline returns ColumnTransformer"""
        pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        assert isinstance(pipeline, ColumnTransformer)

    def test_create_tree_pipeline_has_correct_transformers(self):
        """Test that tree pipeline has correct transformer types"""
        pipeline = PreprocessingPipelineFactory.create_tree_pipeline()

        # Get transformer names and types
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}

        # Tree pipeline should have OrdinalEncoder for categorical
        # and passthrough for continuous/binary
        assert 'cat' in transformer_dict
        assert transformer_dict['cat'] == 'OrdinalEncoder'
        assert 'rest' in transformer_dict

    def test_create_logistic_pipeline_has_correct_transformers(self):
        """Test that logistic pipeline has correct transformer types"""
        pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        # Get transformer names and types
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}

        # Logistic pipeline should have StandardScaler and OneHotEncoder
        assert 'num' in transformer_dict
        assert transformer_dict['num'] == 'StandardScaler'
        assert 'cat' in transformer_dict
        assert transformer_dict['cat'] == 'OneHotEncoder'

    def test_create_tree_pipeline_uses_default_features(self, sample_features):
        """Test that tree pipeline uses default features from config"""
        pipeline = PreprocessingPipelineFactory.create_tree_pipeline()

        # Extract feature names from transformers
        cat_features = None
        rest_features = None

        for name, transformer, features in pipeline.transformers:
            if name == 'cat':
                cat_features = features
            elif name == 'rest':
                rest_features = features

        # Should match default features from config
        assert cat_features == sample_features['categorical']
        expected_rest = sample_features['continuous_numeric'] + sample_features['binary']
        assert rest_features == expected_rest

    def test_create_logistic_pipeline_uses_default_features(self, sample_features):
        """Test that logistic pipeline uses default features from config"""
        pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        # Extract feature names from transformers
        num_features = None
        cat_features = None

        for name, transformer, features in pipeline.transformers:
            if name == 'num':
                num_features = features
            elif name == 'cat':
                cat_features = features

        # Should match default features from config
        assert cat_features == sample_features['categorical']
        expected_num = sample_features['continuous_numeric'] + sample_features['binary']
        assert num_features == expected_num

    def test_create_tree_pipeline_custom_features(self):
        """Test that tree pipeline accepts custom feature lists"""
        custom_cat = ['custom_cat']
        custom_cont = ['custom_cont_1', 'custom_cont_2']
        custom_bin = ['custom_bin_1']

        pipeline = PreprocessingPipelineFactory.create_tree_pipeline(
            categorical_features=custom_cat,
            continuous_numeric=custom_cont,
            binary=custom_bin
        )

        # Extract feature names
        cat_features = None
        rest_features = None

        for name, transformer, features in pipeline.transformers:
            if name == 'cat':
                cat_features = features
            elif name == 'rest':
                rest_features = features

        assert cat_features == custom_cat
        assert rest_features == custom_cont + custom_bin

    def test_create_logistic_pipeline_custom_features(self):
        """Test that logistic pipeline accepts custom feature lists"""
        custom_cat = ['custom_cat']
        custom_cont = ['custom_cont_1', 'custom_cont_2']
        custom_bin = ['custom_bin_1']

        pipeline = PreprocessingPipelineFactory.create_logistic_pipeline(
            categorical_features=custom_cat,
            continuous_numeric=custom_cont,
            binary=custom_bin
        )

        # Extract feature names
        num_features = None
        cat_features = None

        for name, transformer, features in pipeline.transformers:
            if name == 'num':
                num_features = features
            elif name == 'cat':
                cat_features = features

        assert cat_features == custom_cat
        assert num_features == custom_cont + custom_bin

    def test_create_pipeline_with_tree_type(self):
        """Test that create_pipeline with 'tree' type returns tree pipeline"""
        pipeline = PreprocessingPipelineFactory.create_pipeline('tree')

        assert isinstance(pipeline, ColumnTransformer)

        # Should have OrdinalEncoder
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}
        assert 'cat' in transformer_dict
        assert transformer_dict['cat'] == 'OrdinalEncoder'

    def test_create_pipeline_with_logistic_type(self):
        """Test that create_pipeline with 'logistic' type returns logistic pipeline"""
        pipeline = PreprocessingPipelineFactory.create_pipeline('logistic')

        assert isinstance(pipeline, ColumnTransformer)

        # Should have StandardScaler and OneHotEncoder
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}
        assert 'num' in transformer_dict
        assert transformer_dict['num'] == 'StandardScaler'
        assert 'cat' in transformer_dict
        assert transformer_dict['cat'] == 'OneHotEncoder'

    def test_create_pipeline_with_xgboost_alias(self):
        """Test that create_pipeline accepts 'xgboost' as tree alias"""
        pipeline = PreprocessingPipelineFactory.create_pipeline('xgboost')

        assert isinstance(pipeline, ColumnTransformer)

        # Should have OrdinalEncoder (tree-based)
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}
        assert transformer_dict['cat'] == 'OrdinalEncoder'

    def test_create_pipeline_with_random_forest_alias(self):
        """Test that create_pipeline accepts 'random_forest' as tree alias"""
        pipeline = PreprocessingPipelineFactory.create_pipeline('random_forest')

        assert isinstance(pipeline, ColumnTransformer)

        # Should have OrdinalEncoder (tree-based)
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}
        assert transformer_dict['cat'] == 'OrdinalEncoder'

    def test_create_pipeline_with_lr_alias(self):
        """Test that create_pipeline accepts 'lr' as logistic alias"""
        pipeline = PreprocessingPipelineFactory.create_pipeline('lr')

        assert isinstance(pipeline, ColumnTransformer)

        # Should have StandardScaler (logistic-based)
        transformer_dict = {name: type(transformer).__name__
                           for name, transformer, _ in pipeline.transformers}
        assert transformer_dict['num'] == 'StandardScaler'

    def test_create_pipeline_invalid_type_raises_error(self):
        """Test that create_pipeline raises error for invalid model type"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            PreprocessingPipelineFactory.create_pipeline('invalid_model')

    def test_tree_pipeline_fit_transform(self, sample_data):
        """Test that tree pipeline can fit and transform data"""
        pipeline = PreprocessingPipelineFactory.create_tree_pipeline()

        # Should not raise errors
        X_transformed = pipeline.fit_transform(sample_data)

        # Check output shape
        assert X_transformed.shape[0] == len(sample_data)
        # Should have same number of features (OrdinalEncoder doesn't expand)
        assert X_transformed.shape[1] == len(sample_data.columns)

    def test_logistic_pipeline_fit_transform(self, sample_data):
        """Test that logistic pipeline can fit and transform data"""
        pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        # Should not raise errors
        X_transformed = pipeline.fit_transform(sample_data)

        # Check output shape
        assert X_transformed.shape[0] == len(sample_data)
        # OneHotEncoder will expand categorical features
        assert X_transformed.shape[1] >= len(sample_data.columns)

    def test_tree_pipeline_ordinal_encoder_handles_unknown(self):
        """Test that OrdinalEncoder in tree pipeline handles unknown values"""
        pipeline = PreprocessingPipelineFactory.create_tree_pipeline()

        # Get encoder from pipeline
        ordinal_encoder = None
        for name, transformer, _ in pipeline.transformers:
            if name == 'cat':
                ordinal_encoder = transformer
                break

        assert ordinal_encoder is not None
        assert hasattr(ordinal_encoder, 'handle_unknown')
        assert ordinal_encoder.handle_unknown == 'use_encoded_value'
        assert ordinal_encoder.unknown_value == -1

    def test_logistic_pipeline_onehot_encoder_handles_unknown(self):
        """Test that OneHotEncoder in logistic pipeline handles unknown values"""
        pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        # Get encoder from pipeline
        onehot_encoder = None
        for name, transformer, _ in pipeline.transformers:
            if name == 'cat':
                onehot_encoder = transformer
                break

        assert onehot_encoder is not None
        assert hasattr(onehot_encoder, 'handle_unknown')
        assert onehot_encoder.handle_unknown == 'ignore'

    def test_pipeline_remainder_is_drop(self):
        """Test that both pipelines drop remaining columns"""
        tree_pipeline = PreprocessingPipelineFactory.create_tree_pipeline()
        logistic_pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        assert tree_pipeline.remainder == 'drop'
        assert logistic_pipeline.remainder == 'drop'

    def test_pipeline_verbose_feature_names_disabled(self):
        """Test that verbose_feature_names_out is disabled"""
        tree_pipeline = PreprocessingPipelineFactory.create_tree_pipeline()
        logistic_pipeline = PreprocessingPipelineFactory.create_logistic_pipeline()

        assert tree_pipeline.verbose_feature_names_out is False
        assert logistic_pipeline.verbose_feature_names_out is False
