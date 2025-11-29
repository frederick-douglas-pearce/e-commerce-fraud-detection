"""
Tests for src/config/model_config.py

Tests ModelConfig and FeatureListsConfig classes
"""

import pytest
import json
from pathlib import Path
from src.config.model_config import FeatureListsConfig, ModelConfig


class TestFeatureListsConfig:
    """Tests for FeatureListsConfig class"""

    def test_load_default_feature_lists(self):
        """Test loading feature lists from default location"""
        config = FeatureListsConfig.load()

        # Check all required keys exist
        assert 'categorical' in config
        assert 'continuous_numeric' in config
        assert 'binary' in config
        assert 'all_features' in config

    def test_feature_lists_correct_structure(self):
        """Test that feature lists have correct structure"""
        config = FeatureListsConfig.load()

        # Check types
        assert isinstance(config['categorical'], list)
        assert isinstance(config['continuous_numeric'], list)
        assert isinstance(config['binary'], list)
        assert isinstance(config['all_features'], list)

    def test_feature_lists_counts(self):
        """Test that feature lists have expected counts"""
        config = FeatureListsConfig.load()

        # Expected counts: 1 categorical, 12 continuous_numeric, 17 binary = 30 total
        assert len(config['categorical']) == 1
        assert len(config['continuous_numeric']) == 12
        assert len(config['binary']) == 17
        assert len(config['all_features']) == 30

    def test_all_features_matches_sum(self):
        """Test that all_features contains all features from categorical + continuous + binary"""
        config = FeatureListsConfig.load()

        combined = config['categorical'] + config['continuous_numeric'] + config['binary']
        assert len(config['all_features']) == len(combined)
        assert set(config['all_features']) == set(combined)

    def test_feature_lists_no_duplicates(self):
        """Test that each feature list has no duplicates"""
        config = FeatureListsConfig.load()

        assert len(config['categorical']) == len(set(config['categorical']))
        assert len(config['continuous_numeric']) == len(set(config['continuous_numeric']))
        assert len(config['binary']) == len(set(config['binary']))
        assert len(config['all_features']) == len(set(config['all_features']))

    def test_feature_lists_no_overlap(self):
        """Test that categorical, continuous_numeric, and binary don't overlap"""
        config = FeatureListsConfig.load()

        cat_set = set(config['categorical'])
        cont_set = set(config['continuous_numeric'])
        bin_set = set(config['binary'])

        # Check no overlap
        assert len(cat_set & cont_set) == 0
        assert len(cat_set & bin_set) == 0
        assert len(cont_set & bin_set) == 0

    def test_get_categorical_features(self):
        """Test get_categorical_features() method"""
        features = FeatureListsConfig.get_categorical_features()
        assert isinstance(features, list)
        assert len(features) == 1
        assert 'channel' in features

    def test_get_continuous_numeric_features(self):
        """Test get_continuous_numeric_features() method"""
        features = FeatureListsConfig.get_continuous_numeric_features()
        assert isinstance(features, list)
        assert len(features) == 12

    def test_get_binary_features(self):
        """Test get_binary_features() method"""
        features = FeatureListsConfig.get_binary_features()
        assert isinstance(features, list)
        assert len(features) == 17

    def test_get_all_features(self):
        """Test get_all_features() method"""
        features = FeatureListsConfig.get_all_features()
        assert isinstance(features, list)
        assert len(features) == 30

    def test_load_nonexistent_file_raises_error(self):
        """Test that loading from nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            FeatureListsConfig.load(source="nonexistent_file.json")


class TestModelConfig:
    """Tests for ModelConfig class"""

    def test_get_param_grid_xgboost(self):
        """Test get_param_grid returns valid XGBoost parameter grid"""
        param_grid = ModelConfig.get_param_grid('xgboost')

        # Check it's a dictionary
        assert isinstance(param_grid, dict)

        # Check expected parameters exist
        assert 'classifier__n_estimators' in param_grid
        assert 'classifier__max_depth' in param_grid
        assert 'classifier__learning_rate' in param_grid
        assert 'classifier__gamma' in param_grid
        assert 'classifier__scale_pos_weight' in param_grid

    def test_get_param_grid_random_forest(self):
        """Test get_param_grid returns valid Random Forest parameter grid"""
        param_grid = ModelConfig.get_param_grid('random_forest')

        # Check it's a dictionary
        assert isinstance(param_grid, dict)

        # Check expected parameters exist
        assert 'classifier__n_estimators' in param_grid
        assert 'classifier__max_depth' in param_grid
        assert 'classifier__min_samples_split' in param_grid
        assert 'classifier__min_samples_leaf' in param_grid
        assert 'classifier__class_weight' in param_grid

    def test_get_param_grid_invalid_model_type(self):
        """Test get_param_grid raises error for invalid model type"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelConfig.get_param_grid('invalid_model')

    def test_load_hyperparameters_xgboost_from_metadata(self):
        """Test loading XGBoost hyperparameters from metadata"""
        params = ModelConfig.load_hyperparameters('xgboost', source='metadata')

        # Check it's a dictionary
        assert isinstance(params, dict)

        # Check expected parameters exist (either from metadata or fallback)
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'learning_rate' in params
        assert 'eval_metric' in params

    def test_load_hyperparameters_with_random_seed(self):
        """Test that random_seed parameter is added to hyperparameters"""
        params = ModelConfig.load_hyperparameters('xgboost', source='metadata', random_seed=42)

        assert 'random_state' in params
        assert params['random_state'] == 42

    def test_load_hyperparameters_invalid_model_type(self):
        """Test loading hyperparameters with invalid model type raises error"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelConfig.load_hyperparameters('invalid_model', source='metadata')

    def test_fallback_params_exist(self):
        """Test that fallback parameters are defined"""
        # Should not raise errors
        assert hasattr(ModelConfig, 'FALLBACK_XGBOOST_PARAMS')
        assert hasattr(ModelConfig, 'FALLBACK_RANDOM_FOREST_PARAMS')

        # Check they're dictionaries with expected keys
        assert isinstance(ModelConfig.FALLBACK_XGBOOST_PARAMS, dict)
        assert isinstance(ModelConfig.FALLBACK_RANDOM_FOREST_PARAMS, dict)

        assert 'n_estimators' in ModelConfig.FALLBACK_XGBOOST_PARAMS
        assert 'n_estimators' in ModelConfig.FALLBACK_RANDOM_FOREST_PARAMS

    def test_default_param_grids_exist(self):
        """Test that default parameter grids are defined"""
        assert hasattr(ModelConfig, 'DEFAULT_XGBOOST_PARAM_GRID')
        assert hasattr(ModelConfig, 'DEFAULT_RANDOM_FOREST_PARAM_GRID')

        # Check they're dictionaries
        assert isinstance(ModelConfig.DEFAULT_XGBOOST_PARAM_GRID, dict)
        assert isinstance(ModelConfig.DEFAULT_RANDOM_FOREST_PARAM_GRID, dict)
