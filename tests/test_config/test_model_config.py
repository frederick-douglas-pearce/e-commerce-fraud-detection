"""
Tests for src/config/model_config.py

Tests ModelConfig and FeatureListsConfig classes
"""

import pytest
import json
from pathlib import Path
from src.deployment.config.model_config import FeatureListsConfig, ModelConfig


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

    def test_load_hyperparameters_xgboost_from_metadata(self):
        """Test loading XGBoost hyperparameters from metadata"""
        params = ModelConfig.load_hyperparameters('xgboost', source='metadata')

        # Check it's a dictionary
        assert isinstance(params, dict)

        # Check expected parameters exist
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

    def test_load_hyperparameters_missing_metadata_raises_error(self, tmp_path, monkeypatch):
        """Test that missing metadata file raises FileNotFoundError"""
        # Point to a non-existent metadata file
        monkeypatch.setattr(ModelConfig, 'DEFAULT_METADATA_PATH', tmp_path / "nonexistent.json")

        with pytest.raises(FileNotFoundError, match="Could not load hyperparameters"):
            ModelConfig.load_hyperparameters('xgboost', source='metadata')

    def test_default_paths_exist(self):
        """Test that default path attributes are defined"""
        assert hasattr(ModelConfig, 'DEFAULT_METADATA_PATH')
        assert hasattr(ModelConfig, 'DEFAULT_LOGS_DIR')

        # Check they're Path objects
        assert isinstance(ModelConfig.DEFAULT_METADATA_PATH, Path)
        assert isinstance(ModelConfig.DEFAULT_LOGS_DIR, Path)
