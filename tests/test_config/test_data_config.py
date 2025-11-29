"""
Tests for src/config/data_config.py

Tests DataConfig class for data loading and splitting configuration.
"""

import pytest
from pathlib import Path
from src.config.data_config import DataConfig


class TestDataConfig:
    """Tests for DataConfig class"""

    def test_default_random_seed_constant(self):
        """Test that DEFAULT_RANDOM_SEED is defined correctly"""
        assert hasattr(DataConfig, 'DEFAULT_RANDOM_SEED')
        assert DataConfig.DEFAULT_RANDOM_SEED == 1

    def test_test_size_constant(self):
        """Test that TEST_SIZE constant is defined correctly"""
        assert hasattr(DataConfig, 'TEST_SIZE')
        assert DataConfig.TEST_SIZE == 0.2

    def test_val_size_constant(self):
        """Test that VAL_SIZE constant is defined correctly"""
        assert hasattr(DataConfig, 'VAL_SIZE')
        assert DataConfig.VAL_SIZE == 0.25

    def test_default_data_dir_constant(self):
        """Test that DEFAULT_DATA_DIR constant is defined correctly"""
        assert hasattr(DataConfig, 'DEFAULT_DATA_DIR')
        assert DataConfig.DEFAULT_DATA_DIR == Path("data")

    def test_default_data_file_constant(self):
        """Test that DEFAULT_DATA_FILE constant is defined correctly"""
        assert hasattr(DataConfig, 'DEFAULT_DATA_FILE')
        assert DataConfig.DEFAULT_DATA_FILE == "transactions.csv"

    def test_target_column_constant(self):
        """Test that TARGET_COLUMN constant is defined correctly"""
        assert hasattr(DataConfig, 'TARGET_COLUMN')
        assert DataConfig.TARGET_COLUMN == "is_fraud"

    def test_get_data_path_default(self):
        """Test get_data_path with default parameters"""
        path = DataConfig.get_data_path()
        assert isinstance(path, Path)
        assert path == Path("data") / "transactions.csv"

    def test_get_data_path_custom_dir(self):
        """Test get_data_path with custom directory"""
        path = DataConfig.get_data_path(data_dir="custom_dir")
        assert isinstance(path, Path)
        assert path == Path("custom_dir") / "transactions.csv"

    def test_get_data_path_custom_filename(self):
        """Test get_data_path with custom filename"""
        path = DataConfig.get_data_path(filename="custom.csv")
        assert isinstance(path, Path)
        assert path == Path("data") / "custom.csv"

    def test_get_data_path_custom_both(self):
        """Test get_data_path with both custom directory and filename"""
        path = DataConfig.get_data_path(data_dir="my_data", filename="test.csv")
        assert isinstance(path, Path)
        assert path == Path("my_data") / "test.csv"

    def test_get_random_seed_default(self):
        """Test get_random_seed returns default when no seed provided"""
        seed = DataConfig.get_random_seed()
        assert seed == 1

    def test_get_random_seed_custom(self):
        """Test get_random_seed returns custom seed when provided"""
        seed = DataConfig.get_random_seed(seed=42)
        assert seed == 42

    def test_get_random_seed_zero(self):
        """Test get_random_seed handles seed=0 correctly"""
        seed = DataConfig.get_random_seed(seed=0)
        assert seed == 0

    def test_get_split_config_returns_dict(self):
        """Test that get_split_config returns a dictionary"""
        config = DataConfig.get_split_config()
        assert isinstance(config, dict)

    def test_get_split_config_has_expected_keys(self):
        """Test that split config has expected keys"""
        config = DataConfig.get_split_config()
        assert 'test_size' in config
        assert 'val_size' in config
        assert 'stratify_column' in config

    def test_get_split_config_correct_values(self):
        """Test that split config has correct values"""
        config = DataConfig.get_split_config()
        assert config['test_size'] == 0.2
        assert config['val_size'] == 0.25
        assert config['stratify_column'] == 'is_fraud'
