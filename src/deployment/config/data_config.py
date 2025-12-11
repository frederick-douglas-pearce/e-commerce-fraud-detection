"""
Data configuration for fraud detection project.

Centralizes data loading parameters, split ratios, and random seeds
to ensure consistency across all scripts (bias_variance_analysis.py, train.py).

All default values are loaded from deployment_defaults.json (single source of truth).
"""

import json
from pathlib import Path


# Load defaults from JSON file (single source of truth)
_CONFIG_PATH = Path(__file__).parent / "deployment_defaults.json"
with open(_CONFIG_PATH) as f:
    _DEFAULTS = json.load(f)

_DATA_DEFAULTS = _DEFAULTS["data"]


class DataConfig:
    """Configuration for data loading and splitting."""

    # Default random seed for reproducibility
    # Can be overridden by passing random_seed parameter to functions
    DEFAULT_RANDOM_SEED: int = _DATA_DEFAULTS["default_random_seed"]

    # Train/validation/test split ratios
    # First split: 80% train+val, 20% test
    # Second split: 75% train, 25% val (from train+val)
    # Results in 60% train, 20% val, 20% test
    TEST_SIZE: float = _DATA_DEFAULTS["test_size"]
    VAL_SIZE: float = _DATA_DEFAULTS["val_size"]

    # Data paths
    DEFAULT_DATA_DIR: Path = Path(_DATA_DEFAULTS["default_data_dir"])
    DEFAULT_DATA_FILE: str = _DATA_DEFAULTS["default_data_file"]

    # Target column name
    TARGET_COLUMN: str = _DATA_DEFAULTS["target_column"]

    @classmethod
    def get_data_path(cls, data_dir: str = None, filename: str = None) -> Path:
        """Get the full path to the data file.

        Args:
            data_dir: Directory containing data files (default: 'data')
            filename: Name of the data file (default: 'transactions.csv')

        Returns:
            Path object to the data file
        """
        data_dir = Path(data_dir) if data_dir else cls.DEFAULT_DATA_DIR
        filename = filename if filename else cls.DEFAULT_DATA_FILE
        return data_dir / filename

    @classmethod
    def get_random_seed(cls, seed: int = None) -> int:
        """Get random seed, using default if not specified.

        Args:
            seed: Custom random seed (optional)

        Returns:
            The random seed to use
        """
        return seed if seed is not None else cls.DEFAULT_RANDOM_SEED

    @classmethod
    def get_split_config(cls) -> dict:
        """Get train/val/test split configuration.

        Returns:
            Dictionary with split parameters
        """
        return {
            'test_size': cls.TEST_SIZE,
            'val_size': cls.VAL_SIZE,
            'stratify_column': cls.TARGET_COLUMN
        }
