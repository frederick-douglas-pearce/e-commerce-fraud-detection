"""
Data configuration for fraud detection project.

Centralizes data loading parameters, split ratios, and random seeds
to ensure consistency across all scripts (bias_variance_analysis.py, train.py).
"""

import os
from pathlib import Path


class DataConfig:
    """Configuration for data loading and splitting."""

    # Default random seed for reproducibility
    # Can be overridden by passing random_seed parameter to functions
    DEFAULT_RANDOM_SEED = 1

    # Train/validation/test split ratios
    # First split: 80% train+val, 20% test
    # Second split: 75% train, 25% val (from train+val)
    # Results in 60% train, 20% val, 20% test
    TEST_SIZE = 0.2  # 20% for test set
    VAL_SIZE = 0.25  # 25% of train+val -> 20% of total

    # Data paths
    DEFAULT_DATA_DIR = Path("data")
    DEFAULT_DATA_FILE = "transactions.csv"

    # Target column name
    TARGET_COLUMN = "is_fraud"

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
