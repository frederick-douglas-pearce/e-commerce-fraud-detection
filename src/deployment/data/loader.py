"""
Data loading and splitting utilities.

Provides consistent data loading and train/val/test splitting across all scripts.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import DataConfig


def load_and_split_data(
    data_path: str = None,
    test_size: float = None,
    val_size: float = None,
    random_seed: int = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw transaction data and split into train/val/test sets.

    Args:
        data_path: Path to the CSV data file (default: from DataConfig)
        test_size: Fraction of data for test set (default: from DataConfig, 0.2)
        val_size: Fraction of train+val for validation (default: from DataConfig, 0.25)
        random_seed: Random seed for reproducibility (default: from DataConfig, 1)
        verbose: Whether to print loading information (default: True)

    Returns:
        Tuple of (train_df, val_df, test_df) as pandas DataFrames

    Raises:
        FileNotFoundError: If the data file doesn't exist

    Examples:
        >>> train_df, val_df, test_df = load_and_split_data()
        >>> train_df, val_df, test_df = load_and_split_data(random_seed=42)
        >>> train_df, val_df, test_df = load_and_split_data(data_path='data/custom.csv')
    """
    # Use defaults from config if not specified
    if data_path is None:
        csv_path = DataConfig.get_data_path()
    else:
        csv_path = Path(data_path)

    if test_size is None:
        test_size = DataConfig.TEST_SIZE

    if val_size is None:
        val_size = DataConfig.VAL_SIZE

    if random_seed is None:
        random_seed = DataConfig.DEFAULT_RANDOM_SEED

    # Check if file exists
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Please ensure the file exists.\n"
            f"For the default dataset, download from Kaggle: "
            f"umuttuygurr/e-commerce-fraud-detection-dataset"
        )

    # Load raw CSV data
    if verbose:
        print(f"Loading raw transaction data...")
        print(f"  Data file: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)

    if verbose:
        print(f"  Total samples: {len(df):,}")
        print(f"  Fraud rate: {df[DataConfig.TARGET_COLUMN].mean():.2%}")

    # Split into train/val/test (60/20/20 by default)
    # First split: separate test set (80% train+val, 20% test)
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[DataConfig.TARGET_COLUMN],
        random_state=random_seed
    )

    # Second split: separate train and validation (75% train, 25% val of train+val)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        stratify=train_val_df[DataConfig.TARGET_COLUMN],
        random_state=random_seed
    )

    if verbose:
        print(f"\n  Training set:   {len(train_df):,} samples ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Validation set: {len(val_df):,} samples ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test set:       {len(test_df):,} samples ({len(test_df)/len(df)*100:.1f}%)")
        print(f"\n  Fraud rate - Train: {train_df[DataConfig.TARGET_COLUMN].mean():.2%}")
        print(f"  Fraud rate - Val:   {val_df[DataConfig.TARGET_COLUMN].mean():.2%}")
        print(f"  Fraud rate - Test:  {test_df[DataConfig.TARGET_COLUMN].mean():.2%}")

    return train_df, val_df, test_df
