"""
Data loading and preprocessing utilities for exploratory data analysis.

This module provides general-purpose functions for loading datasets,
performing train/validation/test splits, and analyzing basic data characteristics.
All functions are designed to be reusable across different datasets and projects.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def download_data_csv(
    kaggle_source: str,
    data_dir: str,
    csv_file: str
) -> None:
    """
    Download CSV file from Kaggle using the Kaggle API.

    Requires the kaggle Python package to be installed and Kaggle API credentials
    set up in `~/.kaggle/kaggle.json`. Creates data directory if it doesn't exist.

    Args:
        kaggle_source: Kaggle dataset source (e.g., 'username/dataset-name')
        data_dir: Directory path where the dataset will be saved
        csv_file: Name of the CSV file to download

    Returns:
        None

    Example:
        >>> download_data_csv('username/fraud-dataset', './data', 'fraud.csv')
        Downloading dataset from Kaggle...
        Download complete!
    """
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    if not os.path.exists(f"{data_dir}/{csv_file}"):
        print(f"\nDownloading dataset from Kaggle...")
        subprocess.run(['kaggle', 'datasets', 'download', '-d', kaggle_source, '-p', data_dir, '--unzip'])
        print("Download complete!")
    else:
        print(f"\nDataset already exists at {data_dir}/{csv_file}")


def load_data(
    file_path: str,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame.

    Args:
        file_path: Full path to the CSV file
        verbose: If True, print dataset shape and memory usage

    Returns:
        pd.DataFrame: Loaded dataset

    Example:
        >>> df = load_data('./data/fraud.csv', verbose=True)
        Dataset Shape: 100000 rows, 15 columns
        Memory Usage: 12.45 MB
    """
    df = pd.read_csv(file_path, low_memory=False)

    if verbose:
        print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"\nMemory Usage:\n{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def plot_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    label_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Visualize target variable distribution with count and percentage plots.

    Creates two subplots: one showing counts and one showing percentages.

    Args:
        df: DataFrame containing the target column
        target_col: Name of the target column to visualize
        label_names: List of label names for x-axis (e.g., ['Normal', 'Fraud']).
                     If None, uses raw values.
        figsize: Figure size as (width, height) tuple

    Returns:
        None (displays plot)

    Example:
        >>> plot_target_distribution(df, 'is_fraud', ['Normal', 'Fraud'], (12, 4))
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Count plot
    df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'])
    axes[0].set_title('Target Variable Distribution (Count)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Count')
    if label_names:
        axes[0].set_xticklabels(label_names, rotation=0)

    # Percentage plot
    df[target_col].value_counts(normalize=True).plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'])
    axes[1].set_title('Target Variable Distribution (Percentage)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel(target_col)
    axes[1].set_ylabel('Percentage')
    if label_names:
        axes[1].set_xticklabels(label_names, rotation=0)
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

    plt.tight_layout()
    plt.show()


def analyze_target_stats(
    df: pd.DataFrame,
    target_col: str,
    label_names: Optional[List[str]] = None,
    imbalance_threshold: float = 10.0,
    plot: bool = True,
    figsize: Tuple[int, int] = (12, 4)
) -> dict:
    """
    Analyze target variable distribution and check for class imbalance.

    Prints target distribution statistics and optionally creates visualizations.
    Returns a dictionary with imbalance metrics.

    Args:
        df: DataFrame containing the target column
        target_col: Name of the target column to analyze
        label_names: List of label names for display (e.g., ['Normal', 'Fraud'])
        imbalance_threshold: Threshold for warning about class imbalance (default: 10.0)
        plot: If True, display target distribution plot
        figsize: Figure size for plot

    Returns:
        dict: Dictionary containing:
            - 'distribution': Series with value counts (normalized)
            - 'imbalance_ratio': Ratio of majority to minority class
            - 'is_imbalanced': Boolean indicating if imbalance exceeds threshold

    Example:
        >>> stats = analyze_target_stats(df, 'is_fraud', ['Normal', 'Fraud'])
        Target Distribution (%):
        0    97.8
        1     2.2

        Warning: Large class imbalance (44.5) detected!
    """
    # Target distribution
    target_dist = df[target_col].value_counts(normalize=True)
    print("\nTarget Distribution (%):")
    print(target_dist * 100)

    # Check class imbalance
    target_vals = target_dist.values
    target_ratio = target_vals[0] / target_vals[1] if len(target_vals) > 1 else 1.0

    is_imbalanced = target_ratio > imbalance_threshold
    if is_imbalanced:
        print(f"\nWarning: Large class imbalance ({target_ratio:.1f}) detected!")
    else:
        print(f"\nClass imbalance = {target_ratio:.1f}")

    if plot:
        plot_target_distribution(df, target_col, label_names, figsize)

    return {
        'distribution': target_dist,
        'imbalance_ratio': target_ratio,
        'is_imbalanced': is_imbalanced
    }


def analyze_feature_stats(
    df: pd.DataFrame,
    categorical_features: List[str],
    numeric_features: List[str],
    id_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    top_n: int = 5
) -> None:
    """
    Print summary statistics for categorical and numeric features.

    For categorical features, shows unique value counts and top values.
    For numeric features, shows standard statistical summary (mean, std, quartiles, etc.).

    Args:
        df: DataFrame containing the features
        categorical_features: List of categorical feature names
        numeric_features: List of numeric feature names
        id_cols: Optional list of ID columns to exclude from analysis
        target_col: Optional target column to exclude from analysis
        top_n: Number of top values to show for categorical features (default: 5)

    Returns:
        None (prints statistics to console)

    Example:
        >>> analyze_feature_stats(df, ['channel', 'country'], ['amount', 'age'])
        Categorical Features (2): ['channel', 'country']

        Categorical Features Summary:

        channel:
          Unique values: 3
          Top 5 values:
        web     5000
        mobile  3000
        app     2000
        Name: channel, dtype: int64
        ...
    """
    # Summary for categorical features
    print(f"\nCategorical Features ({len(categorical_features)}): {categorical_features}")
    if len(categorical_features) > 0:
        print("\nCategorical Features Summary:")
        for col in categorical_features:
            print(f"\n{col}:")
            print(f"  Unique values: {df[col].nunique()}")
            print(f"  Top {top_n} values:\n{df[col].value_counts().head(top_n)}")

    # Statistical summary for numerical features
    print(f"\n\nNumeric Features ({len(numeric_features)}): {numeric_features}")
    if len(numeric_features) > 0:
        print("\nNumeric Features Summary:")
        print(df[numeric_features].describe())


def split_train_val_test(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, validation, and test sets with optional stratification.

    Performs a stratified split if target_col is provided. The splits are randomly
    shuffled and indices are reset.

    Args:
        df: DataFrame to split
        target_col: Column name to use for stratification (optional).
                   If None, no stratification is performed.
        train_ratio: Proportion of data for training set (default: 0.6)
        val_ratio: Proportion of data for validation set (default: 0.2)
        test_ratio: Proportion of data for test set (default: 0.2)
        random_state: Random seed for reproducibility
        verbose: If True, print split sizes

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (train_df, val_df, test_df)

    Raises:
        ValueError: If train_ratio + val_ratio + test_ratio != 1.0

    Example:
        >>> train, val, test = split_train_val_test(
        ...     df,
        ...     target_col='is_fraud',
        ...     train_ratio=0.6,
        ...     val_ratio=0.2,
        ...     test_ratio=0.2,
        ...     random_state=42,
        ...     verbose=True
        ... )
        All rows in the original dataframe are contained within splits: True

        Train set: 6000 rows (60.0%)
        Validation set: 2000 rows (20.0%)
        Test set: 2000 rows (20.0%)
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    n = len(df)

    # Stratification column
    strat_col = df[target_col] if target_col else None

    # Generate test dataset
    full_train_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=strat_col,
        random_state=random_state
    )
    test_df = test_df.reset_index(drop=True)

    # Generate train and validation splits
    val_ft_ratio = val_ratio / (1 - test_ratio)
    strat_col_train = full_train_df[target_col] if target_col else None

    train_df, val_df = train_test_split(
        full_train_df,
        test_size=val_ft_ratio,
        stratify=strat_col_train,
        random_state=random_state
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    if verbose:
        all_contained = len(train_df) + len(val_df) + len(test_df) == len(df)
        print(f"All rows in the original dataframe are contained within splits: {all_contained}")
        print(f"\nTrain set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
        print(f"Validation set: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
        print(f"Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")

    return train_df, val_df, test_df
