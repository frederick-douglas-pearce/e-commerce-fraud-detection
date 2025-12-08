"""
EDA and feature engineering utilities for fd1 notebook.

This module provides functions for:
- Data loading and preprocessing
- Exploratory data analysis and visualization
- Feature engineering (temporal, interaction, percentile-based)

Example:
    >>> from src.fd1_nb import load_data, split_train_val_test
    >>> from src.fd1_nb import analyze_vif, analyze_correlations
    >>> from src.fd1_nb import create_temporal_features, create_interaction_features
"""

from .data_utils import (
    download_data_csv,
    load_data,
    plot_target_distribution,
    analyze_target_stats,
    analyze_feature_stats,
    split_train_val_test,
)

from .eda_utils import (
    calculate_mi_scores,
    calculate_numeric_correlations,
    calculate_vif,
    plot_numeric_distributions,
    analyze_vif,
    analyze_correlations,
    plot_box_plots,
    analyze_temporal_patterns,
    analyze_categorical_fraud_rates,
    plot_categorical_fraud_rates,
    analyze_mutual_information,
)

from .feature_engineering import (
    convert_utc_to_local_time,
    create_temporal_features,
    create_interaction_features,
    create_percentile_based_features,
)

__all__ = [
    # Data utilities
    'download_data_csv',
    'load_data',
    'plot_target_distribution',
    'analyze_target_stats',
    'analyze_feature_stats',
    'split_train_val_test',
    # EDA utilities
    'calculate_mi_scores',
    'calculate_numeric_correlations',
    'calculate_vif',
    'plot_numeric_distributions',
    'analyze_vif',
    'analyze_correlations',
    'plot_box_plots',
    'analyze_temporal_patterns',
    'analyze_categorical_fraud_rates',
    'plot_categorical_fraud_rates',
    'analyze_mutual_information',
    # Feature engineering
    'convert_utc_to_local_time',
    'create_temporal_features',
    'create_interaction_features',
    'create_percentile_based_features',
]
