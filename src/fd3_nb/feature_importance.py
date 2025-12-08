"""
Feature importance analysis utilities for fd3 notebook.

This module provides functions for extracting and analyzing feature importance
from trained models.
"""

from typing import Any, List

import numpy as np
import pandas as pd


def extract_feature_importance(
    model: Any,
    feature_names: List[str],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model pipeline.

    Args:
        model: Trained sklearn Pipeline with 'classifier' step
        feature_names: List of feature names in order they appear after preprocessing
        verbose: If True, print header

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted by importance descending
    """
    # Extract XGBoost classifier from pipeline
    xgb_model = model.named_steps['classifier']

    # Get feature importance scores (gain)
    importance_scores = xgb_model.feature_importances_

    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)

    if verbose:
        print("=" * 100)
        print("XGBoost Feature Importance (Gain) - Top 20 Features (Final Retrained Model)")
        print("=" * 100)

    return feature_importance_df


def print_feature_importance_summary(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20
) -> None:
    """
    Print a formatted summary of feature importance analysis.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to display
    """
    print("=" * 100)
    print("FEATURE IMPORTANCE SUMMARY")
    print("=" * 100)

    # Display top features
    print(f"\nTop {top_n} Most Important Features for Fraud Detection:")
    print("-" * 100)
    for i, (idx, row) in enumerate(feature_importance_df.head(top_n).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:40s} - Importance: {row['importance']:.6f}")

    print("\n" + "=" * 100)
    print("KEY INSIGHTS:")
    print("=" * 100)

    # Analyze top features
    top_5 = feature_importance_df.head(5)['feature'].tolist()
    print(f"\nTop 5 fraud indicators:")
    for i, feat in enumerate(top_5, 1):
        print(f"  {i}. {feat}")

    # Calculate cumulative importance
    cumulative_importance = (
        feature_importance_df['importance'].cumsum() /
        feature_importance_df['importance'].sum()
    )

    print(f"\nModel concentration:")
    print(f"  • Top 5 features account for {cumulative_importance.iloc[4]:.1%} of total importance")
    print(f"  • Top 10 features account for {cumulative_importance.iloc[9]:.1%} of total importance")

    print("\n" + "=" * 100)
    print("✓ Feature importance analysis complete")
