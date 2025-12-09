"""
Feature importance analysis utilities for fd3 notebook.

This module provides functions for extracting and analyzing feature importance
from trained models using both XGBoost's built-in gain metric and SHAP values.

SHAP (SHapley Additive exPlanations) values provide a more theoretically grounded
measure of feature importance by computing each feature's contribution to individual
predictions. Global SHAP importance is computed as the mean absolute SHAP value
across all samples.
"""

from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb


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


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute SHAP-based feature importance using XGBoost's native pred_contribs.

    This uses XGBoost's built-in SHAP computation which is fast and exact for
    tree-based models. Global importance is computed as mean(|SHAP value|) for
    each feature across all samples.

    Args:
        model: Trained sklearn Pipeline with XGBClassifier
        X: Feature matrix (preprocessed, ready for model)
        feature_names: List of feature names in order after preprocessing
        verbose: If True, print progress information

    Returns:
        Tuple of:
        - DataFrame with 'feature', 'shap_importance', and 'mean_shap' columns
        - Raw SHAP values matrix (n_samples, n_features)
    """
    if verbose:
        print("=" * 100)
        print("Computing SHAP Values using XGBoost Native Interface")
        print("=" * 100)

    # Extract XGBoost classifier and get the booster
    xgb_model = model.named_steps['classifier']
    booster = xgb_model.get_booster()

    # Apply preprocessor to get numeric features
    preprocessor = model.named_steps['preprocessor']
    X_processed = preprocessor.transform(X)

    # Create DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X_processed)

    # Compute SHAP values using pred_contribs
    # Returns (n_samples, n_features + 1) where last column is bias
    if verbose:
        print(f"\nComputing SHAP values for {len(X):,} samples...")

    shap_values = booster.predict(dmatrix, pred_contribs=True)

    # Remove the bias column (last column)
    shap_values = shap_values[:, :-1]

    if verbose:
        print(f"  ✓ SHAP values computed: shape {shap_values.shape}")

    # Compute global importance as mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)  # Signed mean (direction of effect)

    # Create DataFrame
    shap_importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_abs_shap,
        'mean_shap': mean_shap
    }).sort_values('shap_importance', ascending=False)

    if verbose:
        print(f"  ✓ Global SHAP importance computed")
        print("=" * 100)

    return shap_importance_df, shap_values


def compare_importance_methods(
    gain_importance_df: pd.DataFrame,
    shap_importance_df: pd.DataFrame,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare feature rankings between XGBoost gain and SHAP importance.

    Args:
        gain_importance_df: DataFrame with 'feature' and 'importance' (gain)
        shap_importance_df: DataFrame with 'feature' and 'shap_importance'
        top_n: Number of top features to compare

    Returns:
        DataFrame comparing both importance methods
    """
    # Merge the two DataFrames
    gain_df = gain_importance_df.copy()
    gain_df['gain_rank'] = range(1, len(gain_df) + 1)

    shap_df = shap_importance_df.copy()
    shap_df['shap_rank'] = range(1, len(shap_df) + 1)

    comparison_df = gain_df.merge(
        shap_df[['feature', 'shap_importance', 'mean_shap', 'shap_rank']],
        on='feature'
    )

    # Calculate rank difference
    comparison_df['rank_diff'] = comparison_df['gain_rank'] - comparison_df['shap_rank']

    # Sort by average rank
    comparison_df['avg_rank'] = (comparison_df['gain_rank'] + comparison_df['shap_rank']) / 2
    comparison_df = comparison_df.sort_values('avg_rank')

    return comparison_df.head(top_n)


def print_shap_importance_summary(
    shap_importance_df: pd.DataFrame,
    shap_values: np.ndarray,
    top_n: int = 20
) -> None:
    """
    Print a formatted summary of SHAP-based feature importance.

    Args:
        shap_importance_df: DataFrame with SHAP importance values
        shap_values: Raw SHAP values matrix
        top_n: Number of top features to display
    """
    print("=" * 100)
    print("SHAP FEATURE IMPORTANCE SUMMARY")
    print("=" * 100)

    print(f"\nTop {top_n} Features by Mean |SHAP Value|:")
    print("-" * 100)
    for i, (_, row) in enumerate(shap_importance_df.head(top_n).iterrows(), 1):
        direction = "↑ fraud" if row['mean_shap'] > 0 else "↓ fraud"
        print(f"  {i:2d}. {row['feature']:40s} - "
              f"Importance: {row['shap_importance']:.6f}  ({direction})")

    print("\n" + "=" * 100)
    print("SHAP VALUE STATISTICS:")
    print("=" * 100)

    # Overall statistics
    total_abs_shap = np.abs(shap_values).sum(axis=1).mean()
    print(f"\n  • Average total |SHAP| per sample: {total_abs_shap:.4f}")
    print(f"  • Number of features: {shap_values.shape[1]}")
    print(f"  • Number of samples analyzed: {shap_values.shape[0]:,}")

    # Feature concentration
    cumulative_importance = (
        shap_importance_df['shap_importance'].cumsum() /
        shap_importance_df['shap_importance'].sum()
    )
    print(f"\n  • Top 5 features: {cumulative_importance.iloc[4]:.1%} of total importance")
    print(f"  • Top 10 features: {cumulative_importance.iloc[9]:.1%} of total importance")

    print("\n" + "=" * 100)
    print("✓ SHAP importance analysis complete")


def print_importance_comparison(
    comparison_df: pd.DataFrame
) -> None:
    """
    Print a comparison of XGBoost gain vs SHAP importance rankings.

    Args:
        comparison_df: DataFrame from compare_importance_methods()
    """
    print("=" * 100)
    print("FEATURE IMPORTANCE: XGBoost Gain vs SHAP Comparison")
    print("=" * 100)

    print("\n{:<40s} {:>10s} {:>10s} {:>12s}".format(
        "Feature", "Gain Rank", "SHAP Rank", "Difference"
    ))
    print("-" * 100)

    for _, row in comparison_df.iterrows():
        diff_str = f"{row['rank_diff']:+d}" if row['rank_diff'] != 0 else "="
        print(f"{row['feature']:<40s} {row['gain_rank']:>10d} "
              f"{row['shap_rank']:>10d} {diff_str:>12s}")

    # Calculate correlation
    from scipy import stats
    corr, _ = stats.spearmanr(comparison_df['gain_rank'], comparison_df['shap_rank'])
    print(f"\nSpearman rank correlation: {corr:.3f}")

    # Identify major disagreements
    major_diffs = comparison_df[abs(comparison_df['rank_diff']) >= 5]
    if len(major_diffs) > 0:
        print(f"\nFeatures with rank difference >= 5:")
        for _, row in major_diffs.iterrows():
            print(f"  • {row['feature']}: Gain #{row['gain_rank']} vs SHAP #{row['shap_rank']}")

    print("\n" + "=" * 100)
    print("KEY INSIGHT:")
    print("=" * 100)
    print("""
  • XGBoost Gain: Measures how much a feature improves the model when used in splits
  • SHAP Values: Measures each feature's contribution to individual predictions

  High correlation suggests consistent feature importance across methods.
  Differences may indicate features that are important for specific subsets of data.
""")
    print("=" * 100)
    print("✓ Importance comparison complete")
