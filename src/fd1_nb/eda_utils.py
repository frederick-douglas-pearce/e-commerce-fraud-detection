"""
EDA (Exploratory Data Analysis) utilities for visualization and statistical analysis.

This module provides general-purpose functions for analyzing relationships between
features and target variables, detecting multicollinearity, visualizing distributions,
and calculating feature importance metrics. All functions are designed to be reusable
across different datasets and projects.
"""

from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
from statsmodels.stats.outliers_influence import variance_inflation_factor


def _save_figure(fig: plt.Figure, save_path: Optional[str], dpi: int = 150) -> None:
    """
    Save figure to disk if save_path is provided.

    Args:
        fig: Matplotlib figure to save
        save_path: Path to save the figure (if None, figure is not saved)
        dpi: Resolution for saved figure (default: 150)
    """
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"  Figure saved: {save_path}")


def calculate_mi_scores(
    df: pd.DataFrame,
    categorical_features: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Calculate mutual information scores for categorical features against target variable.

    Mutual information measures the mutual dependence between two variables.
    Higher scores indicate stronger association with the target.

    Args:
        df: DataFrame containing the features and target
        categorical_features: List of categorical feature column names
        target_col: Name of the target column

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'mi_score'], sorted by mi_score descending

    Example:
        >>> mi_scores = calculate_mi_scores(df, ['channel', 'country'], 'is_fraud')
        >>> print(mi_scores)
               feature  mi_score
        0      channel    0.0234
        1      country    0.0156
    """
    mi_scores = []
    for feature in categorical_features:
        score = mutual_info_score(df[feature], df[target_col])
        mi_scores.append({'feature': feature, 'mi_score': score})

    mi_df = pd.DataFrame(mi_scores).sort_values('mi_score', ascending=False).reset_index(drop=True)
    return mi_df


def calculate_numeric_correlations(
    df: pd.DataFrame,
    numeric_features: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Calculate Pearson correlations for numeric features with target variable.

    Args:
        df: DataFrame containing the features and target
        numeric_features: List of numeric feature column names
        target_col: Name of the target column

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'correlation'], sorted by absolute correlation descending

    Example:
        >>> corr = calculate_numeric_correlations(df, ['amount', 'distance'], 'is_fraud')
        >>> print(corr)
             feature  correlation
        0     amount       0.1234
        1   distance      -0.0567
    """
    correlations = []
    for feature in numeric_features:
        corr = df[feature].corr(df[target_col])
        correlations.append({'feature': feature, 'correlation': corr, 'abs_correlation': abs(corr)})

    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False).reset_index(drop=True)
    return corr_df[['feature', 'correlation']]


def calculate_vif(
    df: pd.DataFrame,
    numeric_features: List[str]
) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for numeric features.

    VIF measures multicollinearity among features. Common thresholds:
    - VIF < 5: Low multicollinearity (acceptable)
    - VIF 5-10: Moderate multicollinearity (monitor)
    - VIF > 10: High multicollinearity (consider removing)

    Args:
        df: DataFrame containing the numeric features
        numeric_features: List of numeric feature column names

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'VIF'], sorted by VIF descending

    Example:
        >>> vif = calculate_vif(df, ['amount', 'avg_amount', 'total_transactions'])
        >>> print(vif)
                     feature       VIF
        0         avg_amount  15.23456
        1             amount   8.91234
        2  total_transactions   2.34567
    """
    X = df[numeric_features].values
    vif_data = []
    for i, feature in enumerate(numeric_features):
        vif = variance_inflation_factor(X, i)
        vif_data.append({'feature': feature, 'VIF': vif})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif_df


def plot_numeric_distributions(
    df: pd.DataFrame,
    numeric_features: List[str],
    figsize: Tuple[int, int] = (14, 12),
    bins: int = 50,
    ncols: int = 2,
    show_stats: bool = True,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize distributions of numeric features with histograms.

    Creates subplots with histograms showing mean and median lines for each numeric feature.

    Args:
        df: DataFrame containing the numeric features
        numeric_features: List of numeric feature column names
        figsize: Figure size as (width, height) tuple
        bins: Number of bins for histograms
        ncols: Number of columns in subplot grid
        show_stats: If True, display mean and median lines with legend
        save_path: Optional path to save the figure

    Returns:
        None (displays plot)

    Example:
        >>> plot_numeric_distributions(df, ['amount', 'age', 'distance'], bins=30)
        >>> plot_numeric_distributions(df, ['amount'], save_path='images/fd1/numeric_distributions.png')
    """
    nrows = int(np.ceil(len(numeric_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, col in enumerate(numeric_features):
        ax = axes[idx]
        df[col].hist(bins=bins, ax=ax, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

        if show_stats:
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.1f}')
            ax.legend()

    # Remove extra subplots
    for idx in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()


def analyze_vif(
    df: pd.DataFrame,
    numeric_features: List[str],
    vif_threshold_moderate: float = 5.0,
    vif_threshold_high: float = 10.0,
    figsize: Tuple[int, int] = (10, 5),
    verbose: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate and visualize VIF for multicollinearity detection.

    Computes VIF for each feature, displays results, and creates a horizontal bar chart
    showing VIF values with threshold reference lines.

    Args:
        df: DataFrame containing the numeric features
        numeric_features: List of numeric feature column names
        vif_threshold_moderate: VIF value indicating moderate multicollinearity (default: 5.0)
        vif_threshold_high: VIF value indicating high multicollinearity (default: 10.0)
        figsize: Figure size for plot
        verbose: If True, print interpretation and findings
        save_path: Optional path to save the figure

    Returns:
        pd.DataFrame: VIF results with columns ['feature', 'VIF']

    Example:
        >>> vif_df = analyze_vif(df, ['amount', 'avg_amount', 'total_transactions'])
        Variance Inflation Factor (VIF) Analysis:
        ==================================================
                     feature       VIF
                  avg_amount  15.23456
                      amount   8.91234
          total_transactions   2.34567
        ==================================================

        Interpretation:
        - VIF < 5: Low multicollinearity (acceptable)
        - VIF 5-10: Moderate multicollinearity (monitor)
        - VIF > 10: High multicollinearity (consider removing)

        Key Findings:
        - High multicollinearity detected in: ['avg_amount']
        - Consider removing or combining these features
    """
    vif_df = calculate_vif(df, numeric_features)

    if verbose:
        print("Variance Inflation Factor (VIF) Analysis:")
        print("=" * 50)
        print(vif_df.to_string(index=False))
        print("\n" + "=" * 50)
        print("\nInterpretation:")
        print(f"- VIF < {vif_threshold_moderate}: Low multicollinearity (acceptable)")
        print(f"- VIF {vif_threshold_moderate}-{vif_threshold_high}: Moderate multicollinearity (monitor)")
        print(f"- VIF > {vif_threshold_high}: High multicollinearity (consider removing)")

    # Visualization
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(vif_df['feature'], vif_df['VIF'], color='coral', edgecolor='black')
    ax.set_xlabel('VIF Value', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Variance Inflation Factor (VIF) - Multicollinearity Check', fontsize=13, fontweight='bold')
    ax.axvline(x=vif_threshold_moderate, color='orange', linestyle='--', linewidth=2,
               label=f'VIF = {vif_threshold_moderate} (Moderate)')
    ax.axvline(x=vif_threshold_high, color='red', linestyle='--', linewidth=2,
               label=f'VIF = {vif_threshold_high} (High)')
    ax.legend()
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    if verbose:
        print("\nKey Findings:")
        if vif_df['VIF'].max() > vif_threshold_high:
            high_vif = vif_df[vif_df['VIF'] > vif_threshold_high]['feature'].tolist()
            print(f"- High multicollinearity detected in: {high_vif}")
            print("- Consider removing or combining these features")
        elif vif_df['VIF'].max() > vif_threshold_moderate:
            print("- Moderate multicollinearity present but generally acceptable")
        else:
            print("- All features show low multicollinearity")

    return vif_df


def analyze_correlations(
    df: pd.DataFrame,
    numeric_features: List[str],
    target_col: str,
    figsize: Tuple[int, int] = (10, 5),
    verbose: bool = True,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate and visualize correlations of numeric features with target variable.

    Computes Pearson correlations and creates a horizontal bar chart showing correlation
    coefficients, with positive correlations in green and negative in red.

    Args:
        df: DataFrame containing the features and target
        numeric_features: List of numeric feature column names
        target_col: Name of the target column
        figsize: Figure size for plot
        verbose: If True, print correlation values and interpretation
        save_path: Optional path to save the figure

    Returns:
        pd.DataFrame: Correlation results with columns ['feature', 'correlation']

    Example:
        >>> corr_df = analyze_correlations(df, ['amount', 'age'], 'is_fraud')
        Pearson Correlation with Target (is_fraud):
        ============================================================
          feature  correlation
           amount       0.1234
              age      -0.0567
        ============================================================

        Key Insights:
        - Positive correlation: Higher values associated with target=1
        - Negative correlation: Lower values associated with target=1
        - Values closer to ±1 indicate stronger linear relationships
    """
    corr_df = calculate_numeric_correlations(df, numeric_features, target_col)

    if verbose:
        print(f"Pearson Correlation with Target ({target_col}):")
        print("=" * 60)
        print(corr_df.to_string(index=False))
        print("=" * 60)

    # Visualization
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['green' if x > 0 else 'red' for x in corr_df['correlation']]
    ax.barh(corr_df['feature'], corr_df['correlation'], color=colors, edgecolor='black')
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Correlation of Numeric Features with {target_col}', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    if verbose:
        print("\nKey Insights:")
        print("- Positive correlation: Higher values associated with target=1")
        print("- Negative correlation: Lower values associated with target=1")
        print("- Values closer to ±1 indicate stronger linear relationships")

    return corr_df


def plot_box_plots(
    df: pd.DataFrame,
    numeric_features: List[str],
    target_col: str,
    label_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 12),
    ncols: int = 2,
    save_path: Optional[str] = None
) -> None:
    """
    Compare feature distributions between target classes using box plots.

    Creates subplots with box plots showing distribution differences between
    target=0 and target=1 for each numeric feature.

    Args:
        df: DataFrame containing the features and target
        numeric_features: List of numeric feature column names
        target_col: Name of the target column
        label_names: Optional list of label names for target classes (e.g., ['Normal', 'Fraud'])
        figsize: Figure size as (width, height) tuple
        ncols: Number of columns in subplot grid
        save_path: Optional path to save the figure

    Returns:
        None (displays plot)

    Example:
        >>> plot_box_plots(df, ['amount', 'distance'], 'is_fraud', ['Normal', 'Fraud'])
    """
    nrows = int(np.ceil(len(numeric_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, col in enumerate(numeric_features):
        ax = axes[idx]
        df.boxplot(column=col, by=target_col, ax=ax, patch_artist=True)
        ax.set_title(f'{col} by {target_col}', fontsize=11, fontweight='bold')
        ax.set_xlabel(target_col)
        ax.set_ylabel(col)

        if label_names and len(label_names) == 2:
            unique_vals = sorted(df[target_col].unique())
            ax.set_xticklabels([f'{label_names[i]} ({unique_vals[i]})' for i in range(len(unique_vals))])

    # Remove extra subplots
    for idx in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle('')
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print("\nBox Plot Interpretation:")
    print("- Look for differences in medians (center line) between target classes")
    print("- Different distributions suggest the feature is discriminative")
    print("- Overlapping boxes indicate less predictive power")


def analyze_temporal_patterns(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    baseline_rate: float,
    figsize: Tuple[int, int] = (16, 10),
    risk_threshold: float = 1.2,
    save_path: Optional[str] = None
) -> None:
    """
    Analyze and visualize target patterns over time dimensions.

    Creates temporal features (hour, day_of_week, month, is_weekend) and visualizes
    target rates across these dimensions with 4 subplots.

    Args:
        df: DataFrame containing date column and target (will be modified in-place)
        date_col: Name of the datetime column
        target_col: Name of the target column
        baseline_rate: Baseline target rate for reference line
        figsize: Figure size for 2x2 subplot grid
        risk_threshold: Multiplier for identifying high-risk periods (default: 1.2 = 20% above baseline)
        save_path: Optional path to save the figure

    Returns:
        None (displays plot and prints insights)

    Note:
        This function modifies the DataFrame in-place by adding temporal feature columns.

    Example:
        >>> analyze_temporal_patterns(df, 'transaction_time', 'is_fraud', 0.022, risk_threshold=1.5)
        Temporal Insights:
        - Peak fraud hour: 2 (fraud rate: 0.0356)
        - Safest hour: 14 (fraud rate: 0.0145)
        - Weekend fraud rate: 0.0234
        - Weekday fraud rate: 0.0218

        ⚠️  High-risk hours (>20% above baseline):
           - Hour 2: 0.0356 fraud rate (1.6x baseline)
           - Hour 3: 0.0289 fraud rate (1.3x baseline)
    """
    # Extract temporal features (modifies df in-place)
    df['hour'] = df[date_col].dt.hour
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['month'] = df[date_col].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Hour of day
    ax = axes[0, 0]
    hourly_rate = df.groupby('hour')[target_col].mean()
    hourly_rate.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.axhline(y=baseline_rate, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_title('Target Rate by Hour of Day', fontsize=12, fontweight='bold')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Target Rate')
    ax.legend()

    # Day of week
    ax = axes[0, 1]
    dow_rate = df.groupby('day_of_week')[target_col].mean()
    dow_rate.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
    ax.axhline(y=baseline_rate, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_title('Target Rate by Day of Week', fontsize=12, fontweight='bold')
    ax.set_xlabel('Day of Week (0=Monday, 6=Sunday)')
    ax.set_ylabel('Target Rate')
    ax.legend()

    # Month
    ax = axes[1, 0]
    monthly_rate = df.groupby('month')[target_col].mean()
    monthly_rate.plot(kind='bar', ax=ax, color='lightgreen', edgecolor='black')
    ax.axhline(y=baseline_rate, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_title('Target Rate by Month', fontsize=12, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Target Rate')
    ax.legend()

    # Weekend vs Weekday
    ax = axes[1, 1]
    weekend_rate = df.groupby('is_weekend')[target_col].mean()
    weekend_rate.plot(kind='bar', ax=ax, color='orchid', edgecolor='black')
    ax.axhline(y=baseline_rate, color='red', linestyle='--', linewidth=2, label='Baseline')
    ax.set_title('Target Rate: Weekday vs Weekend', fontsize=12, fontweight='bold')
    ax.set_xlabel('Is Weekend')
    ax.set_ylabel('Target Rate')
    ax.set_xticklabels(['Weekday (0)', 'Weekend (1)'], rotation=0)
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    # Print insights
    print("\nTemporal Insights:")
    print(f"- Peak hour: {hourly_rate.idxmax()} (rate: {hourly_rate.max():.4f})")
    print(f"- Safest hour: {hourly_rate.idxmin()} (rate: {hourly_rate.min():.4f})")
    print(f"- Weekend rate: {weekend_rate[1]:.4f}")
    print(f"- Weekday rate: {weekend_rate[0]:.4f}")

    high_risk_hours = hourly_rate[hourly_rate > baseline_rate * risk_threshold]
    if not high_risk_hours.empty:
        print(f"\n⚠️  High-risk hours (>{(risk_threshold-1)*100:.0f}% above baseline):")
        for hour, rate in high_risk_hours.items():
            print(f"   - Hour {hour}: {rate:.4f} rate ({rate/baseline_rate:.1f}x baseline)")


def analyze_categorical_fraud_rates(
    df: pd.DataFrame,
    categorical_features: List[str],
    target_col: str,
    baseline_rate: Optional[float] = None,
    risk_threshold: float = 1.5
) -> None:
    """
    Analyze target rates for each categorical feature.

    For each categorical feature, computes and displays count, total count, and
    target rate for each category, highlighting high-risk categories.

    Args:
        df: DataFrame containing the features and target
        categorical_features: List of categorical feature column names
        target_col: Name of the target column
        baseline_rate: Optional baseline rate (if None, computed from data)
        risk_threshold: Multiplier for identifying high-risk categories (default: 1.5)

    Returns:
        None (prints analysis results)

    Example:
        >>> analyze_categorical_fraud_rates(df, ['channel', 'country'], 'is_fraud', 0.022)
        Target Rates by Categorical Features
        ================================================================================

        CHANNEL:
        --------------------------------------------------------------------------------
          channel  target_count  total_count  target_rate
              web           234         5000        4.680
           mobile           123         3000        4.100
              app            56         2000        2.800
        --------------------------------------------------------------------------------

        ⚠️  High-risk categories (>1.5x baseline rate of 2.20%):
           - web: 4.68% target rate
           - mobile: 4.10% target rate
    """
    if baseline_rate is None:
        baseline_rate = df[target_col].mean()

    print(f"Target Rates by Categorical Features")
    print("=" * 80)

    for feature in categorical_features:
        print(f"\n{feature.upper()}:")
        print("-" * 80)

        rate_df = df.groupby(feature)[target_col].agg(['sum', 'count', 'mean']).reset_index()
        rate_df.columns = [feature, 'target_count', 'total_count', 'target_rate']
        rate_df['target_rate'] = rate_df['target_rate'] * 100
        rate_df = rate_df.sort_values('target_rate', ascending=False)

        print(rate_df.to_string(index=False))

        high_risk = rate_df[rate_df['target_rate'] > baseline_rate * 100 * risk_threshold]
        if not high_risk.empty:
            print(f"\n⚠️  High-risk categories (>{risk_threshold}x baseline rate of {baseline_rate*100:.2f}%):")
            for _, row in high_risk.iterrows():
                print(f"   - {row[feature]}: {row['target_rate']:.2f}% target rate")

    print("\n" + "=" * 80)


def plot_categorical_fraud_rates(
    df: pd.DataFrame,
    categorical_features: List[str],
    target_col: str,
    baseline_rate: float,
    figsize: Tuple[int, int] = (16, 16),
    ncols: int = 2,
    save_path: Optional[str] = None
) -> None:
    """
    Visualize target rates for categorical features with bar charts.

    Creates subplots with bar charts showing target rate for each category
    of each categorical feature, with baseline reference line.

    Args:
        df: DataFrame containing the features and target
        categorical_features: List of categorical feature column names
        target_col: Name of the target column
        baseline_rate: Baseline target rate for reference line
        figsize: Figure size as (width, height) tuple
        ncols: Number of columns in subplot grid
        save_path: Optional path to save the figure

    Returns:
        None (displays plot)

    Example:
        >>> plot_categorical_fraud_rates(df, ['channel', 'country'], 'is_fraud', 0.022)
    """
    nrows = int(np.ceil(len(categorical_features) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, feature in enumerate(categorical_features):
        ax = axes[idx]
        rate = df.groupby(feature)[target_col].mean().sort_values(ascending=False)
        rate.plot(kind='bar', ax=ax, color='coral', edgecolor='black', alpha=0.8)
        ax.axhline(y=baseline_rate, color='red', linestyle='--', linewidth=2,
                   label=f'Baseline: {baseline_rate:.3f}')
        ax.set_title(f'Target Rate by {feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Target Rate')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

    # Remove extra subplots
    for idx in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    print("\nVisualization Insight:")
    print("- Categories significantly above the baseline (red line) are high-risk")
    print("- Large deviations suggest strong predictive power")


def analyze_mutual_information(
    df: pd.DataFrame,
    categorical_features: List[str],
    target_col: str,
    figsize: Tuple[int, int] = (10, 6),
    mi_threshold: float = 0.1,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate and visualize mutual information scores for categorical features.

    Computes MI scores measuring dependence between features and target, and
    creates a horizontal bar chart visualization.

    Args:
        df: DataFrame containing the features and target
        categorical_features: List of categorical feature column names
        target_col: Name of the target column
        figsize: Figure size for plot
        mi_threshold: Threshold for meaningful predictive value (default: 0.1)
        save_path: Optional path to save the figure

    Returns:
        pd.DataFrame: MI results with columns ['feature', 'mi_score']

    Example:
        >>> mi_df = analyze_mutual_information(df, ['channel', 'country'], 'is_fraud')
        Mutual Information Scores (Categorical Features):
        ============================================================
          feature  mi_score
          channel    0.0234
          country    0.0156
        ============================================================

        Interpretation:
        - Higher MI score = stronger association with target
        - MI = 0 means no mutual dependence
        - MI > 0.1 typically indicates meaningful predictive value
    """
    mi_df = calculate_mi_scores(df, categorical_features, target_col)

    print(f"\nMutual Information Scores (Categorical Features):")
    print("=" * 60)
    print(mi_df.to_string(index=False))
    print("=" * 60)
    print("\nInterpretation:")
    print("- Higher MI score = stronger association with target")
    print("- MI = 0 means no mutual dependence")
    print(f"- MI > {mi_threshold} typically indicates meaningful predictive value")

    # Visualization
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(mi_df['feature'], mi_df['mi_score'], color='teal', edgecolor='black')
    ax.set_xlabel('Mutual Information Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title(f'Mutual Information: Categorical Features vs. {target_col}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, save_path)
    plt.show()

    return mi_df
