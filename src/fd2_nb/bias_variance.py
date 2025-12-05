"""
Bias-variance analysis utilities for model diagnostics.

This module provides functions for analyzing bias-variance tradeoffs in
classification models, including train-validation gap analysis, iteration
tracking for boosting models, and cross-validation fold variance analysis.
All functions are designed to be reusable across different models and projects.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold


def calculate_train_val_gap(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = 'pr_auc'
) -> Dict[str, float]:
    """
    Calculate train-validation performance gap for a single model.

    The train-validation gap is a key indicator of overfitting:
    - Gap > 15%: Severe overfitting (high variance)
    - Gap > 10%: Moderate overfitting
    - Gap < 10%: Generally acceptable

    Args:
        model: Trained model with predict_proba() method
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to use ('pr_auc', 'roc_auc', or 'f1')

    Returns:
        Dictionary with train_score, val_score, gap, gap_pct, and diagnosis

    Example:
        >>> gap_info = calculate_train_val_gap(xgb_model, X_train, y_train, X_val, y_val)
        >>> print(f"Train-Val Gap: {gap_info['gap']:.4f} ({gap_info['gap_pct']:.1f}%)")
        >>> print(f"Diagnosis: {gap_info['diagnosis']}")
    """
    # Get predictions
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]

    # Calculate scores based on metric
    if metric == 'pr_auc':
        train_score = average_precision_score(y_train, y_train_proba)
        val_score = average_precision_score(y_val, y_val_proba)
    elif metric == 'roc_auc':
        train_score = roc_auc_score(y_train, y_train_proba)
        val_score = roc_auc_score(y_val, y_val_proba)
    elif metric == 'f1':
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        train_score = f1_score(y_train, y_train_pred)
        val_score = f1_score(y_val, y_val_pred)
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'pr_auc', 'roc_auc', or 'f1'")

    # Calculate gap
    gap = train_score - val_score
    gap_pct = (gap / train_score) * 100 if train_score > 0 else 0

    # Determine diagnosis
    if gap_pct > 15:
        diagnosis = "HIGH VARIANCE (Severe Overfitting)"
    elif gap_pct > 10:
        diagnosis = "HIGH VARIANCE (Moderate Overfitting)"
    elif train_score < 0.3 and val_score < 0.3:
        diagnosis = "HIGH BIAS (Underfitting)"
    else:
        diagnosis = "Good fit"

    return {
        'train_score': train_score,
        'val_score': val_score,
        'gap': gap,
        'gap_pct': gap_pct,
        'diagnosis': diagnosis
    }


def analyze_train_val_gaps(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    cv_folds: int = 4,
    random_seed: int = 1,
    metric: str = 'pr_auc',
    figsize: Tuple[int, int] = (14, 5),
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare train-validation gaps across multiple models using cross-validation.

    Uses stratified k-fold cross-validation to get robust gap estimates,
    matching the methodology used in GridSearchCV for hyperparameter tuning.

    Args:
        models: Dictionary mapping model names to model objects (pipelines or classifiers).
            Note: Models should be unfitted - they will be retrained during CV.
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (combined with X_train for CV)
        y_val: Validation labels (combined with y_train for CV)
        cv_folds: Number of CV folds (default: 4 to match GridSearchCV)
        random_seed: Random seed for CV strategy
        metric: Metric to use for evaluation ('pr_auc', 'roc_auc', 'f1')
        figsize: Figure size for gap visualization
        verbose: If True, print analysis and create plots

    Returns:
        DataFrame with gap analysis for each model

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import xgboost as xgb
        >>> models = {
        ...     'Random Forest': RandomForestClassifier(n_estimators=100),
        ...     'XGBoost': xgb.XGBClassifier(n_estimators=100)
        ... }
        >>> gap_df = analyze_train_val_gaps(models, X_train, y_train, X_val, y_val)
    """
    # Combine train and val for CV
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    # CV strategy
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    results = []
    fold_details = []

    for model_name, model_template in models.items():
        if verbose:
            print(f"\nAnalyzing {model_name} with {cv_folds}-fold CV...")

        train_scores = []
        val_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_combined, y_combined)):
            if verbose:
                print(f"  Processing fold {fold_idx + 1}/{cv_folds}...", end=" ", flush=True)

            X_fold_train = X_combined.iloc[train_idx]
            y_fold_train = y_combined.iloc[train_idx]
            X_fold_val = X_combined.iloc[val_idx]
            y_fold_val = y_combined.iloc[val_idx]

            # Clone model for this fold
            try:
                from sklearn.base import clone
                model = clone(model_template)
            except Exception:
                # If clone fails, use the template directly (will be fitted multiple times)
                model = model_template

            # Fit and evaluate
            model.fit(X_fold_train, y_fold_train)

            # Calculate scores
            gap_info = calculate_train_val_gap(
                model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, metric
            )

            train_scores.append(gap_info['train_score'])
            val_scores.append(gap_info['val_score'])

            fold_details.append({
                'model': model_name,
                'fold': fold_idx + 1,
                'train_score': gap_info['train_score'],
                'val_score': gap_info['val_score'],
                'gap': gap_info['gap']
            })

            if verbose:
                print("done")

        # Calculate averages
        train_mean = np.mean(train_scores)
        train_std = np.std(train_scores)
        val_mean = np.mean(val_scores)
        val_std = np.std(val_scores)
        gap = train_mean - val_mean
        gap_pct = (gap / train_mean) * 100 if train_mean > 0 else 0

        # Determine diagnosis
        if gap_pct > 15:
            diagnosis = "HIGH VARIANCE (Severe Overfitting)"
        elif gap_pct > 10:
            diagnosis = "HIGH VARIANCE (Moderate Overfitting)"
        elif train_mean < 0.3 and val_mean < 0.3:
            diagnosis = "HIGH BIAS (Underfitting)"
        else:
            diagnosis = "Good fit"

        results.append({
            'model': model_name,
            'train_score': train_mean,
            'train_std': train_std,
            'val_score': val_mean,
            'val_std': val_std,
            'gap': gap,
            'gap_pct': gap_pct,
            'diagnosis': diagnosis
        })

    gap_df = pd.DataFrame(results)

    if verbose:
        _print_gap_analysis(gap_df, metric, cv_folds)
        _plot_gap_analysis(gap_df, figsize)

    return gap_df


def _print_gap_analysis(gap_df: pd.DataFrame, metric: str, cv_folds: int) -> None:
    """Print formatted gap analysis results."""
    print("\n" + "-" * 80)
    print(f"Train vs Validation Performance ({metric.upper()}) - {cv_folds}-FOLD CV AVERAGE")
    print("-" * 80)

    for _, row in gap_df.iterrows():
        print(f"\n{row['model']}:")
        print(f"  Train {metric.upper()}:      {row['train_score']:.4f} +/- {row['train_std']:.4f}")
        print(f"  Validation {metric.upper()}: {row['val_score']:.4f} +/- {row['val_std']:.4f}")
        print(f"  Gap:               {row['gap']:.4f} ({row['gap_pct']:.1f}%)")
        print(f"  Diagnosis:         {row['diagnosis']}")


def _plot_gap_analysis(gap_df: pd.DataFrame, figsize: Tuple[int, int]) -> None:
    """Create gap analysis visualization."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Train vs Val
    x = np.arange(len(gap_df))
    width = 0.35
    axes[0].bar(x - width/2, gap_df['train_score'], width, label='Train', alpha=0.8)
    axes[0].bar(x + width/2, gap_df['val_score'], width, label='Validation', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Train vs Validation Score')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gap_df['model'], rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Gap
    colors = ['red' if 'VARIANCE' in d else 'orange' if 'BIAS' in d else 'green'
              for d in gap_df['diagnosis']]
    axes[1].bar(gap_df['model'], gap_df['gap'], color=colors, alpha=0.7)
    axes[1].axhline(y=0.10, color='orange', linestyle='--', label='Moderate Overfit (10%)', alpha=0.5)
    axes[1].axhline(y=0.15, color='red', linestyle='--', label='Severe Overfit (15%)', alpha=0.5)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Train-Val Gap')
    axes[1].set_title('Train-Validation Gap')
    axes[1].set_xticklabels(gap_df['model'], rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def track_xgboost_iterations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    xgb_params: Dict,
    preprocessor: Any = None,
    max_iterations: int = 200,
    cv_folds: int = 4,
    random_seed: int = 1,
    figsize: Tuple[int, int] = (12, 6),
    verbose: bool = True
) -> Tuple[int, pd.DataFrame]:
    """
    Track XGBoost performance per iteration using cross-validation.

    Trains XGBoost with extended iterations to visualize when overfitting begins,
    using cross-validation for robust estimates with confidence bands.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (combined with X_train for CV)
        y_val: Validation labels (combined with y_train for CV)
        xgb_params: XGBoost parameters dictionary (n_estimators will be overridden)
        preprocessor: Optional preprocessor (e.g., ColumnTransformer) to apply before XGBoost
        max_iterations: Maximum number of boosting iterations to track
        cv_folds: Number of CV folds
        random_seed: Random seed
        figsize: Figure size for iteration plot
        verbose: If True, print analysis and create plot

    Returns:
        Tuple of (optimal_iteration, iteration_tracking_df)

    Example:
        >>> xgb_params = {'max_depth': 4, 'learning_rate': 0.1, 'scale_pos_weight': 44}
        >>> optimal_iter, tracking_df = track_xgboost_iterations(
        ...     X_train, y_train, X_val, y_val,
        ...     xgb_params=xgb_params,
        ...     max_iterations=200
        ... )
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required for track_xgboost_iterations")

    # Combine train and val for CV
    X_combined = pd.concat([X_train, X_val])
    y_combined = pd.concat([y_train, y_val])

    # CV strategy
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    # Prepare model parameters
    model_params = xgb_params.copy()
    model_params['n_estimators'] = max_iterations
    model_params['random_state'] = random_seed
    model_params['eval_metric'] = 'aucpr'
    if 'n_jobs' not in model_params:
        model_params['n_jobs'] = -1

    if verbose:
        print(f"\nTracking XGBoost iterations with {cv_folds}-fold CV...")
        print(f"Training up to {max_iterations} iterations per fold...")

    # Store iteration scores from all folds
    all_fold_train_scores = []
    all_fold_val_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_combined, y_combined)):
        if verbose:
            print(f"  Processing fold {fold_idx + 1}/{cv_folds}...", end=" ", flush=True)

        X_fold_train = X_combined.iloc[train_idx]
        y_fold_train = y_combined.iloc[train_idx]
        X_fold_val = X_combined.iloc[val_idx]
        y_fold_val = y_combined.iloc[val_idx]

        # Apply preprocessor if provided
        if preprocessor is not None:
            from sklearn.base import clone
            prep = clone(preprocessor)
            X_fold_train_processed = prep.fit_transform(X_fold_train)
            X_fold_val_processed = prep.transform(X_fold_val)
        else:
            X_fold_train_processed = X_fold_train
            X_fold_val_processed = X_fold_val

        # Create and train model with eval_set for iteration tracking
        xgb_model = xgb.XGBClassifier(**model_params)
        xgb_model.fit(
            X_fold_train_processed, y_fold_train,
            eval_set=[(X_fold_train_processed, y_fold_train),
                      (X_fold_val_processed, y_fold_val)],
            verbose=False
        )

        # Extract iteration-by-iteration scores
        results = xgb_model.evals_result()
        fold_train_scores = results['validation_0']['aucpr']
        fold_val_scores = results['validation_1']['aucpr']

        all_fold_train_scores.append(fold_train_scores)
        all_fold_val_scores.append(fold_val_scores)

        if verbose:
            print("done")

    # Average scores across folds
    train_scores = np.mean(all_fold_train_scores, axis=0)
    val_scores = np.mean(all_fold_val_scores, axis=0)
    train_std = np.std(all_fold_train_scores, axis=0)
    val_std = np.std(all_fold_val_scores, axis=0)
    iterations = np.arange(1, len(train_scores) + 1)

    # Determine optimal iteration (from original params or best validation)
    if 'n_estimators' in xgb_params and xgb_params['n_estimators'] <= max_iterations:
        optimal_iter = xgb_params['n_estimators']
        if verbose:
            print(f"\nUsing specified n_estimators: {optimal_iter}")
    else:
        optimal_iter = int(np.argmax(val_scores) + 1)
        if verbose:
            print(f"\nBest validation score at iteration: {optimal_iter}")

    # Create tracking DataFrame
    tracking_df = pd.DataFrame({
        'iteration': iterations,
        'train_score_mean': train_scores,
        'train_score_std': train_std,
        'val_score_mean': val_scores,
        'val_score_std': val_std,
        'gap': train_scores - val_scores
    })

    if verbose:
        optimal_val = val_scores[optimal_iter - 1]
        optimal_train = train_scores[optimal_iter - 1]
        print(f"\nAt iteration {optimal_iter} ({cv_folds}-fold average):")
        print(f"  Training PR-AUC:   {optimal_train:.4f} +/- {train_std[optimal_iter - 1]:.4f}")
        print(f"  Validation PR-AUC: {optimal_val:.4f} +/- {val_std[optimal_iter - 1]:.4f}")
        print(f"  Gap:               {optimal_train - optimal_val:.4f}")

        _plot_iteration_tracking(iterations, train_scores, val_scores,
                                 train_std, val_std, optimal_iter, figsize)

    return optimal_iter, tracking_df


def _plot_iteration_tracking(
    iterations: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    train_std: np.ndarray,
    val_std: np.ndarray,
    optimal_iter: int,
    figsize: Tuple[int, int]
) -> None:
    """Create iteration tracking plot with confidence bands."""
    fig, ax = plt.subplots(figsize=figsize)

    # Plot mean lines
    ax.plot(iterations, train_scores, label='Training PR-AUC (mean)', linewidth=2, color='#1f77b4')
    ax.plot(iterations, val_scores, label='Validation PR-AUC (mean)', linewidth=2, color='#ff7f0e')

    # Add +/-1 std confidence bands
    ax.fill_between(iterations, train_scores - train_std, train_scores + train_std,
                    alpha=0.2, color='#1f77b4', label='Training +/-1 std')
    ax.fill_between(iterations, val_scores - val_std, val_scores + val_std,
                    alpha=0.2, color='#ff7f0e', label='Validation +/-1 std')

    # Mark optimal iteration
    ax.axvline(optimal_iter, color='red', linestyle='--', alpha=0.5,
               label=f'Selected ({optimal_iter})')
    ax.scatter([optimal_iter], [val_scores[optimal_iter - 1]], color='red', s=100, zorder=5)

    ax.set_xlabel('Boosting Iteration')
    ax.set_ylabel('PR-AUC')
    ax.set_title('XGBoost: Performance by Iteration (CV Average)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    # Add diagnosis
    final_val = val_scores[-1]
    optimal_val = val_scores[optimal_iter - 1]
    if final_val < optimal_val - 0.01:
        diagnosis = f"Overfitting after iteration {optimal_iter}"
    else:
        diagnosis = "No severe overfitting detected"

    ax.text(0.02, 0.98, diagnosis, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def analyze_cv_fold_variance(
    cv_results_paths: Dict[str, str],
    figsize: Tuple[int, int] = (14, 5),
    verbose: bool = True
) -> pd.DataFrame:
    """
    Analyze variance across CV folds from saved CV results.

    Computes coefficient of variation (CV) to assess model stability:
    - CV < 3%: Low variance (stable)
    - CV 3-5%: Moderate variance
    - CV > 5%: High variance (unstable)

    Args:
        cv_results_paths: Dictionary mapping model names to CV results CSV paths.
            Supports glob patterns (e.g., 'models/logs/rf_cv_results_*.csv')
        figsize: Figure size for variance plot
        verbose: If True, print analysis and create plots

    Returns:
        DataFrame with variance analysis for each model

    Example:
        >>> variance_df = analyze_cv_fold_variance({
        ...     'Random Forest': 'models/logs/random_forest_cv_results_*.csv',
        ...     'XGBoost': 'models/logs/xgboost_cv_results_*.csv'
        ... })
    """
    results = []

    for model_name, path_pattern in cv_results_paths.items():
        # Handle glob patterns
        path = Path(path_pattern)
        if '*' in str(path):
            matching_files = sorted(path.parent.glob(path.name))
            if not matching_files:
                if verbose:
                    print(f"Warning: No files found matching {path_pattern}")
                continue
            cv_path = matching_files[-1]  # Use most recent
        else:
            cv_path = path

        if not cv_path.exists():
            if verbose:
                print(f"Warning: {cv_path} not found")
            continue

        # Load and analyze
        cv_results = pd.read_csv(cv_path)
        best_idx = cv_results['rank_test_score'].idxmin()
        best_row = cv_results.loc[best_idx]

        mean_score = best_row['mean_test_score']
        std_score = best_row['std_test_score']
        cv_coef = (std_score / mean_score) * 100 if mean_score > 0 else 0

        results.append({
            'model': model_name,
            'mean_score': mean_score,
            'std_score': std_score,
            'cv_coef_pct': cv_coef
        })

        if verbose:
            print(f"\n{model_name} (Best Config):")
            print(f"  Mean Score: {mean_score:.4f} +/- {std_score:.4f}")
            print(f"  CV Coefficient: {cv_coef:.2f}%")

    variance_df = pd.DataFrame(results)

    if verbose and len(variance_df) > 0:
        print("\n" + "-" * 60)
        print("Stability Diagnosis:")
        for _, row in variance_df.iterrows():
            if row['cv_coef_pct'] > 5:
                diag = "High variance (unstable)"
            elif row['cv_coef_pct'] > 3:
                diag = "Moderate variance"
            else:
                diag = "Low variance (stable)"
            print(f"  {row['model']}: {diag}")

        _plot_cv_variance(variance_df, figsize)

    return variance_df


def _plot_cv_variance(variance_df: pd.DataFrame, figsize: Tuple[int, int]) -> None:
    """Create CV variance visualization."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Mean score with error bars
    axes[0].bar(variance_df['model'], variance_df['mean_score'],
                yerr=variance_df['std_score'], capsize=10, alpha=0.7)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Mean Score Across CV Folds')
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: CV coefficient
    colors = ['red' if cv > 5 else 'orange' if cv > 3 else 'green'
              for cv in variance_df['cv_coef_pct']]
    axes[1].bar(variance_df['model'], variance_df['cv_coef_pct'], color=colors, alpha=0.7)
    axes[1].axhline(5, color='red', linestyle='--', alpha=0.5, label='High (5%)')
    axes[1].axhline(3, color='orange', linestyle='--', alpha=0.5, label='Moderate (3%)')
    axes[1].set_ylabel('Coefficient of Variation (%)')
    axes[1].set_title('Model Stability (Lower is Better)')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()


def generate_bias_variance_report(
    gap_df: pd.DataFrame,
    optimal_xgb_iter: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Generate diagnostic report with recommendations.

    Args:
        gap_df: DataFrame from analyze_train_val_gaps()
        optimal_xgb_iter: Optimal XGBoost iteration from track_xgboost_iterations()
        output_path: Optional path to save report as text file
        verbose: If True, print report to console

    Returns:
        Report text as string

    Example:
        >>> report = generate_bias_variance_report(gap_df, optimal_xgb_iter=75)
        >>> print(report)
    """
    report = []
    report.append("=" * 80)
    report.append("BIAS-VARIANCE DIAGNOSTIC REPORT")
    report.append("=" * 80)

    for _, row in gap_df.iterrows():
        model = row['model']
        report.append(f"\n{model}")
        report.append("-" * len(model))
        report.append(f"Training Score:     {row['train_score']:.4f}")
        report.append(f"Validation Score:   {row['val_score']:.4f}")
        report.append(f"Train-Val Gap:      {row['gap']:.4f} ({row['gap_pct']:.1f}%)")
        report.append(f"Diagnosis:          {row['diagnosis']}")

        report.append("\nRecommendations:")
        if 'VARIANCE' in row['diagnosis']:
            report.append("  - Model is overfitting")
            report.append("  - Consider: stronger regularization, early stopping, more data")
        elif 'BIAS' in row['diagnosis']:
            report.append("  - Model is underfitting")
            report.append("  - Consider: more complex model, better features")
        else:
            report.append("  - Good bias-variance tradeoff")

    if optimal_xgb_iter:
        report.append(f"\nXGBoost Iteration Analysis:")
        report.append(f"  - Optimal n_estimators: {optimal_xgb_iter}")
        report.append(f"  - Based on CV validation performance")

    report.append("\n" + "=" * 80)
    best_model = gap_df.loc[gap_df['val_score'].idxmax(), 'model']
    report.append(f"\nOVERALL: Best model is {best_model}")
    report.append("=" * 80)

    report_text = "\n".join(report)

    if verbose:
        print("\n" + report_text)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        if verbose:
            print(f"\nReport saved to: {output_path}")

    return report_text
