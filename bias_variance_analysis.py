"""
Bias-Variance Analysis for Fraud Detection Models

This script performs comprehensive bias-variance diagnostics using:
1. Existing CV results from hyperparameter tuning
2. Simple retrained models with train/val tracking
3. XGBoost iteration tracking
4. Diagnostic visualizations and recommendations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

# XGBoost
import xgboost as xgb

# Shared modules from src/
from src.config import DataConfig, FeatureListsConfig, ModelConfig, TrainingConfig
from src.data import load_and_split_data
from src.preprocessing import FraudFeatureTransformer, PreprocessingPipelineFactory

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# Configuration
# ============================================================================

# Use shared configuration (default random seed = 1, configurable)
RANDOM_SEED = DataConfig.DEFAULT_RANDOM_SEED
TARGET_COL = DataConfig.TARGET_COLUMN
OUTPUT_DIR = Path('analysis/bias_variance')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load feature lists from shared config
feature_config = FeatureListsConfig.load()
CONTINUOUS_FEATURES = feature_config['continuous_numeric'] + feature_config['binary']
CATEGORICAL_FEATURES = feature_config['categorical']
ALL_FEATURES = feature_config['all_features']

print(f"Loaded {len(ALL_FEATURES)} features from config")
print(f"  - Continuous + Binary: {len(CONTINUOUS_FEATURES)}")
print(f"  - Categorical: {len(CATEGORICAL_FEATURES)}")

# ============================================================================
# Data Loading with Feature Engineering
# ============================================================================

def load_and_prepare_data():
    """Load and prepare data with feature engineering using production transformer.

    Uses shared modules for consistency:
    - load_and_split_data() from src.data
    - FraudFeatureTransformer from src.preprocessing
    """
    # Load and split raw data using shared function
    train_raw, val_raw, test_raw = load_and_split_data(random_seed=RANDOM_SEED, verbose=True)

    # Load or create feature transformer
    transformer_path = Path('models/transformer_config.json')
    if transformer_path.exists():
        print(f"\n✓ Loading transformer config from {transformer_path}")
        transformer = FraudFeatureTransformer.load(str(transformer_path))
    else:
        print(f"\n⚠️  Transformer config not found at {transformer_path}")
        print("  Creating new transformer and fitting on training data...")
        transformer = FraudFeatureTransformer()
        transformer.fit(train_raw)

        # Save for future use
        transformer.save(str(transformer_path))
        print(f"  ✓ Saved transformer config to {transformer_path}")

    # Apply feature engineering to all splits
    print("\nApplying feature engineering...")
    train_df = transformer.transform(train_raw)
    val_df = transformer.transform(val_raw)
    test_df = transformer.transform(test_raw)

    # Extract features and targets
    X_train = train_df[ALL_FEATURES]
    y_train = train_raw[TARGET_COL]
    X_val = val_df[ALL_FEATURES]
    y_val = val_raw[TARGET_COL]
    X_test = test_df[ALL_FEATURES]
    y_test = test_raw[TARGET_COL]

    print(f"  ✓ Engineered features: {len(ALL_FEATURES)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_preprocessors():
    """Create preprocessing pipelines using shared factory."""
    # Get feature lists from config (already loaded at module level)
    continuous_numeric = feature_config['continuous_numeric']
    binary = feature_config['binary']
    categorical = feature_config['categorical']

    # Create pipelines using shared factory with explicit feature lists
    logistic_preprocessor = PreprocessingPipelineFactory.create_logistic_pipeline(
        categorical_features=categorical,
        continuous_numeric=continuous_numeric,
        binary=binary
    )
    tree_preprocessor = PreprocessingPipelineFactory.create_tree_pipeline(
        categorical_features=categorical,
        continuous_numeric=continuous_numeric,
        binary=binary
    )

    return logistic_preprocessor, tree_preprocessor


# ============================================================================
# Helper: Load Best Parameters from CV Results
# ============================================================================

def load_best_params_from_cv():
    """
    Load best hyperparameters using shared ModelConfig.
    Returns dict with 'rf' and 'xgb' keys containing best params, or None if not found.
    """
    best_params = {'rf': None, 'xgb': None}

    # Load Random Forest params from CV results
    try:
        rf_params = ModelConfig.load_hyperparameters(
            model_type='random_forest',
            source='cv_results',
            random_seed=RANDOM_SEED
        )
        best_params['rf'] = rf_params
    except Exception:
        # Fallback handled by ModelConfig
        pass

    # Load XGBoost params from CV results
    try:
        xgb_params = ModelConfig.load_hyperparameters(
            model_type='xgboost',
            source='cv_results',
            random_seed=RANDOM_SEED
        )
        best_params['xgb'] = xgb_params
    except Exception:
        # Fallback handled by ModelConfig
        pass

    return best_params


# ============================================================================
# Helper: Perform Cross-Validation for a Model
# ============================================================================

def perform_cv_evaluation(pipeline, cv_strategy, X_train_val, y_train_val, model_name):
    """Perform k-fold cross-validation and return train/val scores and fold details.

    Args:
        pipeline: Sklearn Pipeline with preprocessor and classifier
        cv_strategy: Cross-validation splitter (e.g., StratifiedKFold)
        X_train_val: Combined training and validation features
        y_train_val: Combined training and validation targets
        model_name: Name of the model (for reporting)

    Returns:
        Tuple of (train_scores, val_scores, fold_details)
    """
    train_scores = []
    val_scores = []
    fold_details = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train_val, y_train_val)):
        X_fold_train = X_train_val.iloc[train_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_val = y_train_val.iloc[val_idx]

        pipeline.fit(X_fold_train, y_fold_train)
        train_score = average_precision_score(y_fold_train, pipeline.predict_proba(X_fold_train)[:, 1])
        val_score = average_precision_score(y_fold_val, pipeline.predict_proba(X_fold_val)[:, 1])
        train_scores.append(train_score)
        val_scores.append(val_score)

        fold_details.append({
            'model': model_name,
            'fold': fold_idx + 1,
            'train_pr_auc': train_score,
            'val_pr_auc': val_score,
            'gap': train_score - val_score
        })

    return train_scores, val_scores, fold_details


# ============================================================================
# 1. Train-Validation Gap Analysis
# ============================================================================

def train_validation_gap_analysis(X_train, y_train, X_val, y_val, use_tuned_params=True):
    """
    Compare training vs validation performance using cross-validation.
    Uses 4-fold stratified CV to match GridSearchCV methodology.
    If use_tuned_params=True, loads best params from CV results.
    Otherwise uses baseline parameters.
    """
    print("\n" + "="*80)
    print("1. TRAIN-VALIDATION GAP ANALYSIS (4-FOLD CV)")
    print("="*80)
    print("NOTE: Using cross-validation (not single split) for robust gap estimates")
    print("      matching GridSearchCV methodology")

    logistic_preprocessor, tree_preprocessor = create_preprocessors()

    # Combine train and val for CV (to match GridSearchCV data)
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    scale_pos_weight = (y_train_val == 0).sum() / (y_train_val == 1).sum()

    # Use same CV strategy as GridSearchCV (from shared config)
    cv_strategy = TrainingConfig.get_cv_strategy(random_seed=RANDOM_SEED)

    # Load tuned parameters if requested
    tuned_params = None
    if use_tuned_params:
        print("\nAttempting to load tuned hyperparameters from CV results...")
        tuned_params = load_best_params_from_cv()
    else:
        print("\nUsing baseline hyperparameters (no tuning)")

    results = []
    fold_details = []

    # Logistic Regression (no tuning - always baseline)
    print("\nTraining Logistic Regression with 4-fold CV...")
    lr = Pipeline([
        ('preprocessor', logistic_preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED))
    ])

    # Perform CV to get train and val scores for each fold
    lr_train_scores, lr_val_scores, lr_fold_details = perform_cv_evaluation(
        lr, cv_strategy, X_train_val, y_train_val, 'Logistic Regression'
    )
    fold_details.extend(lr_fold_details)

    lr_train_pr = np.mean(lr_train_scores)
    lr_val_pr = np.mean(lr_val_scores)
    lr_train_std = np.std(lr_train_scores)
    lr_val_std = np.std(lr_val_scores)

    results.append({'model': 'Logistic Regression', 'dataset': 'Train', 'pr_auc': lr_train_pr, 'std': lr_train_std})
    results.append({'model': 'Logistic Regression', 'dataset': 'Validation', 'pr_auc': lr_val_pr, 'std': lr_val_std})
    print(f"  Train PR-AUC: {lr_train_pr:.4f} ± {lr_train_std:.4f}")
    print(f"  Val PR-AUC:   {lr_val_pr:.4f} ± {lr_val_std:.4f}")

    # Random Forest - use tuned params if available
    print("\nTraining Random Forest with 4-fold CV...")
    if tuned_params and tuned_params['rf']:
        rf_params = tuned_params['rf']
        print(f"  Using tuned params: n_estimators={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}, "
              f"min_samples_split={rf_params['min_samples_split']}, min_samples_leaf={rf_params['min_samples_leaf']}")
        rf = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=rf_params['n_estimators'],
                max_depth=rf_params['max_depth'],
                min_samples_split=rf_params['min_samples_split'],
                min_samples_leaf=rf_params['min_samples_leaf'],
                max_features=rf_params['max_features'],
                class_weight=rf_params['class_weight'],
                random_state=RANDOM_SEED,
                n_jobs=-1
            ))
        ])
    else:
        print("  Using baseline params: n_estimators=100, class_weight='balanced'")
        rf = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1
            ))
        ])

    # Perform CV
    rf_train_scores, rf_val_scores, rf_fold_details = perform_cv_evaluation(
        rf, cv_strategy, X_train_val, y_train_val, 'Random Forest'
    )
    fold_details.extend(rf_fold_details)

    rf_train_pr = np.mean(rf_train_scores)
    rf_val_pr = np.mean(rf_val_scores)
    rf_train_std = np.std(rf_train_scores)
    rf_val_std = np.std(rf_val_scores)

    results.append({'model': 'Random Forest', 'dataset': 'Train', 'pr_auc': rf_train_pr, 'std': rf_train_std})
    results.append({'model': 'Random Forest', 'dataset': 'Validation', 'pr_auc': rf_val_pr, 'std': rf_val_std})
    print(f"  Train PR-AUC: {rf_train_pr:.4f} ± {rf_train_std:.4f}")
    print(f"  Val PR-AUC:   {rf_val_pr:.4f} ± {rf_val_std:.4f}")

    # XGBoost - use tuned params if available
    print("\nTraining XGBoost with 4-fold CV...")
    if tuned_params and tuned_params['xgb']:
        xgb_params = tuned_params['xgb']
        print(f"  Using tuned params: n_estimators={xgb_params['n_estimators']}, max_depth={xgb_params['max_depth']}, "
              f"learning_rate={xgb_params['learning_rate']}, min_child_weight={xgb_params['min_child_weight']}, "
              f"gamma={xgb_params['gamma']}")
        if 'reg_alpha' in xgb_params or 'reg_lambda' in xgb_params:
            print(f"  + L1/L2 regularization: reg_alpha={xgb_params.get('reg_alpha', 0)}, reg_lambda={xgb_params.get('reg_lambda', 1)}")

        xgb_model = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=xgb_params['n_estimators'],
                max_depth=xgb_params['max_depth'],
                learning_rate=xgb_params['learning_rate'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                min_child_weight=xgb_params['min_child_weight'],
                gamma=xgb_params['gamma'],
                reg_alpha=xgb_params.get('reg_alpha', 0.0),
                reg_lambda=xgb_params.get('reg_lambda', 1.0),
                scale_pos_weight=xgb_params['scale_pos_weight'],
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='aucpr'
            ))
        ])
    else:
        print("  Using baseline params: n_estimators=100, scale_pos_weight=auto")
        xgb_model = Pipeline([
            ('preprocessor', tree_preprocessor),
            ('classifier', xgb.XGBClassifier(
                n_estimators=100, scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED, n_jobs=-1, eval_metric='aucpr'
            ))
        ])

    # Perform CV
    xgb_train_scores, xgb_val_scores, xgb_fold_details = perform_cv_evaluation(
        xgb_model, cv_strategy, X_train_val, y_train_val, 'XGBoost'
    )
    fold_details.extend(xgb_fold_details)

    xgb_train_pr = np.mean(xgb_train_scores)
    xgb_val_pr = np.mean(xgb_val_scores)
    xgb_train_std = np.std(xgb_train_scores)
    xgb_val_std = np.std(xgb_val_scores)

    results.append({'model': 'XGBoost', 'dataset': 'Train', 'pr_auc': xgb_train_pr, 'std': xgb_train_std})
    results.append({'model': 'XGBoost', 'dataset': 'Validation', 'pr_auc': xgb_val_pr, 'std': xgb_val_std})
    print(f"  Train PR-AUC: {xgb_train_pr:.4f} ± {xgb_train_std:.4f}")
    print(f"  Val PR-AUC:   {xgb_val_pr:.4f} ± {xgb_val_std:.4f}")

    # Analyze gaps
    df_results = pd.DataFrame(results)
    df_fold_details = pd.DataFrame(fold_details)

    print("\n" + "-"*80)
    print("Train vs Validation Performance (PR-AUC) - 4-FOLD CV AVERAGE")
    print("-"*80)
    print("NOTE: These gaps are averaged across 4 folds, matching GridSearchCV methodology")

    gap_summary = []
    for model_name in ['Logistic Regression', 'Random Forest', 'XGBoost']:
        train_row = df_results[(df_results['model'] == model_name) & (df_results['dataset'] == 'Train')].iloc[0]
        val_row = df_results[(df_results['model'] == model_name) & (df_results['dataset'] == 'Validation')].iloc[0]

        train_score = train_row['pr_auc']
        val_score = val_row['pr_auc']
        train_std = train_row['std']
        val_std = val_row['std']

        gap = train_score - val_score
        gap_pct = (gap / train_score) * 100

        print(f"\n{model_name}:")
        print(f"  Train PR-AUC:      {train_score:.4f} ± {train_std:.4f}")
        print(f"  Validation PR-AUC: {val_score:.4f} ± {val_std:.4f}")
        print(f"  Gap:               {gap:.4f} ({gap_pct:.1f}%)")

        # Calculate gap across folds for more robust diagnosis
        fold_gaps = df_fold_details[df_fold_details['model'] == model_name]['gap'].values
        avg_fold_gap = np.mean(fold_gaps)

        if avg_fold_gap > 0.15:
            diagnosis = "⚠️  HIGH VARIANCE (Severe Overfitting)"
        elif avg_fold_gap > 0.10:
            diagnosis = "⚠️  HIGH VARIANCE (Moderate Overfitting)"
        elif train_score < 0.3 and val_score < 0.3:
            diagnosis = "⚠️  HIGH BIAS (Underfitting)"
        else:
            diagnosis = "✓ Good fit"
        print(f"  Diagnosis:         {diagnosis}")

        gap_summary.append({
            'model': model_name,
            'train_pr_auc': train_score,
            'train_std': train_std,
            'val_pr_auc': val_score,
            'val_std': val_std,
            'gap': gap,
            'gap_pct': gap_pct,
            'diagnosis': diagnosis
        })

    df_gap = pd.DataFrame(gap_summary)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Train vs Val
    x = np.arange(len(df_gap))
    width = 0.35
    axes[0].bar(x - width/2, df_gap['train_pr_auc'], width, label='Train', alpha=0.8)
    axes[0].bar(x + width/2, df_gap['val_pr_auc'], width, label='Validation', alpha=0.8)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('PR-AUC')
    axes[0].set_title('Train vs Validation PR-AUC')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_gap['model'], rotation=15, ha='right')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Plot 2: Gap
    colors = ['red' if 'VARIANCE' in d else 'orange' if 'BIAS' in d else 'green' for d in df_gap['diagnosis']]
    axes[1].bar(df_gap['model'], df_gap['gap'], color=colors, alpha=0.7)
    axes[1].axhline(y=0.10, color='orange', linestyle='--', label='Moderate Overfit Threshold', alpha=0.5)
    axes[1].axhline(y=0.15, color='red', linestyle='--', label='Severe Overfit Threshold', alpha=0.5)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('Train-Val Gap (PR-AUC)')
    axes[1].set_title('Train-Validation Gap')
    axes[1].set_xticklabels(df_gap['model'], rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_train_val_gap.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / '01_train_val_gap.png'}")
    plt.close()

    df_results.to_csv(OUTPUT_DIR / '01_train_val_metrics.csv', index=False)
    df_gap.to_csv(OUTPUT_DIR / '01_gap_summary.csv', index=False)
    df_fold_details.to_csv(OUTPUT_DIR / '01_fold_details.csv', index=False)
    print(f"\n✓ Saved fold-by-fold details: {OUTPUT_DIR / '01_fold_details.csv'}")

    return df_gap


# ============================================================================
# 2. XGBoost Iteration Tracking
# ============================================================================

def xgboost_iteration_tracking(X_train, y_train, X_val, y_val, use_tuned_params=True):
    """Track XGBoost performance per iteration using 4-fold CV average.

    This function uses the same CV methodology as GridSearchCV to provide
    robust iteration-by-iteration performance estimates across folds.
    """
    print("\n" + "="*80)
    print("2. XGBOOST PER-ITERATION TRACKING (4-FOLD CV AVERAGE)")
    print("="*80)

    _, tree_preprocessor = create_preprocessors()

    # Combine train and val for CV (matching GridSearchCV methodology)
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    scale_pos_weight = (y_train_val == 0).sum() / (y_train_val == 1).sum()

    # Load tuned parameters if available
    tuned_params = None
    if use_tuned_params:
        print("\nAttempting to load tuned hyperparameters from CV results...")
        tuned_params = load_best_params_from_cv()

    # Get CV strategy (same as GridSearchCV)
    cv_strategy = TrainingConfig.get_cv_strategy(random_seed=RANDOM_SEED)

    print(f"\nTraining XGBoost with {cv_strategy.n_splits}-fold CV iteration tracking...")
    print("This matches the GridSearchCV methodology used for hyperparameter tuning")

    # Store scores from all folds
    all_fold_train_scores = []
    all_fold_val_scores = []

    # Perform CV
    for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train_val, y_train_val)):
        print(f"  Processing fold {fold_idx + 1}/{cv_strategy.n_splits}...", end=" ", flush=True)

        X_fold_train = X_train_val.iloc[train_idx]
        y_fold_train = y_train_val.iloc[train_idx]
        X_fold_val = X_train_val.iloc[val_idx]
        y_fold_val = y_train_val.iloc[val_idx]

        # Preprocess this fold
        fold_preprocessor = PreprocessingPipelineFactory.create_tree_pipeline()
        X_fold_train_processed = fold_preprocessor.fit_transform(X_fold_train)
        X_fold_val_processed = fold_preprocessor.transform(X_fold_val)

        # Use tuned params if available, otherwise baseline
        if tuned_params and tuned_params['xgb']:
            xgb_params = tuned_params['xgb']
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,  # Use more iterations to see full curve
                max_depth=xgb_params['max_depth'],
                learning_rate=xgb_params['learning_rate'],
                subsample=xgb_params['subsample'],
                colsample_bytree=xgb_params['colsample_bytree'],
                min_child_weight=xgb_params['min_child_weight'],
                gamma=xgb_params['gamma'],
                reg_alpha=xgb_params.get('reg_alpha', 0.0),
                reg_lambda=xgb_params.get('reg_lambda', 1.0),
                scale_pos_weight=xgb_params['scale_pos_weight'],
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='aucpr'
            )
        else:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
                n_jobs=-1,
                eval_metric='aucpr'
            )

        # Train with eval_set to track iterations
        xgb_model.fit(
            X_fold_train_processed, y_fold_train,
            eval_set=[(X_fold_train_processed, y_fold_train), (X_fold_val_processed, y_fold_val)],
            verbose=False
        )

        # Extract iteration-by-iteration scores
        results = xgb_model.evals_result()
        all_fold_train_scores.append(results['validation_0']['aucpr'])
        all_fold_val_scores.append(results['validation_1']['aucpr'])

        print("✓")

    # Average scores across all folds
    train_scores = np.mean(all_fold_train_scores, axis=0)
    val_scores = np.mean(all_fold_val_scores, axis=0)
    train_std = np.std(all_fold_train_scores, axis=0)
    val_std = np.std(all_fold_val_scores, axis=0)
    iterations = range(1, len(train_scores) + 1)

    print(f"\n✓ Averaged scores across {cv_strategy.n_splits} folds")

    # Use CV-tuned n_estimators
    if tuned_params and tuned_params['xgb']:
        cv_tuned_iter = tuned_params['xgb']['n_estimators']
        print(f"\n✓ Using CV-tuned n_estimators: {cv_tuned_iter}")
    else:
        # Fallback to CV average peak if no tuned params available
        cv_tuned_iter = np.argmax(val_scores) + 1
        print(f"\n⚠️  No tuned params found, using CV average peak: {cv_tuned_iter}")

    # Get performance at CV-tuned iteration
    cv_tuned_val = val_scores[cv_tuned_iter - 1]
    cv_tuned_train = train_scores[cv_tuned_iter - 1]
    cv_tuned_val_std = val_std[cv_tuned_iter - 1]
    cv_tuned_train_std = train_std[cv_tuned_iter - 1]

    print(f"\nAt CV-tuned iteration ({cv_tuned_iter}) - {cv_strategy.n_splits}-FOLD AVERAGE:")
    print(f"  Training PR-AUC:   {cv_tuned_train:.4f} ± {cv_tuned_train_std:.4f}")
    print(f"  Validation PR-AUC: {cv_tuned_val:.4f} ± {cv_tuned_val_std:.4f}")
    print(f"  Gap:               {cv_tuned_train - cv_tuned_val:.4f}")

    # Report CV average peak for comparison
    cv_avg_peak_iter = np.argmax(val_scores) + 1
    cv_avg_peak_val = val_scores[cv_avg_peak_iter - 1]
    if cv_avg_peak_iter != cv_tuned_iter:
        print(f"\nNote: CV average peak at iteration {cv_avg_peak_iter} (PR-AUC: {cv_avg_peak_val:.4f})")
        print(f"      GridSearchCV selected {cv_tuned_iter}, which is appropriate")

    final_val = val_scores[-1]
    if final_val < cv_tuned_val - 0.01:
        print(f"\n⚠️  WARNING: Validation performance degraded after iteration {cv_tuned_iter}")
        print(f"  CV-tuned: {cv_tuned_val:.4f} | Final: {final_val:.4f}")
        print(f"  Recommendation: CV-tuned value ({cv_tuned_iter}) is appropriate")

    # Plot with confidence bands
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot mean curves
    ax.plot(iterations, train_scores, label='Training PR-AUC (mean)', linewidth=2, color='C0')
    ax.plot(iterations, val_scores, label='Validation PR-AUC (mean)', linewidth=2, color='C1')

    # Add confidence bands (±1 std dev)
    ax.fill_between(iterations, train_scores - train_std, train_scores + train_std,
                     alpha=0.2, color='C0', label='Training ±1 std')
    ax.fill_between(iterations, val_scores - val_std, val_scores + val_std,
                     alpha=0.2, color='C1', label='Validation ±1 std')

    # Mark CV-tuned iteration
    ax.axvline(cv_tuned_iter, color='red', linestyle='--', alpha=0.5, label=f'CV-Tuned ({cv_tuned_iter})')
    ax.scatter([cv_tuned_iter], [cv_tuned_val], color='red', s=100, zorder=5)

    ax.set_xlabel('Boosting Iteration')
    ax.set_ylabel('PR-AUC')
    ax.set_title(f'XGBoost: Performance by Iteration ({cv_strategy.n_splits}-Fold CV Average)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    diagnosis = f"⚠️ Overfitting after iteration {cv_tuned_iter}" if final_val < cv_tuned_val - 0.01 else "✓ No severe overfitting"
    ax.text(0.02, 0.98, diagnosis, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_xgboost_iterations.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / '02_xgboost_iterations.png'}")
    plt.close()

    # Save data with CV statistics
    pd.DataFrame({
        'iteration': iterations,
        'train_pr_auc_mean': train_scores,
        'train_pr_auc_std': train_std,
        'val_pr_auc_mean': val_scores,
        'val_pr_auc_std': val_std,
        'gap': train_scores - val_scores
    }).to_csv(OUTPUT_DIR / '02_iteration_tracking.csv', index=False)

    return cv_tuned_iter


# ============================================================================
# 3. CV Fold Variance Analysis
# ============================================================================

def analyze_cv_fold_variance():
    """Analyze variance across CV folds."""
    print("\n" + "="*80)
    print("3. CROSS-VALIDATION FOLD VARIANCE ANALYSIS")
    print("="*80)

    logs_dir = Path('models/logs')
    rf_files = sorted(logs_dir.glob('random_forest_cv_results_*.csv'))
    xgb_files = sorted(logs_dir.glob('xgboost_cv_results_*.csv'))

    if not rf_files or not xgb_files:
        print("⚠️  No CV results found")
        return

    rf_cv = pd.read_csv(rf_files[-1])
    xgb_cv = pd.read_csv(xgb_files[-1])

    results = []

    # Random Forest
    rf_best = rf_cv.nlargest(1, 'mean_test_score').iloc[0]
    rf_mean, rf_std = rf_best['mean_test_score'], rf_best['std_test_score']
    rf_cv_coef = (rf_std / rf_mean) * 100

    print(f"\nRandom Forest (Best Config):")
    print(f"  Mean PR-AUC: {rf_mean:.4f} ± {rf_std:.4f}")
    print(f"  CV Coefficient: {rf_cv_coef:.2f}%")

    results.append({'model': 'Random Forest', 'mean_pr_auc': rf_mean, 'std_pr_auc': rf_std, 'cv_coef_pct': rf_cv_coef})

    # XGBoost
    xgb_best = xgb_cv.nlargest(1, 'mean_test_score').iloc[0]
    xgb_mean, xgb_std = xgb_best['mean_test_score'], xgb_best['std_test_score']
    xgb_cv_coef = (xgb_std / xgb_mean) * 100

    print(f"\nXGBoost (Best Config):")
    print(f"  Mean PR-AUC: {xgb_mean:.4f} ± {xgb_std:.4f}")
    print(f"  CV Coefficient: {xgb_cv_coef:.2f}%")

    results.append({'model': 'XGBoost', 'mean_pr_auc': xgb_mean, 'std_pr_auc': xgb_std, 'cv_coef_pct': xgb_cv_coef})

    # Diagnosis
    print("\nStability Diagnosis:")
    for r in results:
        if r['cv_coef_pct'] > 5:
            diag = "⚠️  High variance (unstable)"
        elif r['cv_coef_pct'] > 3:
            diag = "⚠️  Moderate variance"
        else:
            diag = "✓ Low variance (stable)"
        print(f"  {r['model']}: {diag}")

    # Plot
    df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(df['model'], df['mean_pr_auc'], yerr=df['std_pr_auc'], capsize=10, alpha=0.7)
    axes[0].set_ylabel('PR-AUC')
    axes[0].set_title('Mean PR-AUC Across CV Folds')
    axes[0].grid(axis='y', alpha=0.3)

    colors = ['red' if cv > 5 else 'orange' if cv > 3 else 'green' for cv in df['cv_coef_pct']]
    axes[1].bar(df['model'], df['cv_coef_pct'], color=colors, alpha=0.7)
    axes[1].axhline(5, color='red', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Coefficient of Variation (%)')
    axes[1].set_title('Model Stability')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_cv_variance.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR / '03_cv_variance.png'}")
    plt.close()

    df.to_csv(OUTPUT_DIR / '03_cv_variance.csv', index=False)


# ============================================================================
# 4. Diagnostic Summary
# ============================================================================

def generate_summary(gap_df, best_iter):
    """Generate diagnostic report."""
    print("\n" + "="*80)
    print("4. DIAGNOSTIC SUMMARY")
    print("="*80)

    report = []
    report.append("="*80)
    report.append("BIAS-VARIANCE DIAGNOSTIC REPORT")
    report.append("E-Commerce Fraud Detection Models")
    report.append("="*80)

    for _, row in gap_df.iterrows():
        model = row['model']
        report.append(f"\n{model}")
        report.append("-" * len(model))
        report.append(f"Training PR-AUC:     {row['train_pr_auc']:.4f}")
        report.append(f"Validation PR-AUC:   {row['val_pr_auc']:.4f}")
        report.append(f"Train-Val Gap:       {row['gap']:.4f} ({row['gap_pct']:.1f}%)")
        report.append(f"Diagnosis:           {row['diagnosis']}")

        report.append("\nRecommendations:")
        if 'VARIANCE' in row['diagnosis']:
            report.append("  • Model is overfitting")
            report.append("  • Consider: stronger regularization, early stopping, more data")
        elif 'BIAS' in row['diagnosis']:
            report.append("  • Model is underfitting")
            report.append("  • Consider: more complex model, better features")
        else:
            report.append("  • Good bias-variance tradeoff")

    if best_iter:
        report.append(f"\nXGBoost Iteration Analysis:")
        report.append(f"  • Using CV-tuned n_estimators: {best_iter}")
        report.append(f"  • This value is based on 4-fold GridSearchCV (more robust than single split)")
        report.append(f"  • Plot shows performance at CV-tuned value, not single-split peak")
        report.append(f"  • Recommendation: CV-tuned value is appropriate and well-validated")

    report.append("\n" + "="*80)
    best_model = gap_df.loc[gap_df['val_pr_auc'].idxmax(), 'model']
    report.append(f"\nOVERALL: Best model is {best_model}")
    report.append("="*80)

    report_text = "\n".join(report)
    print("\n" + report_text)

    with open(OUTPUT_DIR / '04_summary.txt', 'w') as f:
        f.write(report_text)

    print(f"\n✓ Saved: {OUTPUT_DIR / '04_summary.txt'}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*80)
    print("BIAS-VARIANCE ANALYSIS")
    print("="*80)
    print("\nThis script will:")
    print("  1. Load BEST hyperparameters from most recent CV results")
    print("  2. Retrain models with those parameters")
    print("  3. Analyze train-validation gap to detect overfitting")
    print("  4. Generate diagnostic plots and recommendations")
    print("\nNOTE: If no CV results found, will use baseline parameters for comparison")
    print("="*80)

    X_train, y_train, X_val, y_val, _, _ = load_and_prepare_data()

    # use_tuned_params=True will auto-load best params from CV results
    gap_df = train_validation_gap_analysis(X_train, y_train, X_val, y_val, use_tuned_params=True)
    best_iter = xgboost_iteration_tracking(X_train, y_train, X_val, y_val, use_tuned_params=True)
    analyze_cv_fold_variance()
    generate_summary(gap_df, best_iter)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  • {f.name}")
    print("\nTo verify improvements:")
    print("  • Compare train-val gaps in 01_train_val_gap.png")
    print("  • Check if gaps reduced from baseline (RF: 14.8%, XGB: 12.8%)")
    print("  • Review 04_summary.txt for updated diagnostics")


if __name__ == "__main__":
    main()
