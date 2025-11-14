# E-Commerce Fraud Detection Project

## Overview
This project builds machine learning models to detect fraudulent e-commerce transactions using a realistic synthetic dataset from Kaggle. The dataset models real-life fraudulent activity patterns observed in 2024, including:
- Cards tested with $1 purchases at midnight
- Transactions shipping "gaming accessories" 5,000 km away
- Promo code reuse from freshly created accounts

**Dataset Source**: [Kaggle - E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)

## Project Structure

```
.
├── fraud_detection_EDA_FE.ipynb    # EDA & feature engineering notebook
├── fraud_detection_modeling.ipynb  # Model training & evaluation notebook
├── data/                            # Data directory (gitignored)
│   ├── transactions.csv             # Downloaded dataset (~300k rows, 17 columns)
│   ├── train_features.pkl           # Engineered training set (179,817 × 31)
│   ├── val_features.pkl             # Engineered validation set (59,939 × 31)
│   └── test_features.pkl            # Engineered test set (59,939 × 31)
├── src/                             # Source code for production
│   └── preprocessing/               # Feature engineering pipeline
│       ├── config.py                # FeatureConfig dataclass
│       ├── features.py              # Feature engineering functions
│       ├── transformer.py           # FraudFeatureTransformer (sklearn-compatible)
│       └── __init__.py              # Package exports
├── tests/                           # Test suite
│   ├── conftest.py                  # Shared pytest fixtures
│   └── test_preprocessing/          # Preprocessing tests
│       ├── test_config.py           # FeatureConfig tests
│       ├── test_features.py         # Feature function tests
│       └── test_transformer.py      # Transformer integration tests
├── models/                          # Model artifacts (gitignored except .json)
│   ├── feature_config.json          # Training-time configuration (tracked in git)
│   └── logs/                        # Hyperparameter tuning logs (gitignored)
│       ├── *_tuning_*.log           # Timestamped CV progress logs
│       └── *_cv_results_*.csv       # CV results for analysis
├── pyproject.toml                   # Python dependencies
├── uv.lock                          # Locked dependency versions
├── .python-version                  # Python version specification
├── .gitignore                       # Git exclusions
├── claude.md                        # This file
└── README.md                        # Project readme
```

## Dataset Information

### Original Dataset (Raw)
- **Source**: Kaggle - E-Commerce Fraud Detection Dataset
- **Rows**: 299,695 transactions
- **Columns**: 17 original features
- **Target**: `is_fraud` (binary: 0=normal, 1=fraud)
- **Class Distribution**:
  - Normal: 97.8%
  - Fraud: 2.2%
  - **Class Imbalance Ratio**: 44.3:1 (highly imbalanced!)
- **Data Quality**: No missing values, no duplicates
- **Memory usage**: ~107 MB

### Original Features (17)
- **Transaction Identifiers**: `transaction_id`, `user_id`
- **User Behavior**: `account_age_days`, `total_transactions_user`, `avg_amount_user`
- **Transaction Details**: `amount`, `transaction_time`, `merchant_category`
- **Geographic**: `country`, `bin_country`, `shipping_distance_km`
- **Security Flags**: `avs_match`, `cvv_result`, `three_ds_flag`
- **Channel & Promotions**: `channel` (web/app), `promo_used`

### Engineered Dataset (After Feature Engineering)
- **Total Features**: 30 features + 1 target = 31 columns
- **Splits**: 60% train, 20% validation, 20% test (stratified)
- **Format**: Pickle files for fast loading
- **Feature Categories**:
  1. **Original Numeric (5)**: account_age_days, total_transactions_user, avg_amount_user, amount, shipping_distance_km
  2. **Original Categorical (5)**: channel, promo_used, avs_match, cvv_result, three_ds_flag
  3. **Temporal Local (6)**: hour_local, day_of_week_local, month_local, is_weekend_local, is_late_night_local, is_business_hours_local
  4. **Amount Features (4)**: amount_deviation, amount_vs_avg_ratio, is_micro_transaction, is_large_transaction
  5. **User Behavior (3)**: transaction_velocity, is_new_account, is_high_frequency_user
  6. **Geographic (3)**: country_mismatch, high_risk_distance, zero_distance
  7. **Security (1)**: security_score
  8. **Interaction (3)**: new_account_with_promo, late_night_micro_transaction, high_value_long_distance

### Feature Selection Decisions
- **Excluded UTC temporal features** (6): Local time more meaningful for fraud patterns
- **Excluded country/bin_country** (2): Replaced by country_mismatch (more specific)
- **Excluded merchant_category** (1): Low predictive signal (all near baseline)
- **Excluded redundant security** (3): verification_failures, all_verifications_passed/failed
- **Excluded generic interactions** (3): Covered by base features
- **Total reduced**: From 45 available → 30 selected (33% reduction)

## Technical Stack

### Core Dependencies (from pyproject.toml)
- **Python**: 3.12+
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **ML Models**: scikit-learn, xgboost
- **Statistics**: statsmodels
- **Timezone Handling**: pytz (for UTC to local time conversion)
- **Data Source**: kaggle (API client)
- **Notebook**: jupyter
- **Testing**: pytest (for unit and integration tests)
- **API (future)**: fastapi, uvicorn

### Package Management
This project uses `uv` for fast, reliable Python dependency management.

## Production Feature Engineering Pipeline

### Overview
The production feature engineering pipeline is implemented in `src/preprocessing/` as a scikit-learn compatible transformer. This architecture enables seamless integration with sklearn pipelines and consistent feature engineering between training and inference.

### Architecture: Option 4 (Hybrid Class + Config)

**Design Pattern**: Sklearn-compatible transformer with external JSON configuration

**Key Benefits**:
- ✅ Sklearn Pipeline compatible (fit/transform pattern)
- ✅ Lightweight serialization (JSON config, not pickled Python objects)
- ✅ Version control friendly (config changes visible in git diffs)
- ✅ Type-safe configuration (dataclass with validation)
- ✅ Testable (unit tests for each component)
- ✅ Production-ready (industry standard pattern)

### Module Structure

#### 1. `src/preprocessing/config.py` - Configuration Management
**Purpose**: Type-safe configuration for feature engineering

**FeatureConfig dataclass** stores training-time statistics:
- `amount_95th_percentile`: Threshold for is_large_transaction feature
- `total_transactions_75th_percentile`: Threshold for is_high_frequency_user feature
- `shipping_distance_75th_percentile`: Threshold for high_risk_distance feature
- `timezone_mapping`: Dict mapping country codes to capital city timezones (10 countries)
- `final_features`: List of 30 selected features for model input
- `date_col`: Name of datetime column (default: 'transaction_time')
- `country_col`: Name of country column (default: 'country')

**Methods**:
- `from_training_data(train_df)`: Calculate thresholds from training set
- `save(path)`: Serialize to JSON file
- `load(path)`: Deserialize from JSON file

**Usage**:
```python
from src.preprocessing import FeatureConfig

# During training (in EDA notebook)
config = FeatureConfig.from_training_data(train_df)
config.save("models/feature_config.json")

# During inference (in API)
config = FeatureConfig.load("models/feature_config.json")
```

#### 2. `src/preprocessing/features.py` - Feature Engineering Functions
**Purpose**: Modular functions for each feature engineering step

**Helper Functions**:
- `get_country_timezone_mapping()`: Returns dict of country → timezone (10 countries)
- `get_final_feature_names()`: Returns list of 30 selected features (categorized)

**Feature Engineering Functions** (all return `Tuple[DataFrame, List[str]]`):
- `convert_to_local_time(df, date_col, country_col, timezone_mapping)`:
  - Converts UTC to local time by country capital timezone
  - Strict validation: raises ValueError if input not timezone-aware UTC
  - Returns timezone-naive local_time column

- `create_temporal_features(df, date_col, use_local_time=False)`:
  - Creates 6 features: hour, day_of_week, month, is_weekend, is_late_night, is_business_hours
  - Can create UTC or local time features (with '_local' suffix)

- `create_amount_features(df, amount_threshold)`:
  - Creates 4 features: amount_deviation, amount_vs_avg_ratio, is_micro_transaction, is_large_transaction
  - Handles division by zero (when avg_amount_user = 0)

- `create_user_behavior_features(df, transaction_threshold)`:
  - Creates 3 features: transaction_velocity, is_new_account, is_high_frequency_user
  - Handles division by zero (when account_age_days = 0)

- `create_geographic_features(df, distance_threshold)`:
  - Creates 3 features: country_mismatch, high_risk_distance, zero_distance

- `create_security_features(df)`:
  - Creates 4 features: security_score, verification_failures, all_verifications_passed, all_verifications_failed
  - Only security_score used in final 30 features

- `create_interaction_features(df)`:
  - Creates 6 features targeting fraud scenarios
  - Only 3 used in final 30: new_account_with_promo, late_night_micro_transaction, high_value_long_distance

#### 3. `src/preprocessing/transformer.py` - Sklearn Transformer
**Purpose**: Orchestrate complete feature engineering pipeline

**FraudFeatureTransformer class** (inherits from `BaseEstimator`, `TransformerMixin`):

**Methods**:
- `__init__(config=None)`: Initialize with optional configuration
- `fit(X, y=None)`: Calculate FeatureConfig from training data, return self
- `transform(X)`: Apply full pipeline, return DataFrame with 30 features
- `fit_transform(X, y=None)`: Convenience method (fit + transform)
- `save(path)`: Save config to JSON
- `load(path)`: Class method to load from JSON config

**Pipeline Steps** (executed in `transform()`):
1. **Preprocessing**: Convert transaction_time to UTC timezone-aware datetime
2. **Timezone conversion**: UTC → local time by country
3. **Temporal features (UTC)**: 6 features (excluded from final 30)
4. **Temporal features (local)**: 6 features (included in final 30)
5. **Amount features**: 4 features using amount_95th_percentile threshold
6. **User behavior features**: 3 features using transaction_75th_percentile threshold
7. **Geographic features**: 3 features using distance_75th_percentile threshold
8. **Security features**: 4 features (only security_score in final 30)
9. **Interaction features**: 6 features (only 3 in final 30)
10. **Feature selection**: Return only 30 selected features

**Usage**:
```python
from src.preprocessing import FraudFeatureTransformer

# Training workflow
transformer = FraudFeatureTransformer()
transformer.fit(train_df)  # Calculates quantile thresholds
X_train = transformer.transform(train_df)
transformer.save("models/transformer_config.json")

# Inference workflow
transformer = FraudFeatureTransformer.load("models/transformer_config.json")
X_new = transformer.transform(new_df)

# Sklearn Pipeline integration
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('feature_engineering', FraudFeatureTransformer()),
    ('model', LogisticRegression())
])
pipeline.fit(train_df, y_train)
predictions = pipeline.predict(test_df)
```

### Testing Strategy

**Test Coverage**: Unit tests for each component + integration tests

**Test Structure**:
- `tests/conftest.py`: Shared pytest fixtures
  - `sample_raw_df()`: Small raw DataFrame (10 rows)
  - `sample_raw_df_utc()`: Same data with UTC timestamps
  - `sample_config()`: Pre-configured FeatureConfig
  - `fitted_transformer()`: Fitted FraudFeatureTransformer
  - `sample_engineered_df()`: Transformed data (30 features)

- `tests/test_preprocessing/test_config.py`: FeatureConfig tests
  - Configuration creation and validation
  - Save/load round-trip testing
  - JSON structure verification
  - Quantile calculation from training data

- `tests/test_preprocessing/test_features.py`: Feature function tests
  - Individual function testing (isolation)
  - Edge case handling (zero values, division by zero)
  - Timezone validation (strict UTC enforcement)
  - Binary feature output verification

- `tests/test_preprocessing/test_transformer.py`: Integration tests
  - Full pipeline execution
  - Output shape verification (30 features)
  - Sklearn Pipeline compatibility
  - Save/load consistency
  - Multiple transform consistency

**Run Tests**:
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/preprocessing --cov-report=html

# Run specific test file
uv run pytest tests/test_preprocessing/test_transformer.py
```

### Notebook Integration

The EDA notebook (`fraud_detection_EDA_FE.ipynb`) now includes a cell that automatically generates and saves the FeatureConfig:

```python
# Create and save feature configuration for deployment
from src.preprocessing import FeatureConfig

feature_config = FeatureConfig.from_training_data(train_fe)
feature_config.save("models/feature_config.json")
```

This config file (`models/feature_config.json`) is:
- ✅ Tracked in git (added to repo)
- ✅ Human-readable JSON format
- ✅ Contains all training-time statistics needed for inference
- ✅ Used by transformer during deployment

### Design Decisions

1. **Config as JSON (not pickle)**:
   - Version control friendly (diffs are readable)
   - Lightweight (no Python object serialization)
   - Cross-language compatible (can be read by non-Python services)

2. **Quantile thresholds from training data**:
   - Prevents data leakage (test set never seen during threshold calculation)
   - Consistent between training and inference
   - Stored in config for reproducibility

3. **Strict timezone validation**:
   - Fails fast if input data missing timezone info
   - Prevents silent errors from timezone assumptions
   - Clear error messages guide users to fix data issues

4. **30 features hardcoded in `get_final_feature_names()`**:
   - Explicit feature selection (no magic)
   - Easy to audit and modify
   - Clear categorization (original, temporal, amount, etc.)

5. **Sklearn-compatible transformer**:
   - Standard fit/transform pattern
   - Works with sklearn Pipeline
   - Familiar API for ML practitioners

## Development Setup

### Prerequisites
1. **Kaggle API Credentials**: Place `kaggle.json` in `~/.kaggle/`
   - Get from: https://www.kaggle.com/account
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. **Python 3.12+**: Check version with `python --version`

3. **uv package manager**: Install if needed

### Installation
```bash
# Install dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook
```

### Data Download
The notebook automatically downloads the dataset from Kaggle on first run if not present in `./data/`. The download function checks for existing data to avoid redundant downloads.

## Notebook Structure

### fraud_detection_EDA_FE.ipynb (EDA & Feature Engineering)

#### 1. Setup
- Parameter definitions (data paths, split ratios, feature lists)
- Package imports (pandas, numpy, matplotlib, seaborn, sklearn, statsmodels, pytz)
- Utility function definitions

#### 2. Data Loading
- `download_data_csv()`: Kaggle API download with caching
- `load_data()`: Efficient pandas CSV loading

#### 3. Preprocessing
- Table grain verification
- Target class balance analysis (before splitting for stratification decision)
- Date type conversion with UTC enforcement (`pd.to_datetime(..., utc=True)`)
- Train/validation/test splits (60/20/20, stratified on target)

#### 4. Exploratory Data Analysis (EDA)
- Baseline fraud rate calculation
- Numeric feature distributions with histograms
- Multicollinearity detection (VIF analysis)
- Correlations with target (Pearson for numeric)
- Box plots (fraud vs non-fraud comparison)
- Temporal patterns analysis (hour, day_of_week, weekend, month)
- Categorical feature fraud rates
- Mutual information scores for categorical features
- Initial feature selection recommendations

#### 5. Feature Engineering
- Timezone conversion (UTC → local time by country capital)
- Temporal features (UTC and local): hour, day_of_week, is_late_night, etc.
- Amount features: deviations, ratios, micro/large transaction flags
- User behavior: velocity, new account, high frequency flags
- Geographic: country mismatch, high-risk distance, zero distance
- Security: composite security score
- Interaction features: fraud scenario-specific combinations
- **Output**: 32 engineered features created

#### 6. Final Feature Selection
- Analysis of all 45 available features (13 original + 32 engineered)
- Elimination of redundant features (UTC, country fields, etc.)
- Selection criteria: EDA insights, fraud scenarios, interpretability
- **Output**: 30 selected features stored in categorized lists

#### 7. Dataset Persistence
- Save train/val/test DataFrames with selected features as pickle files
- Format: 30 features + 1 target = 31 columns
- Location: `data/train_features.pkl`, `data/val_features.pkl`, `data/test_features.pkl`

### fraud_detection_modeling.ipynb (Model Training & Evaluation)

#### 1. Setup
- Parameter definitions (data paths, random seed, model directory)
- Package imports (sklearn, xgboost, preprocessing, metrics)

#### 2. Data Loading
- Load pre-engineered datasets from pickle files
- Feature type identification (numeric, categorical, binary)
- Dataset shape and target distribution validation

#### 3. Preprocessing Pipeline
- **Sklearn Pipeline Architecture**: Model-agnostic preprocessing using ColumnTransformer
- **Numeric Features**: StandardScaler applied (Logistic Regression only)
- **Categorical Features**: OneHotEncoder with `drop='first'` to avoid multicollinearity
- **Tree Models**: Minimal preprocessing (XGBoost and Random Forest handle raw features)
- **Pipeline Configuration**: Stored in `pipeline_lr`, `pipeline_rf`, `pipeline_xgb`

#### 4. Baseline Models
- **Logistic Regression**: `class_weight='balanced'`, max_iter=1000, random_state=42
  - Performance: PR-AUC 0.6975, Precision 41.54%, Recall 83.60%
  - Weakness: High false positive rate (2,841 FP)

- **Random Forest**: `class_weight='balanced'`, n_estimators=100, random_state=42
  - Performance: PR-AUC 0.8456, Precision 94.19%, Recall 71.13%
  - Strength: Excellent precision, low false positives

- **XGBoost**: `scale_pos_weight=44.3` (class imbalance ratio), n_estimators=100
  - Performance: PR-AUC 0.8460, Precision 54.78%, Recall 84.05%
  - Issue: Recall-dominated, high false positives (918 FP)

**Helper Functions**:
- `create_preprocessing_pipeline(numeric_features, categorical_features, scale_numeric)`: Creates sklearn Pipeline
- `train_and_evaluate_model(model_name, pipeline, X_train, y_train, X_val, y_val)`: Training and evaluation wrapper
- `compare_models(results_dict)`: Side-by-side model comparison with metrics

#### 5. Hyperparameter Tuning
- **Optimization Metric**: PR-AUC (Precision-Recall Area Under Curve) - ideal for imbalanced datasets
- **Cross-Validation**: 4-fold Stratified CV to maintain class distribution
- **Search Strategy**: Flexible GridSearchCV/RandomizedSearchCV switching via `create_search_object()`
- **Logging**: Comprehensive logs saved to `models/logs/` directory

**Helper Functions**:
- `create_search_object(search_type, estimator, param_grid, scoring, cv, n_iter, verbose, random_state, n_jobs)`:
  - Flexible switching between 'grid' and 'random' search strategies
  - Automatic calculation of total parameter combinations
  - Returns configured search object ready for fitting

- `tune_with_logging(search_type, pipeline, param_grid, X_train, y_train, cv, model_name, random_state, n_iter)`:
  - Executes hyperparameter search with progress logging
  - Saves detailed logs to timestamped files in `models/logs/`
  - Saves CV results to CSV for analysis
  - Returns: (search_object, log_path, csv_path)

- `analyze_cv_results(cv_results_csv_path, top_n=5)`:
  - Production-focused analysis of CV results
  - Identifies best model by PR-AUC and stability (std_test_score)
  - **⚠ Timing Caveats**: Includes warnings about unreliable timing from parallel processing (n_jobs=-1)
  - Displays comprehensive metrics with reliability labels (✓ Reliable / ⚠ Unreliable)
  - Returns best parameters dictionary

**Random Forest Tuning**:
- **Search Type**: GridSearchCV (8 combinations)
- **Parameter Grid**:
  - n_estimators: [350, 400, 450, 500]
  - max_depth: [25, 30]
  - min_samples_split: [2]
  - min_samples_leaf: [2]
  - max_features: ['sqrt']
  - class_weight: ['balanced_subsample']
- **Best Parameters**: n_estimators=500, max_depth=30, min_samples_leaf=2
- **Performance**: PR-AUC 0.8583 (+1.5% vs baseline)
- **Trade-off**: Sacrificed 4.4% precision to gain 7.3% recall

**XGBoost Tuning**:
- **Search Type**: GridSearchCV (108 combinations)
- **Parameter Grid**:
  - n_estimators: [90, 100, 110]
  - max_depth: [4, 5]
  - learning_rate: [0.08, 0.1, 0.12]
  - subsample: [0.9]
  - colsample_bytree: [0.9]
  - min_child_weight: [5]
  - gamma: [0.5, 0.6]
  - scale_pos_weight: [8, 10, 12] ⭐ **Key parameter - tunable, not fixed at class ratio**
  - eval_metric: ['aucpr'] (changed from 'logloss')
- **Best Parameters**: n_estimators=90, max_depth=5, learning_rate=0.08, scale_pos_weight=8, gamma=0.6
- **Performance**: PR-AUC 0.8679 (+2.6% vs baseline)
- **Major Win**: Precision 54.78% → 72.33% (+32.1% improvement!)
- **False Positive Reduction**: 918 → 423 (54% reduction)

**Key Insights**:
- Making `scale_pos_weight` tunable (not just using class imbalance ratio) was crucial
- Changing `eval_metric` to 'aucpr' aligned training with optimization goal
- Shallow trees (max_depth=5) consistently outperformed deeper trees
- High regularization (gamma=0.6, min_child_weight=5) essential for precision

#### 6. Model Evaluation
- **Primary Metrics**: PR-AUC (optimization target), ROC-AUC, F1 Score
- **Secondary Metrics**: Precision, Recall, Confusion Matrix
- **Evaluation Functions**:
  - `calculate_metrics(y_true, y_pred, y_pred_proba)`: Comprehensive metric calculation
  - `print_evaluation_metrics(metrics_dict, model_name, dataset_name)`: Formatted metric display
  - `plot_confusion_matrix(y_true, y_pred, model_name)`: Confusion matrix visualization

**Model Comparison (Validation Set)**:

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall | FP | FN |
|-------|--------|---------|-------|-----------|--------|-----|-----|
| Logistic Regression | 0.6975 | 0.9647 | 0.5523 | 41.54% | 83.60% | 2,841 | 217 |
| Random Forest (Baseline) | 0.8456 | 0.9762 | 0.8146 | 94.19% | 71.13% | 89 | 382 |
| Random Forest (Tuned) | 0.8583 | 0.9777 | 0.8257 | 90.02% | 76.34% | 146 | 313 |
| XGBoost (Baseline) | 0.8460 | 0.9767 | 0.6633 | 54.78% | 84.05% | 918 | 211 |
| **XGBoost (Tuned)** | **0.8679** | **0.9790** | **0.7756** | **72.33%** | **83.60%** | **423** | **217** |

**Performance Targets vs Achieved (XGBoost Tuned)**:
- ✅ PR-AUC > 0.85: **0.8679**
- ✅ ROC-AUC > 0.95: **0.9790**
- ✅ F1 > 0.75: **0.7756**
- ✅ Precision > 0.70: **0.7233**
- ✅ Recall > 0.80: **0.8360**

#### 7. Final Model Selection
- **Selected Model**: XGBoost (Tuned) with PR-AUC 0.8679
- **Rationale**:
  - Best PR-AUC score (primary optimization metric)
  - Excellent precision-recall balance for production deployment
  - All performance targets exceeded
  - Significant improvement over baseline (+32.1% precision, +2.6% PR-AUC)
  - 54% reduction in false positives vs XGBoost baseline

- **Use Cases by Model**:
  - **XGBoost (Tuned)**: Production deployment - best overall balance
  - **Random Forest (Tuned)**: Applications requiring very low false positive rates (precision 90%)

- **Next Steps**:
  - Evaluate XGBoost (Tuned) on held-out test set
  - Analyze feature importance
  - Consider threshold optimization for custom precision-recall trade-offs
  - Prepare model serialization for deployment

## Notebook Best Practices

### Import Organization

**Core Principle**: Keep all imports at the top of the notebook in the "Import packages" section. Never add imports scattered throughout the notebook.

#### Import Guidelines
1. **Centralized imports**: All `import` and `from ... import` statements go in the "Import packages" section at the top
2. **Alphabetical ordering**: Maintain alphabetical order for readability and easier scanning
3. **Grouping**: Organize imports into logical groups (standard library, third-party, local)
4. **No inline imports**: Avoid importing packages in the middle of analysis cells

#### Example Structure
```python
# Standard library
import sys
from datetime import datetime
from pathlib import Path

# Third-party packages (alphabetical)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

# Sklearn (grouped and alphabetical)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Local imports (if any)
from src.preprocessing import FeatureConfig, FraudFeatureTransformer
```

### Keep Cells Clean with Functions

**Core Principle**: Notebook cells should contain minimal logic - ideally just a single function call. All complex logic should be encapsulated in well-named functions defined in the "Define functions" section.

#### Benefits
1. **Readability**: Cells are easy to scan and understand at a glance
2. **Reusability**: Functions can be called multiple times or on different datasets
3. **Maintainability**: Changes to logic happen in one place
4. **Testability**: Functions can be unit tested
5. **Organization**: All logic is centralized in the functions section

#### Examples

**❌ Bad Practice** - Complex logic directly in cell:
```python
# Cell with 50+ lines of matplotlib code
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()
for idx, col in enumerate(numeric_features):
    ax = axes[idx]
    train_df[col].hist(bins=50, ax=ax, alpha=0.7, color='steelblue')
    # ... 40 more lines of plotting code
```

**✅ Good Practice** - Clean single function call:
```python
# Cell with single function call
plot_numeric_distributions(train_df, numeric_features)
```

#### Implementation Pattern

1. **Define functions in "Define functions" section**:
   ```python
   def plot_numeric_distributions(df, numeric_features):
       """Visualize distributions of numeric features with histograms."""
       # All plotting logic here
       fig, axes = plt.subplots(3, 2, figsize=(14, 12))
       # ... implementation
       plt.show()
       print("\nKey Observations:...")
   ```

2. **Call functions in notebook cells**:
   ```python
   plot_numeric_distributions(train_df, numeric_features)
   ```

#### Function Naming Conventions
- **Action-based names**: `analyze_`, `plot_`, `calculate_`, `print_`
- **Descriptive**: Name should clearly indicate what the function does
- **Examples**:
  - `analyze_vif()` - Calculate and visualize VIF
  - `plot_categorical_fraud_rates()` - Visualize fraud rates by category
  - `analyze_temporal_patterns()` - Analyze time-based fraud patterns

#### When to Create a Function
Create a function when:
- Logic exceeds ~5-10 lines
- Code involves visualization (matplotlib/seaborn)
- Analysis might be reused or repeated
- Cell logic becomes hard to read at a glance

Keep inline when:
- Single line operations (e.g., `df.head()`)
- Simple variable assignments
- Direct function calls to existing functions

## Key Functions

### Data Loading & Preprocessing
- `download_data_csv(kaggle_source, data_dir, csv_file)`: Download from Kaggle with caching
- `load_data(data_dir, csv_file, verbose)`: Load CSV efficiently
- `split_train_val_test(df, val_ratio, test_ratio, stratify, r_seed)`: Create train/val/test splits with stratification

### Preprocessing & Analysis Functions
- `analyze_target_stats(df, target_col)`: Target distribution and imbalance detection with visualization
- `analyze_feature_stats(df, id_cols, target_col, categorical_features, numeric_features)`: Feature summary statistics
- `calculate_mi_scores(df, categorical_features, target_col)`: Mutual information for categorical features
- `calculate_numeric_correlations(df, numeric_features, target_col)`: Pearson correlations
- `calculate_vif(df, numeric_features)`: Variance Inflation Factor for multicollinearity

### EDA Visualization Functions
- `plot_numeric_distributions(df, numeric_features)`: Histogram visualizations with mean/median lines
- `analyze_vif(df, numeric_features)`: VIF calculation and visualization with interpretation
- `analyze_correlations(df, numeric_features, target_col)`: Correlation analysis with bar chart visualization
- `plot_box_plots(df, numeric_features, target_col)`: Box plots comparing fraud vs non-fraud distributions
- `analyze_temporal_patterns(df, date_feature, target_col, baseline_fraud_rate)`: Time-based fraud patterns by hour/day/month
- `analyze_categorical_fraud_rates(df, categorical_features, target_col)`: Fraud rate calculations by category with high-risk identification
- `plot_categorical_fraud_rates(df, categorical_features, target_col, baseline_fraud_rate)`: Visualize categorical fraud rates with baseline comparison
- `analyze_mutual_information(df, categorical_features, target_col)`: MI score calculation and visualization
- `print_feature_recommendations(corr_df, mi_df, vif_df, numeric_features, categorical_features)`: Comprehensive feature recommendations based on EDA

### Feature Engineering Functions
- `get_country_timezone_mapping()`: Returns dictionary mapping countries to capital city timezones
- `convert_utc_to_local_time(df, date_col, country_col)`: Convert UTC to local time with timezone validation
- `create_temporal_features(df, date_col, use_local_time)`: Generate temporal features (hour, day_of_week, is_late_night, etc.)
- `create_amount_features(df)`: Transaction amount patterns (deviation, ratios, micro/large flags)
- `create_user_behavior_features(df)`: User account patterns (velocity, new account, high frequency)
- `create_geographic_features(df, risk_distance_quantile)`: Geographic features (country mismatch, distance flags)
- `create_security_features(df)`: Security verification features (composite score, failures)
- `create_interaction_features(df)`: Fraud scenario-specific interaction features
- `engineer_features(df, date_col, country_col)`: Master function to create all 32 engineered features with progress logging

### Feature Selection Function
- `analyze_final_feature_selection(train_new_features)`: Comprehensive feature selection analysis that returns categorized dictionary of 30 selected features with rationale for inclusions/exclusions

### Model Training & Evaluation Functions

#### Preprocessing Functions
- `create_preprocessing_pipeline(numeric_features, categorical_features, scale_numeric=True)`: Creates sklearn Pipeline with ColumnTransformer for numeric scaling and categorical encoding

#### Training & Evaluation Functions
- `train_and_evaluate_model(model_name, pipeline, X_train, y_train, X_val, y_val)`: Trains model pipeline and evaluates on validation set, returns metrics dictionary
- `compare_models(results_dict)`: Compares multiple models side-by-side with formatted output of key metrics
- `calculate_metrics(y_true, y_pred, y_pred_proba)`: Calculates comprehensive metrics (PR-AUC, ROC-AUC, F1, Precision, Recall, Confusion Matrix)
- `print_evaluation_metrics(metrics_dict, model_name, dataset_name='Validation')`: Formatted display of evaluation metrics with confusion matrix breakdown
- `plot_confusion_matrix(y_true, y_pred, model_name)`: Visualizes confusion matrix with seaborn heatmap

#### Hyperparameter Tuning Functions
- `create_search_object(search_type, estimator, param_grid, scoring='average_precision', cv=4, n_iter=None, verbose=1, random_state=42, n_jobs=-1)`:
  - Creates either GridSearchCV or RandomizedSearchCV based on `search_type` parameter ('grid' or 'random')
  - Calculates and displays total parameter combinations
  - Flexible switching between exhaustive and sampling-based search strategies
  - Returns configured search object ready for fitting

- `tune_with_logging(search_type, pipeline, param_grid, X_train, y_train, cv, model_name, random_state=42, n_iter=None)`:
  - Executes hyperparameter search with comprehensive logging
  - Creates timestamped log files in `models/logs/` directory
  - Saves CV results to CSV for post-analysis
  - Supports both GridSearchCV and RandomizedSearchCV
  - Returns: (search_object, log_path, csv_path)

- `analyze_cv_results(cv_results_csv_path, top_n=5)`:
  - Production-focused analysis of cross-validation results
  - Identifies best model by PR-AUC score and stability (std_test_score)
  - Displays top N parameter combinations with comprehensive metrics
  - **Includes timing caveats** for parallel processing (n_jobs=-1) unreliability
  - Labels metrics as "✓ Reliable" (PR-AUC, std) or "⚠ Unreliable" (timing)
  - Returns best parameters dictionary for easy model instantiation

## Important Notes

### Gitignore
The following are excluded from version control:
- `data/` directory (contains large CSV files)
- `.kaggle/` directory and `kaggle.json` (credentials)
- Jupyter checkpoints
- Python cache files
- Virtual environments
- Model artifacts (`.pkl`, `.h5`, `.pt`, etc.)

### Class Imbalance Strategy
With a 44:1 imbalance ratio, the following techniques are implemented:
- ✅ **Stratified sampling**: Applied in train/val/test splits (60/20/20)
- ✅ **Class weights**:
  - Logistic Regression: `class_weight='balanced'`
  - Random Forest: `class_weight='balanced'` and `class_weight='balanced_subsample'` (tuned)
  - XGBoost: `scale_pos_weight` parameter (tuned from 8-12, optimal=8)
- ✅ **Appropriate metrics**: PR-AUC as primary optimization metric (ideal for imbalanced data)
  - Secondary metrics: ROC-AUC, F1 Score, Precision, Recall
  - Avoid accuracy as evaluation metric
- ✅ **Stratified Cross-Validation**: 4-fold StratifiedKFold during hyperparameter tuning
- ❌ **SMOTE not used**: Class weights proved sufficient for strong performance

### Data Split Configuration
Default split ratios (configurable in notebook):
- Training: 60%
- Validation: 20%
- Test: 20%
- Stratification: Applied on `is_fraud` target

## Common Tasks

### Add new dependencies
```bash
uv add <package-name>
```

### Update dependencies
```bash
uv sync
```

### Run notebook
```bash
uv run jupyter notebook ec_fraud_detection.ipynb
```

### Git workflow
```bash
# Check status
git status

# Commit changes
git add <files>
git commit -m "Description"

# Push to remote
git push
```

## Model Development Guidelines

1. **Always use stratified splits** to maintain class distribution
2. **Track metrics appropriate for imbalanced data**: F1, ROC-AUC, Precision-Recall
3. **Avoid data leakage**: Keep test set completely separate until final evaluation
4. **Document experiments**: Record model configurations, hyperparameters, and results
5. **Handle class imbalance**: Use appropriate techniques (weights, sampling, threshold tuning)

## Future Enhancements

### Completed ✅
- ✅ **Feature engineering pipeline**: Implemented as sklearn-compatible transformer in `src/preprocessing/`
- ✅ **Baseline model training**: Logistic Regression, Random Forest, XGBoost
- ✅ **Hyperparameter tuning**: GridSearchCV/RandomizedSearchCV with comprehensive logging
- ✅ **Model selection**: XGBoost (Tuned) selected as best performer

### Remaining 🚧
- **Test set evaluation**: Final evaluation of XGBoost (Tuned) on held-out test set
- **Feature importance analysis**: Understand key fraud detection drivers
- **Model serialization**: Save final model for deployment (pickle or joblib)
- **Threshold optimization**: Custom precision-recall trade-offs for different use cases
- **Model deployment with FastAPI**: Production API endpoint for real-time predictions
- **Model monitoring and drift detection**: Track performance degradation over time
- **Automated retraining workflow**: Pipeline for periodic model updates
