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
│   └── feature_config.json          # Training-time configuration (tracked in git)
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

#### 3. Preprocessing (To be implemented)
- Model-specific transformations
- One-hot encoding for categorical features (Logistic Regression)
- StandardScaler for numeric features (Logistic Regression)
- Minimal preprocessing for tree models (XGBoost, Random Forest)

#### 4. Baseline Models (To be implemented)
- Logistic Regression with class weights
- Random Forest with class weights
- XGBoost with scale_pos_weight
- Initial performance comparison

#### 5. Hyperparameter Tuning (To be implemented)
- Grid search or randomized search
- Cross-validation with stratified folds
- Optimize for ROC-AUC or F1 score

#### 6. Model Evaluation (To be implemented)
- ROC-AUC, F1, Precision, Recall scores
- Confusion matrices
- ROC curves and Precision-Recall curves
- Feature importance analysis

#### 7. Final Model Selection (To be implemented)
- Compare all models on validation set
- Select best performing model
- Final evaluation on test set
- Save model for deployment

## Notebook Best Practices

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
With a 44:1 imbalance ratio, consider:
- Stratified sampling (already implemented in splits)
- SMOTE or other oversampling techniques
- Class weights in model training
- Appropriate metrics (F1, ROC-AUC, PR-AUC instead of accuracy)

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
- Model deployment with FastAPI
- Real-time prediction API
- Model monitoring and drift detection
- Feature engineering pipeline
- Automated retraining workflow
