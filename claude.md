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
├── train.py                         # Model training script
├── predict.py                       # FastAPI web service for real-time fraud prediction
├── Dockerfile                       # Multi-stage Docker image definition
├── docker-compose.yml               # Docker Compose configuration for local deployment
├── requirements.txt                 # Python dependencies for Docker
├── notebooks/                       # Jupyter notebooks
│   ├── fd1_EDA_FE.ipynb             # Notebook 1: EDA & feature engineering
│   ├── fd2_model_selection_tuning.ipynb    # Notebook 2: Model selection & tuning
│   └── fd3_model_evaluation_deployment.ipynb # Notebook 3: Evaluation & deployment
├── docs/                            # Supplementary documentation
│   ├── CHANGES_SUMMARY.md           # Summary of major changes
│   ├── GCP_DEPLOYMENT.md            # Google Cloud Platform deployment guide
│   └── REFACTORING_SUMMARY.md       # Code refactoring documentation
├── analysis/                        # Analysis scripts and outputs
│   └── bias_variance_analysis.py    # Bias-variance diagnostics script
├── benchmarks/                      # Performance testing suite
│   ├── benchmark.py                 # Custom benchmarking script
│   ├── locustfile.py                # Locust load testing configuration
│   └── results/                     # Generated outputs (gitignored)
├── data/                            # Data directory (gitignored)
│   └── transactions.csv             # Downloaded dataset (~300k rows, 17 columns)
├── src/                             # Source code modules
│   ├── deployment/                  # Production code for model deployment
│   │   ├── config/                  # Configuration management
│   │   │   ├── data_config.py       # Data loading configuration
│   │   │   ├── model_config.py      # Hyperparameters & feature lists
│   │   │   ├── training_config.py   # CV strategy & thresholds
│   │   │   └── __init__.py          # Package exports
│   │   ├── data/                    # Data loading utilities
│   │   │   ├── loader.py            # load_and_split_data()
│   │   │   └── __init__.py          # Package exports
│   │   ├── preprocessing/           # Feature engineering pipeline
│   │   │   ├── config.py            # FeatureConfig dataclass
│   │   │   ├── features.py          # Feature engineering functions
│   │   │   ├── transformer.py       # FraudFeatureTransformer (sklearn-compatible)
│   │   │   ├── pipelines.py         # PreprocessingPipelineFactory
│   │   │   └── __init__.py          # Package exports
│   │   ├── evaluation/              # Model evaluation utilities
│   │   │   ├── metrics.py           # evaluate_model()
│   │   │   ├── thresholds.py        # optimize_thresholds()
│   │   │   └── __init__.py          # Package exports
│   │   └── explainability/          # SHAP-based prediction explanations
│   │       ├── __init__.py          # Package exports
│   │       ├── shap_explainer.py    # FraudExplainer class for SHAP explanations
│   │       └── feature_descriptions.py  # Human-readable feature names
│   ├── fd1_nb/                      # Notebook 1 utility functions (EDA & FE)
│   │   ├── __init__.py              # Package exports (21 functions)
│   │   ├── data_utils.py            # Data loading, splitting, analysis
│   │   ├── eda_utils.py             # EDA functions (VIF, correlations, MI)
│   │   └── feature_engineering.py   # Feature engineering utilities
│   ├── fd2_nb/                      # Notebook 2 utility functions (Model Selection & Tuning)
│   │   ├── __init__.py              # Package exports (14 functions)
│   │   ├── model_comparison.py      # Model comparison and visualization
│   │   ├── hyperparameter_tuning.py # GridSearchCV/RandomizedSearchCV utilities
│   │   ├── cv_analysis.py           # CV results analysis and train-val gap detection
│   │   └── bias_variance.py         # Bias-variance diagnostics
│   ├── fd3_nb/                      # Notebook 3 utility functions (Evaluation & Deployment)
│   │   ├── __init__.py              # Package exports (18 functions)
│   │   ├── evaluation.py            # Model evaluation and performance comparison
│   │   ├── visualization.py         # ROC/PR curves, feature importance, SHAP beeswarm plots
│   │   ├── threshold_optimization.py # Threshold optimization strategies
│   │   ├── feature_importance.py    # Feature importance (XGBoost gain + SHAP)
│   │   └── deployment.py            # Deployment artifact generation
│   └── README.md                    # Source code organization documentation
├── tests/                           # Test suite (425 passing tests)
│   ├── conftest.py                  # Shared pytest fixtures
│   ├── test_api.py                  # API integration tests (33 tests)
│   ├── test_config/                 # Shared config tests (44 tests)
│   │   ├── test_data_config.py      # DataConfig tests (16 tests)
│   │   ├── test_model_config.py     # ModelConfig tests (19 tests)
│   │   └── test_training_config.py  # TrainingConfig tests (9 tests)
│   ├── test_eda/                    # EDA utility tests (68 tests)
│   │   ├── test_data_utils.py       # Data loading/splitting tests (21 tests)
│   │   ├── test_eda_utils.py        # VIF, correlations, MI tests (27 tests)
│   │   └── test_feature_engineering.py # Feature engineering tests (20 tests)
│   ├── test_data/                   # Data loading tests (12 tests)
│   │   └── test_loader.py           # load_and_split_data tests
│   ├── test_evaluation/             # Evaluation tests (26 tests)
│   │   ├── test_metrics.py          # Metrics tests (14 tests)
│   │   └── test_thresholds.py       # Threshold tests (12 tests)
│   ├── test_fd2_nb/                 # Notebook 2 utility tests (63 tests)
│   │   ├── test_model_comparison.py # Model comparison tests (16 tests)
│   │   ├── test_hyperparameter_tuning.py # Tuning utility tests (15 tests)
│   │   ├── test_cv_analysis.py      # CV analysis tests (25 tests)
│   │   └── test_bias_variance.py    # Bias-variance tests (7 tests)
│   ├── test_fd3_nb/                 # Notebook 3 utility tests (83 tests)
│   │   ├── test_evaluation.py       # Evaluation tests (15 tests)
│   │   ├── test_feature_importance.py # Feature importance tests (19 tests)
│   │   ├── test_threshold_optimization.py # Threshold tests (23 tests)
│   │   ├── test_deployment.py       # Deployment tests (26 tests)
│   │   └── test_visualization.py    # SHAP beeswarm plot tests (7 tests)
│   ├── test_explainability/         # Explainability tests (9 tests)
│   │   └── test_shap_explainer.py   # FraudExplainer tests
│   └── test_preprocessing/          # Preprocessing tests (61 tests)
│       ├── test_config.py           # FeatureConfig tests (8 tests)
│       ├── test_features.py         # Feature function tests (15 tests)
│       ├── test_pipelines.py        # Pipeline factory tests (18 tests)
│       └── test_transformer.py      # Transformer integration tests (20 tests)
├── models/                          # Model artifacts
│   ├── xgb_fraud_detector.joblib    # Trained XGBoost model (gitignored)
│   ├── transformer_config.json      # Feature transformer configuration (tracked)
│   ├── threshold_config.json        # Fraud detection thresholds (tracked)
│   ├── model_metadata.json          # Model version and performance metrics (tracked)
│   ├── feature_lists.json           # Feature names and categories (tracked)
│   ├── best_params.json             # Optimal hyperparameters (output from fd2)
│   ├── validation_metrics.json      # Validation performance metrics (output from fd2)
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
- **Processing**: Applied on-the-fly using `FraudFeatureTransformer` pipeline
- **Feature Type Classification** (stored in `models/feature_lists.json`):
  - **Categorical (1)**: channel
  - **Continuous Numeric (12)**: account_age_days, total_transactions_user, avg_amount_user, amount, shipping_distance_km, hour_local, day_of_week_local, month_local, amount_deviation, amount_vs_avg_ratio, transaction_velocity, security_score
  - **Binary (17)**: promo_used, avs_match, cvv_result, three_ds_flag, is_weekend_local, is_late_night_local, is_business_hours_local, is_micro_transaction, is_large_transaction, is_new_account, is_high_frequency_user, country_mismatch, high_risk_distance, zero_distance, new_account_with_promo, late_night_micro_transaction, high_value_long_distance
- **Feature Groupings** (by engineering type):
  1. **Temporal Local (6)**: hour_local, day_of_week_local, month_local, is_weekend_local, is_late_night_local, is_business_hours_local
  2. **Amount Features (4)**: amount_deviation, amount_vs_avg_ratio, is_micro_transaction, is_large_transaction
  3. **User Behavior (3)**: transaction_velocity, is_new_account, is_high_frequency_user
  4. **Geographic (3)**: country_mismatch, high_risk_distance, zero_distance
  5. **Security (1)**: security_score
  6. **Interaction (3)**: new_account_with_promo, late_night_micro_transaction, high_value_long_distance
  7. **Original Features Retained (10)**: account_age_days, total_transactions_user, avg_amount_user, amount, shipping_distance_km, channel, promo_used, avs_match, cvv_result, three_ds_flag

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
- **API Deployment**: fastapi, uvicorn, pydantic (REST API for real-time predictions)
- **Model Serialization**: joblib (for saving/loading trained models)

### Package Management
This project uses `uv` for fast, reliable Python dependency management.

**Important**: Always use `uv add <package>` to install new dependencies, NOT `uv pip install <package>`. This ensures:
- Dependencies are added to `pyproject.toml`
- Lock file (`uv.lock`) is automatically updated
- Consistent dependency resolution across all environments

Example:
```bash
# ✅ Correct - adds to project dependencies
uv add requests locust pytest

# ❌ Wrong - does not update pyproject.toml
uv pip install requests
```

## Shared Infrastructure & Code Organization

### Overview
The project uses a modular architecture with shared modules in `src/` organized into two main categories:

1. **`src/deployment/`**: Production code for model training, evaluation, and API deployment
2. **`src/fd*_nb/`**: Notebook-specific utility functions for exploratory analysis

This separation enables:
- Minimal Docker deployments (only `src/deployment/`)
- Clear distinction between production and exploratory code
- Reusable notebook utilities without polluting production code

### Module Structure

#### 1. `src/deployment/config/` - Configuration Management
Centralizes all configuration for data loading, model hyperparameters, and training strategies.

**`src/deployment/config/data_config.py`**:
- `DataConfig.DEFAULT_RANDOM_SEED`: Default random seed (1) for reproducibility
- `DataConfig.TARGET_COLUMN`: Target column name ('is_fraud')
- `DataConfig.DATA_DIR`: Default data directory path
- Train/val/test split ratios (60/20/20)

**`src/deployment/config/model_config.py`**:
- `FeatureListsConfig.load()`: Loads feature categorization from `models/feature_lists.json`
- `ModelConfig.load_hyperparameters()`: Loads hyperparameters from model metadata or CV results
- `ModelConfig.get_param_grid()`: Returns parameter grid for GridSearchCV
- Fallback hyperparameters for XGBoost and Random Forest
- Supports loading from multiple sources: metadata, CV results, or custom JSON files

**`src/deployment/config/training_config.py`**:
- `TrainingConfig.get_cv_strategy()`: Returns StratifiedKFold(4) for cross-validation
- `TrainingConfig.get_threshold_targets()`: Returns target recall values for threshold optimization (80%, 85%, 90%)

#### 2. `src/deployment/data/` - Data Loading Utilities
Provides unified data loading and splitting functionality.

**`src/deployment/data/loader.py`**:
- `load_and_split_data(data_path, random_seed, verbose)`: Loads raw CSV, performs stratified train/val/test splits, returns 3 DataFrames
- Used by `train.py`, `analysis/bias_variance_analysis.py`, and can be used in notebooks
- Ensures consistent data splitting across all scripts

#### 3. `src/deployment/preprocessing/` - Feature Engineering Pipeline
Production-ready feature engineering with sklearn compatibility.

**`src/deployment/preprocessing/pipelines.py`**:
- `PreprocessingPipelineFactory.create_logistic_pipeline()`: Creates pipeline with StandardScaler + OneHotEncoder
- `PreprocessingPipelineFactory.create_tree_pipeline()`: Creates minimal pipeline (OrdinalEncoder only) for tree models
- Used by `train.py` and `analysis/bias_variance_analysis.py` for consistent preprocessing

**Other modules**: See "Production Feature Engineering Pipeline" section below for details on `config.py`, `features.py`, `transformer.py`.

#### 4. `src/deployment/evaluation/` - Model Evaluation Utilities
Provides standardized model evaluation and threshold optimization.

**`src/deployment/evaluation/metrics.py`**:
- `evaluate_model(model, X, y, model_name, dataset_name)`: Comprehensive evaluation with PR-AUC, ROC-AUC, F1, Precision, Recall
- Prints formatted results with confusion matrix
- Returns metrics dictionary
- Used by `train.py` and `analysis/bias_variance_analysis.py`

**`src/deployment/evaluation/thresholds.py`**:
- `optimize_thresholds(model, X_val, y_val)`: Finds optimal thresholds for 80%, 85%, 90% recall targets
- Returns threshold configuration dictionary
- Used by `train.py` to generate `models/threshold_config.json`

#### 5. `src/fd1_nb/` - Notebook 1 Utilities (EDA & Feature Engineering)
General-purpose utility functions extracted from notebook 1 for reusability.

**`src/fd1_nb/data_utils.py`**:
- `load_data()`: Efficient pandas CSV loading
- `split_train_val_test()`: Stratified train/val/test splitting
- `analyze_target_stats()`: Target distribution and imbalance detection
- `analyze_feature_stats()`: Feature summary statistics

**`src/fd1_nb/eda_utils.py`**:
- `calculate_vif()`, `analyze_vif()`: Variance Inflation Factor analysis
- `calculate_numeric_correlations()`, `analyze_correlations()`: Correlation analysis
- `calculate_mi_scores()`, `analyze_mutual_information()`: Mutual information for categorical features
- `plot_numeric_distributions()`, `plot_box_plots()`: Distribution visualizations
- `analyze_temporal_patterns()`: Time-based fraud pattern analysis

**`src/fd1_nb/feature_engineering.py`**:
- `convert_utc_to_local_time()`: Timezone conversion with validation
- `create_temporal_features()`: Hour, day_of_week, is_late_night, etc.
- `create_interaction_features()`: Fraud scenario-specific combinations
- `create_percentile_based_features()`: Threshold-based binary flags

**Design Philosophy**: General-purpose, configurable functions that work with any dataset when given proper parameters. Includes verbose output and visualizations for exploratory analysis.

#### 6. `src/fd2_nb/` - Notebook 2 Utilities (Model Selection & Tuning)
Utility functions extracted from notebook 2 for model comparison, hyperparameter tuning, and CV analysis.

**`src/fd2_nb/model_comparison.py`**:
- `compare_models()`: Side-by-side model performance comparison
- `get_best_model()`: Select best model by specified metric
- `plot_comprehensive_comparison()`: Full visualization of model metrics
- `plot_model_comparison()`: Simple bar chart comparison

**`src/fd2_nb/hyperparameter_tuning.py`**:
- `create_search_object()`: Create GridSearchCV or RandomizedSearchCV
- `tune_with_logging()`: Execute search with detailed logging to files
- `extract_cv_metrics()`: Extract metrics from CV results
- `get_best_params_summary()`: Format best parameters for display

**`src/fd2_nb/cv_analysis.py`**:
- `analyze_cv_results()`: Production-focused CV results analysis
- `analyze_cv_train_val_gap()`: Detect overfitting via train-val gap analysis
- `analyze_iteration_performance()`: Analyze XGBoost iteration curves
- `get_cv_statistics()`: Extract statistics from CV results

**`src/fd2_nb/bias_variance.py`**:
- `analyze_cv_fold_variance()`: Analyze variance across CV folds

#### 7. `src/fd3_nb/` - Notebook 3 Utilities (Evaluation & Deployment)
Utility functions extracted from notebook 3 for model evaluation, threshold optimization, and deployment.

**`src/fd3_nb/evaluation.py`**:
- `evaluate_model()`: Comprehensive model evaluation with all metrics
- `compare_val_test_performance()`: Compare validation vs test performance

**`src/fd3_nb/visualization.py`**:
- `plot_roc_pr_curves()`: ROC and PR curve visualization
- `plot_feature_importance()`: XGBoost gain-based feature importance bar chart
- `plot_shap_importance()`: SHAP-based feature importance bar chart
- `plot_shap_beeswarm()`: SHAP beeswarm plot showing value distributions (handles categorical features)
- `plot_importance_comparison()`: Side-by-side comparison of gain vs SHAP importance
- `plot_threshold_optimization()`: Threshold vs metrics visualization

**`src/fd3_nb/threshold_optimization.py`**:
- `find_threshold_for_recall()`: Find threshold for target recall
- `find_optimal_f1_threshold()`: Find threshold maximizing F1
- `find_target_performance_threshold()`: Find threshold meeting precision constraint
- `optimize_thresholds()`: Complete threshold optimization pipeline
- `create_threshold_comparison_df()`: Create comparison DataFrame

**`src/fd3_nb/feature_importance.py`**:
- `extract_feature_importance()`: Extract gain-based importance from XGBoost pipeline
- `print_feature_importance_summary()`: Print formatted importance summary
- `compute_shap_importance()`: Compute SHAP values and importance using XGBoost native method
- `compare_importance_methods()`: Compare XGBoost gain vs SHAP importance rankings
- `print_shap_importance_summary()`: Print formatted SHAP importance with direction indicators
- `print_importance_comparison()`: Print Spearman correlation between methods

**`src/fd3_nb/deployment.py`**:
- `save_production_model()`: Save trained model with joblib
- `save_threshold_config()`: Save threshold configuration JSON
- `save_model_metadata()`: Save model metadata JSON
- `print_deployment_summary()`: Print deployment summary

### Benefits of Shared Infrastructure
✅ **Single Source of Truth**: Configuration, data loading, evaluation logic defined once
✅ **Code Reduction**: Eliminated ~240 lines of duplicated code across scripts
✅ **Consistency**: All scripts use same random seed, CV strategy, evaluation metrics
✅ **Maintainability**: Changes to common logic only need to be made in one place
✅ **Testability**: Shared modules can be unit tested independently
✅ **Extensibility**: New analysis scripts can easily reuse existing infrastructure

### Usage Example

**From Production Scripts (train.py, predict.py, analysis/bias_variance_analysis.py):**
```python
# Import deployment modules
from src.deployment.config import DataConfig, FeatureListsConfig, ModelConfig, TrainingConfig
from src.deployment.data import load_and_split_data
from src.deployment.preprocessing import FraudFeatureTransformer, PreprocessingPipelineFactory
from src.deployment.evaluation import evaluate_model, optimize_thresholds

# Load data with consistent splitting
train_df, val_df, test_df = load_and_split_data(random_seed=1)

# Load feature configuration
feature_config = FeatureListsConfig.load()
categorical = feature_config['categorical']
continuous_numeric = feature_config['continuous_numeric']
binary = feature_config['binary']

# Load hyperparameters
params = ModelConfig.load_hyperparameters('xgboost', source='metadata')

# Create preprocessing pipeline
preprocessor = PreprocessingPipelineFactory.create_tree_pipeline(
    categorical, continuous_numeric, binary
)

# Get CV strategy
cv = TrainingConfig.get_cv_strategy(random_seed=1)

# Evaluate model
metrics = evaluate_model(model, X_test, y_test, "XGBoost", "Test")

# Optimize thresholds
threshold_config = optimize_thresholds(model, X_val, y_val)
```

**From Notebooks (fd1_EDA_FE.ipynb):**
```python
# Import notebook-specific utilities
from src.fd1_nb.data_utils import load_data, split_train_val_test, analyze_target_stats
from src.fd1_nb.eda_utils import analyze_vif, analyze_correlations, analyze_mutual_information
from src.fd1_nb.feature_engineering import convert_utc_to_local_time, create_temporal_features

# Load and split data
df = load_data(data_dir, csv_file)
train_df, val_df, test_df = split_train_val_test(df, target_col='is_fraud')

# Run EDA analyses (single function calls with visualizations)
analyze_target_stats(train_df, target_col='is_fraud')
analyze_vif(train_df, numeric_features)
analyze_correlations(train_df, numeric_features, target_col='is_fraud')
```

## Production Feature Engineering Pipeline

### Overview
The production feature engineering pipeline is implemented in `src/deployment/preprocessing/` as a scikit-learn compatible transformer. This architecture enables seamless integration with sklearn pipelines and consistent feature engineering between training and inference.

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

#### 1. `src/deployment/preprocessing/config.py` - Configuration Management
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
from src.deployment.preprocessing import FeatureConfig

# During training (in EDA notebook)
config = FeatureConfig.from_training_data(train_df)
config.save("models/feature_config.json")

# During inference (in API)
config = FeatureConfig.load("models/feature_config.json")
```

#### 2. `src/deployment/preprocessing/features.py` - Feature Engineering Functions
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

#### 3. `src/deployment/preprocessing/transformer.py` - Sklearn Transformer
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
from src.deployment.preprocessing import FraudFeatureTransformer

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

**Test Coverage**: Comprehensive unit and integration tests for all components (425 passing tests)

**Test Organization**: Tests mirror source code structure for easy navigation

**Shared Fixtures** (`tests/conftest.py`):
- `sample_raw_df()`: Small raw DataFrame (10 rows) for preprocessing tests
- `sample_raw_df_utc()`: Same data with UTC timestamps
- `sample_config()`: Pre-configured FeatureConfig
- `fitted_transformer()`: Fitted FraudFeatureTransformer
- `sample_engineered_df()`: Transformed data (30 features)

**Test Modules**:

1. **Configuration Tests** (`tests/test_config/` - 44 tests):
   - `test_data_config.py` (16 tests): DataConfig constants, get_data_path(), get_random_seed(), get_split_config()
   - `test_model_config.py` (19 tests): FeatureListsConfig.load(), ModelConfig.get_param_grid(), hyperparameter loading
   - `test_training_config.py` (9 tests): CV strategy (StratifiedKFold), threshold targets, random seed handling

2. **Data Loading Tests** (`tests/test_data/` - 12 tests):
   - `test_loader.py`: load_and_split_data() with correct ratios, stratification, no data leakage, custom parameters

3. **EDA Utility Tests** (`tests/test_eda/` - 68 tests):
   - `test_data_utils.py` (21 tests): load_data(), download_data_csv(), split_train_val_test(), analyze_target_stats()
   - `test_eda_utils.py` (27 tests): VIF, correlations, mutual information, temporal patterns, categorical fraud rates
   - `test_feature_engineering.py` (20 tests): Timezone conversion, temporal features, interaction features

4. **Evaluation Tests** (`tests/test_evaluation/` - 26 tests):
   - `test_metrics.py` (14 tests): calculate_metrics(), evaluate_model(), perfect predictions, verbose control
   - `test_thresholds.py` (12 tests): optimize_thresholds(), default/custom targets, threshold config structure

5. **Preprocessing Tests** (`tests/test_preprocessing/` - 61 tests):
   - `test_config.py` (8 tests): FeatureConfig creation, save/load, JSON structure, quantile calculation
   - `test_features.py` (15 tests): Individual feature functions, edge cases, timezone validation
   - `test_pipelines.py` (18 tests): PreprocessingPipelineFactory, tree/logistic pipelines, model type aliases
   - `test_transformer.py` (20 tests): Full pipeline execution, sklearn compatibility, save/load consistency

6. **Notebook 2 Utility Tests** (`tests/test_fd2_nb/` - 63 tests):
   - `test_model_comparison.py` (16 tests): compare_models(), get_best_model(), plot functions
   - `test_hyperparameter_tuning.py` (15 tests): create_search_object(), tune_with_logging()
   - `test_cv_analysis.py` (25 tests): analyze_cv_results(), analyze_cv_train_val_gap(), analyze_iteration_performance()
   - `test_bias_variance.py` (7 tests): analyze_cv_fold_variance()

7. **Notebook 3 Utility Tests** (`tests/test_fd3_nb/` - 83 tests):
   - `test_evaluation.py` (15 tests): evaluate_model(), compare_val_test_performance()
   - `test_feature_importance.py` (19 tests): extract_feature_importance(), compute_shap_importance(), compare_importance_methods()
   - `test_threshold_optimization.py` (23 tests): find_threshold_for_recall(), optimize_thresholds(), comparison DataFrame
   - `test_deployment.py` (26 tests): save_production_model(), save_threshold_config(), save_model_metadata()
   - `test_visualization.py` (7 tests): plot_shap_beeswarm() with numeric, categorical, and edge cases

8. **Explainability Tests** (`tests/test_explainability/` - 9 tests):
   - `test_shap_explainer.py`: FraudExplainer initialization, explain() output structure, top_n parameter

9. **API Tests** (`tests/test_api.py` - 33 tests):
   - Endpoint testing (root, health, model info, predict)
   - Request/response validation
   - Error handling scenarios
   - Threshold strategies
   - Explainability feature tests (include_explanation, top_n parameters)
   - Performance validation

**Run Tests**:
```bash
# Run all tests
uv run pytest tests/ -v

# Run by component
uv run pytest tests/test_config/ -v        # Configuration tests
uv run pytest tests/test_data/ -v          # Data loading tests
uv run pytest tests/test_eda/ -v           # EDA utility tests
uv run pytest tests/test_evaluation/ -v    # Evaluation tests
uv run pytest tests/test_preprocessing/ -v # Preprocessing tests
uv run pytest tests/test_fd2_nb/ -v        # Notebook 2 utility tests
uv run pytest tests/test_fd3_nb/ -v        # Notebook 3 utility tests
uv run pytest tests/test_api.py -v         # API tests

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_config/test_model_config.py -v
```

**Test Results**: All 425 tests passing with 100% success rate

### Notebook Integration

The EDA notebook (`notebooks/fd1_EDA_FE.ipynb`) now includes a cell that automatically generates and saves the FeatureConfig:

```python
# Create and save feature configuration for deployment
from src.deployment.preprocessing import FeatureConfig

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

## Production API Deployment

### Overview
The fraud detection model is deployed as a FastAPI web service (`predict.py`) that provides real-time fraud predictions via REST API. The service integrates the production feature engineering pipeline (`FraudFeatureTransformer`) with the trained XGBoost model.

### Architecture

**Key Components**:
- **FastAPI Application**: Async web framework with automatic API documentation
- **Lifespan Context Manager**: Loads model artifacts on startup (no per-request overhead)
- **Feature Transformer**: Applies production feature engineering pipeline
- **XGBoost Model**: Tuned fraud detection model (PR-AUC 0.8679)
- **Threshold Strategies**: Configurable precision-recall trade-offs

### API Endpoints

#### 1. `POST /predict` - Fraud Prediction
**Purpose**: Predict fraud for a raw transaction with automatic feature engineering

**Input**: `RawTransactionRequest` (15 raw features)
- User features: `user_id`, `account_age_days`, `total_transactions_user`, `avg_amount_user`
- Transaction: `amount`, `country`, `bin_country`, `channel`, `merchant_category`
- Security: `promo_used`, `avs_match`, `cvv_result`, `three_ds_flag`
- Geographic/Temporal: `shipping_distance_km`, `transaction_time`

**Parameters**:
- `threshold_strategy` (query param): Choose precision-recall trade-off
  - `conservative_90pct_recall`: Catches 90% of fraud (more false positives)
  - `balanced_85pct_recall`: Balanced approach (default)
  - `aggressive_80pct_recall`: Fewer false positives (may miss some fraud)
  - `optimal_f1`: Maximum F1 score threshold
- `include_explanation` (query param, bool): Include SHAP-based feature explanations (default: false)
- `top_n` (query param, int): Number of top contributing features to include in explanation (1-10, default: 3)

**Output**: `PredictionResponse`
```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_fraud": false,
  "fraud_probability": 0.12,
  "risk_level": "low",
  "threshold_used": "balanced_85pct_recall",
  "threshold_value": 0.35,
  "model_version": "1.0",
  "processing_time_ms": 15.3,
  "explanation": null
}
```

**Processing Pipeline**:
1. Validate raw transaction data (Pydantic validation)
2. Convert to pandas DataFrame
3. Apply `FraudFeatureTransformer` (15 raw → 30 engineered features)
4. XGBoost prediction (fraud probability)
5. Apply threshold strategy (fraud classification)
6. Calculate risk level (low/medium/high)
7. Return prediction with metadata

#### 2. `GET /health` - Health Check
**Purpose**: Monitor service status and uptime

**Output**: `HealthResponse`
- Service status (healthy/unhealthy)
- Model loaded status
- Model version
- Uptime in seconds
- Current timestamp

#### 3. `GET /model/info` - Model Information
**Purpose**: Get model metadata and configuration

**Output**: `ModelInfoResponse`
- Model name, version, training date
- Algorithm details (XGBoost)
- Performance metrics (test set)
- Available threshold strategies
- Raw features required (15)
- Engineered features count (30)

#### 4. `GET /` - API Root
**Purpose**: API documentation links and endpoint overview

#### 5. `GET /docs` - Interactive API Documentation
**Purpose**: Auto-generated Swagger UI for testing endpoints

#### 6. `GET /redoc` - Alternative Documentation
**Purpose**: ReDoc-style API documentation

### Model Artifacts

The API loads the following artifacts on startup from `models/` directory:

1. **`xgb_fraud_detector.joblib`** (gitignored)
   - Trained XGBoost model (best hyperparameters)
   - Loaded via joblib

2. **`transformer_config.json`** (tracked in git)
   - FeatureConfig for FraudFeatureTransformer
   - Quantile thresholds from training data
   - Timezone mapping for local time conversion

3. **`threshold_config.json`** (tracked in git)
   - Three threshold strategies with precision-recall trade-offs
   - Threshold values calibrated on validation set

4. **`model_metadata.json`** (tracked in git)
   - Model version, training date, algorithm
   - Performance metrics (PR-AUC, ROC-AUC, F1, Precision, Recall)
   - Dataset information

5. **`feature_lists.json`** (tracked in git, created in fd1 notebook)
   - Feature categorization by dtype for preprocessing pipeline
   - Categories: categorical (1), continuous_numeric (12), binary (17)
   - Used by fd2 and fd3 notebooks for consistent feature handling

### Usage

#### Start the API Server
```bash
# Development mode (with auto-reload)
uvicorn predict:app --host 0.0.0.0 --port 8000 --reload

# Production mode (without auto-reload for better performance)
uvicorn predict:app --host 0.0.0.0 --port 8000
```

**Note**: The `--reload` flag enables automatic reloading when files change, but can cause elevated CPU usage during idle time due to file watching. Use without `--reload` in production or when performance is critical.

#### Make a Prediction
```bash
curl -X POST "http://localhost:8000/predict?threshold_strategy=balanced_85pct_recall" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "account_age_days": 180,
    "total_transactions_user": 25,
    "avg_amount_user": 250.50,
    "amount": 850.75,
    "country": "US",
    "bin_country": "US",
    "channel": "web",
    "merchant_category": "retail",
    "promo_used": 0,
    "avs_match": 1,
    "cvv_result": 1,
    "three_ds_flag": 1,
    "shipping_distance_km": 12.5,
    "transaction_time": "2024-01-15 14:30:00"
  }'
```

#### Check Health
```bash
curl http://localhost:8000/health
```

#### Get Model Info
```bash
curl http://localhost:8000/model/info
```

### Explainability Feature

The API includes optional SHAP-based explainability to show which features contributed most to a fraud prediction.

**Request with Explanation**:
```bash
curl -X POST "http://localhost:8000/predict?include_explanation=true&top_n=3" \
  -H "Content-Type: application/json" \
  -d '{ ... transaction data ... }'
```

**Response with Explanation**:
```json
{
  "transaction_id": "...",
  "is_fraud": true,
  "fraud_probability": 0.85,
  "explanation": {
    "top_contributors": [
      {
        "feature": "avs_match",
        "display_name": "Address Verification Match",
        "value": 0,
        "contribution": 0.32
      },
      {
        "feature": "account_age_days",
        "display_name": "Account Age (days)",
        "value": 5,
        "contribution": 0.28
      },
      {
        "feature": "security_score",
        "display_name": "Security Score",
        "value": 1,
        "contribution": 0.15
      }
    ],
    "base_fraud_rate": 0.022,
    "explanation_method": "shap"
  }
}
```

**Key Points**:
- Uses XGBoost's native `pred_contribs=True` for fast SHAP computation (~1-5ms overhead)
- Only returns features that **increase** fraud risk (positive SHAP contributions)
- `contribution` values are per-sample SHAP values (direction is always correct for individual predictions)
- `base_fraud_rate` is the expected value (baseline) before feature contributions
- Human-readable `display_name` provided for each feature

### Key Features

1. **Automatic Feature Engineering**: API accepts raw transaction data (15 features) and automatically applies the production feature engineering pipeline to create 30 engineered features

2. **Input Validation**: Pydantic models ensure data quality with type checking, range validation, and pattern matching

3. **Flexible Thresholds**: Four pre-configured threshold strategies allow customization of precision-recall trade-offs without retraining

4. **SHAP-based Explainability**: Optional per-prediction explanations showing top risk-increasing features

5. **Fast Inference**: ~15-20ms average processing time (feature engineering + prediction + optional explanation)

6. **Error Handling**: Comprehensive error handling with structured error responses

7. **Logging**: Request/response logging for monitoring and debugging

8. **Auto-Documentation**: Swagger UI and ReDoc for interactive API exploration

### Design Decisions

1. **Raw Input Features**: API accepts raw transaction data (not engineered features) to simplify client integration and ensure consistent feature engineering

2. **Lifespan Context Manager**: Replaced deprecated `@app.on_event` with modern `@asynccontextmanager` pattern for startup/shutdown events

3. **Threshold Strategies**: Pre-calibrated thresholds stored in JSON allow flexibility without model retraining

4. **JSON Configuration**: All configuration stored as JSON (not pickle) for version control and cross-language compatibility

5. **Pydantic Validation**: Strong typing and validation prevent invalid inputs from reaching the model

## Docker Containerization

### Overview
The API is containerized using Docker for consistent deployment across environments. The implementation uses a multi-stage build for optimized image size and includes security best practices.

### Architecture

**Multi-Stage Build**:
- **Stage 1 (Builder)**: Installs build dependencies and Python packages
- **Stage 2 (Runtime)**: Minimal production image with only runtime dependencies

**Security Features**:
- Non-root user (appuser, UID 1000) for container execution
- Read-only model volume mounting in docker-compose
- Health checks for container orchestration
- Minimal base image (python:3.12-slim)

### Files

#### 1. `Dockerfile` - Container Image Definition

**Key Features**:
- **Base Image**: `python:3.12-slim` (minimal footprint)
- **Multi-stage Build**: Separates build-time and runtime dependencies
- **Non-root User**: Runs as `appuser` for security
- **Optimizations**:
  - `--no-cache-dir` for pip to reduce image size
  - Only copies necessary files (predict.py, train.py, src/, models/)
  - Uses COPY --from=builder to transfer installed packages
- **Health Check**: Python-based health check using requests library
- **Port**: Exposes 8000 for API access

**Build Stages**:
```dockerfile
# Stage 1: Install dependencies
FROM python:3.12-slim AS builder
# ... install build-essential, pip packages

# Stage 2: Runtime image
FROM python:3.12-slim
# ... copy packages, application code, run as non-root
```

**Artifacts Copied**:
- Application: `predict.py`, `train.py`, `src/`
- Model artifacts: `models/*.json` (configs), `models/*.joblib` (trained model)

**Important**: Model files must exist before building. Run `train.py` locally first to generate model artifacts.

#### 2. `docker-compose.yml` - Orchestration Configuration

**Service Definition**:
- **Service Name**: `fraud-api`
- **Container Name**: `fraud-detection-api`
- **Port Mapping**: 8000:8000 (host:container)
- **Network**: Custom bridge network (`fraud-detection-network`)
- **Restart Policy**: `unless-stopped` (auto-restart on failure)

**Environment Variables**:
- `PYTHONUNBUFFERED=1` - Unbuffered Python output for real-time logs
- `PORT=8000` - API port configuration

**Volume Mounts** (Development):
- `./predict.py:/app/predict.py` - Hot reload for API changes
- `./src:/app/src` - Hot reload for source code changes
- `./models:/app/models:ro` - Read-only model mounting for easy updates

**Note**: Comment out source code volumes for production deployment.

**Health Check**:
- Test command: `curl -f http://localhost:8000/health`
- Interval: 30s
- Timeout: 10s
- Retries: 3
- Start period: 5s

### Usage

#### Building the Image

```bash
# Build Docker image
docker build -t fraud-detection-api .

# Check image size
docker images fraud-detection-api

# Expected size: ~400-500MB (optimized multi-stage build)
```

#### Running with Docker

```bash
# Run container directly
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  fraud-detection-api

# View logs
docker logs fraud-api

# Follow logs in real-time
docker logs -f fraud-api

# Stop container
docker stop fraud-api

# Remove container
docker rm fraud-api
```

#### Running with Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# View service status
docker-compose ps
```

#### Verification

```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "account_age_days": 180,
    "total_transactions_user": 25,
    "avg_amount_user": 250.50,
    "amount": 850.75,
    "country": "US",
    "bin_country": "US",
    "channel": "web",
    "merchant_category": "retail",
    "promo_used": 0,
    "avs_match": 1,
    "cvv_result": 1,
    "three_ds_flag": 1,
    "shipping_distance_km": 12.5,
    "transaction_time": "2024-01-15 14:30:00"
  }'

# Access interactive docs
open http://localhost:8000/docs
```

### Development vs Production

**Development Mode** (current docker-compose.yml):
- Source code volumes mounted for hot reload
- Models mounted read-only for easy updates
- Suitable for local testing and development

**Production Mode**:
```yaml
# Comment out these lines in docker-compose.yml
# volumes:
#   - ./predict.py:/app/predict.py
#   - ./src:/app/src
#   - ./models:/app/models:ro

# Or build image with code baked in (no volumes)
docker build -t fraud-detection-api:prod .
docker run -d -p 8000:8000 fraud-detection-api:prod
```

### Image Optimization

**Size Reduction Techniques**:
1. **Multi-stage build**: ~40% size reduction vs single-stage
2. **Slim base image**: python:3.12-slim vs python:3.12 (~600MB savings)
3. **No cache pip installs**: `--no-cache-dir` flag
4. **Minimal dependencies**: Only production packages in requirements.txt
5. **Cleanup**: Remove apt lists after package installation

**Expected Image Size**: ~400-500MB (includes Python runtime, FastAPI, XGBoost, scikit-learn, pandas)

### Security Best Practices

✅ **Non-root User**: Container runs as `appuser` (UID 1000), not root
✅ **Read-only Models**: Model volume mounted as read-only (`:ro`)
✅ **Minimal Base**: Slim Python image reduces attack surface
✅ **Health Checks**: Container orchestration can detect failures
✅ **No Secrets in Image**: No credentials or sensitive data baked in
✅ **Explicit COPY**: Only necessary files copied to image

### Troubleshooting

**Issue: Model files not found**
```bash
# Solution: Train model first
uv run python train.py --skip-tuning

# Verify artifacts exist
ls -lh models/*.json models/*.joblib
```

**Issue: Container health check failing**
```bash
# Check container logs
docker logs fraud-api

# Exec into container
docker exec -it fraud-api bash

# Test health endpoint manually
curl http://localhost:8000/health
```

**Issue: Port already in use**
```bash
# Find process using port 8000
lsof -i :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Map to different host port
```

**Issue: Image too large**
```bash
# Analyze image layers
docker history fraud-detection-api

# Check for unnecessary files
docker run --rm -it fraud-detection-api ls -lhR /app
```

## Performance Benchmarking

### Overview

Comprehensive performance testing suite to measure API latency, throughput, and scalability. The benchmarking infrastructure includes:
- **`benchmarks/benchmark.py`**: Python script for automated performance testing
- **`benchmarks/locustfile.py`**: Load testing configuration for distributed testing

### Quick Benchmarking (benchmarks/benchmark.py)

**Purpose**: Measure single-request latency and basic concurrent performance

**Features**:
- Cold start latency measurement (first request after startup)
- Sequential request performance (P50, P95, P99 percentiles)
- Concurrent user simulation with thread pool
- Server-side vs end-to-end timing (network overhead)
- JSON report export for trend analysis

**Usage**:
```bash
# Default benchmark (100 requests, 10 concurrent users)
uv run python benchmarks/benchmark.py --url http://localhost:8000

# Custom configuration
uv run python benchmarks/benchmark.py \
  --url http://localhost:8000 \
  --iterations 500 \
  --concurrent 20 \
  --output benchmarks/results/benchmark_$(date +%Y%m%d_%H%M%S).json
```

**Output Example**:
```
================================================================================
FRAUD DETECTION API - PERFORMANCE BENCHMARK REPORT
================================================================================

📊 Test Configuration:
  Timestamp: 2025-11-15T16:22:46.343758
  Base URL: http://localhost:8000
  Iterations: 100
  Concurrent Users: 10

🏥 Health Check:
  Status: healthy
  Model Loaded: True
  Model Version: 1.0

❄️  Cold Start Performance:
  Server Processing: 25.93 ms
  End-to-End: 28.45 ms
  Network Overhead: 2.52 ms

🔥 Single Request Performance (100/100 successful):
  Server Processing Time:
    Mean:   29.21 ms
    Median: 27.29 ms
    P95:    45.64 ms
    P99:    74.22 ms

⚡ Concurrent Request Performance (10 users):
  Throughput: 34.08 requests/second
  Success Rate: 100.0%
```

### Load Testing (benchmarks/locustfile.py)

**Purpose**: Simulate realistic user behavior and identify breaking points

**User Scenarios**:
1. **`FraudDetectionUser`** (Production simulation):
   - 70% normal transactions (established accounts, typical amounts)
   - 30% suspicious transactions (new accounts, high amounts)
   - Realistic wait times (100-500ms between requests)
   - Task weights: 10 (normal) : 3 (suspicious) : 1 (health checks) : 1 (model info)

2. **`StressTestUser`** (Maximum throughput):
   - Rapid-fire requests with minimal wait time (0-100ms)
   - Identifies API breaking points and resource limits

**Usage**:
```bash
# Interactive web UI (recommended for testing)
locust -f benchmarks/locustfile.py --host=http://localhost:8000
# Open browser to http://localhost:8089

# Headless mode (CI/CD integration)
locust -f benchmarks/locustfile.py \
  --host=http://localhost:8000 \
  --users 50 \
  --spawn-rate 10 \
  --run-time 60s \
  --headless

# Generate HTML report
locust -f benchmarks/locustfile.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 120s \
  --headless \
  --html=benchmarks/results/locust_$(date +%Y%m%d_%H%M%S).html
```

**Key Metrics**:
- **RPS (Requests Per Second)**: Sustained throughput under load
- **Response Time Distribution**: P50, P90, P95, P99 percentiles
- **Failure Rate**: Percentage of failed requests
- **Concurrent Users**: Maximum users before degradation

### Benchmark Results (Baseline)

**Environment**:
- **Platform**: Linux (Ubuntu 24.04)
- **CPU**: x86_64
- **Python**: 3.12
- **Deployment**: Local (uvicorn, single worker)
- **Date**: 2025-11-15

**Single Request Performance (100 iterations)**:

| Metric | Mean | Median | P95 | P99 | Min | Max |
|--------|------|--------|-----|-----|-----|-----|
| **Server Processing** | 29.21 ms | 27.29 ms | 45.64 ms | 74.22 ms | 17.42 ms | 74.22 ms |
| **End-to-End Latency** | 31.61 ms | 29.65 ms | 48.84 ms | 76.29 ms | - | - |
| **Network Overhead** | 2.40 ms | 2.23 ms | - | - | - | - |

**Concurrent Load Performance (10 users, 100 requests)**:

| Metric | Value |
|--------|-------|
| **Throughput** | 34.08 requests/second |
| **Success Rate** | 100% |
| **Total Time** | 2.93 seconds |
| **Server P95** | 43.15 ms |
| **Server P99** | 47.35 ms |
| **E2E P95** | 358.08 ms |
| **E2E P99** | 383.48 ms |

**Cold Start Performance**:

| Metric | Latency |
|--------|---------|
| **Server Processing** | 25.93 ms |
| **End-to-End** | 28.45 ms |
| **Network Overhead** | 2.52 ms |

### Performance Analysis

**✅ Excellent Results**:
- **Sub-50ms P95**: Server processing consistently under 50ms (target: <100ms)
- **Low Variance**: P50→P95 delta of only 18ms (27ms→46ms) indicates stable performance
- **Fast Cold Start**: <30ms first request (no warm-up penalty)
- **Perfect Reliability**: 100% success rate under concurrent load
- **High Throughput**: 34 RPS on single instance (extrapolates to 120k+ requests/hour)

**Key Observations**:
1. **Network Overhead**: Average 2.4ms indicates local deployment. Production will add 10-50ms depending on geographic distance.
2. **Concurrent Queueing**: E2E latency increases to 289ms (P50) under concurrent load due to request queueing, but server processing remains stable (26ms).
3. **XGBoost Efficiency**: Feature engineering + prediction completes in <30ms on average.
4. **No Memory Leaks**: Consistent performance across 100 requests indicates stable memory usage.

### Performance Targets vs Achieved

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **Server P95 Latency** | < 50ms | 45.64 ms | ✅ Pass |
| **Server P99 Latency** | < 100ms | 74.22 ms | ✅ Pass |
| **Cold Start** | < 100ms | 25.93 ms | ✅ Pass |
| **Throughput** | > 20 RPS | 34.08 RPS | ✅ Pass |
| **Success Rate** | 100% | 100% | ✅ Pass |

**All performance targets met or exceeded.**

### Production Scaling Recommendations

**Current Capacity (Single Instance)**:
- **Sustained Load**: ~30 RPS (108k requests/hour)
- **Peak Load**: ~50 RPS for short bursts

**Horizontal Scaling** (Multiple Instances):
- **Load Balancer**: Nginx/HAProxy with round-robin or least-connections
- **Auto-scaling**: Scale based on CPU (>70%) or latency (P95 >75ms)
- **Target**: 3-5 instances for 100k+ daily transactions

**Vertical Scaling** (Resource Optimization):
- **Multi-worker Uvicorn**: `--workers 4` (CPU cores)
- **Expected**: 2-3x throughput improvement (60-100 RPS per instance)

**Monitoring Alerts**:
- **P95 latency > 75ms**: Trigger scale-up
- **Success rate < 99.5%**: Page on-call engineer
- **Throughput drop > 25%**: Check for resource constraints

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
uv run --with jupyter jupyter lab
```

### Data Download
The notebook automatically downloads the dataset from Kaggle on first run if not present in `./data/`. The download function checks for existing data to avoid redundant downloads.

## Notebook Structure

The project uses a modular three-notebook workflow with clear separation of concerns:

### Notebook 1: fd1_EDA_FE.ipynb (EDA & Feature Engineering)

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

#### 7. Production Configuration Export
- Generate and save `FraudFeatureTransformer` configuration for deployment
- Create `transformer_config.json` with quantile thresholds from training data
- Store 30 selected feature names with categorical groupings
- Include timezone mappings for consistent local time conversion
- Ensure reproducible feature engineering between training and inference

### Notebook 2: fd2_model_selection_tuning.ipynb (Model Selection & Hyperparameter Tuning)

#### 1. Setup
- Parameter definitions (data paths, random seed, model directory)
- Package imports (sklearn, xgboost, preprocessing, metrics)

#### 2. Data Loading
- Load raw transaction data from CSV
- Apply `FraudFeatureTransformer` pipeline to generate engineered features
- Transform raw data consistently across train/val/test splits using same configuration
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
  - Performance: PR-AUC 0.6973, Precision 20.50%, Recall 89.34%
  - Weakness: Very high false positive rate (low precision)

- **Random Forest**: `class_weight='balanced'`, n_estimators=100, random_state=42
  - Performance: PR-AUC 0.8482, Precision 93.84%, Recall 71.43%
  - Strength: Excellent precision, low false positives

- **XGBoost**: `scale_pos_weight=44.3` (class imbalance ratio), n_estimators=100, random_state=42
  - Performance: PR-AUC 0.8458, Precision 55.42%, Recall 84.66%
  - Issue: Recall-dominated, high false positives, moderate precision

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
- **Best CV Parameters**: n_estimators=500, max_depth=25, min_samples_leaf=2, class_weight='balanced_subsample'
- **Best CV PR-AUC**: 0.8587
- **Validation Set Performance**: PR-AUC 0.8579, Precision 89.16%, Recall 76.49%, F1 0.8234
- **Improvement**: +1.2% PR-AUC vs baseline
- **Trade-off**: Sacrificed 5.0% precision to gain 7.1% recall (good for fraud detection)

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
- **Best CV Parameters**: n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, min_child_weight=5, gamma=0.6, scale_pos_weight=8
- **Best CV PR-AUC**: 0.8675
- **Validation Set Performance**: PR-AUC 0.8672, Precision 72.87%, Recall 83.22%, F1 0.7770
- **Major Win**: Precision +31.5% (55.42% → 72.87%), F1 +16.0%
- **False Positive Reduction**: 54% fewer false positives vs baseline XGBoost

**Key Insights**:
- **Critical Finding**: Making `scale_pos_weight` tunable (not fixed at class ratio of 44.3) was crucial
  - Optimal value: **8** (much lower than class ratio)
  - This allows the model to balance precision-recall more effectively
- Changing `eval_metric` to 'aucpr' aligned training objective with optimization goal (PR-AUC)
- **Shallow trees** (max_depth=4) with **high regularization** (gamma=0.6, min_child_weight=5) essential for precision
- Default learning_rate (0.1) performed best - lower values didn't improve results

#### 6. Model Evaluation
- **Primary Metrics**: PR-AUC (optimization target), ROC-AUC, F1 Score
- **Secondary Metrics**: Precision, Recall, Confusion Matrix
- **Evaluation Functions**:
  - `calculate_metrics(y_true, y_pred, y_pred_proba)`: Comprehensive metric calculation
  - `print_evaluation_metrics(metrics_dict, model_name, dataset_name)`: Formatted metric display
  - `plot_confusion_matrix(y_true, y_pred, model_name)`: Confusion matrix visualization

**Model Comparison (Validation Set)**:

| Model | PR-AUC | ROC-AUC | F1 | Precision | Recall |
|-------|--------|---------|-------|-----------|--------|
| Logistic Regression | 0.6973 | 0.9662 | 0.3334 | 20.50% | 89.34% |
| Random Forest (Baseline) | 0.8482 | 0.9627 | 0.8112 | 93.84% | 71.43% |
| Random Forest (Tuned) | 0.8579 | - | 0.8234 | 89.16% | 76.49% |
| XGBoost (Baseline) | 0.8458 | 0.9667 | 0.6699 | 55.42% | 84.66% |
| **XGBoost (Tuned)** | **0.8672** | **-** | **0.7770** | **72.87%** | **83.22%** |

**Performance Targets vs Achieved (XGBoost Tuned - Validation Set)**:
- ✅ PR-AUC > 0.85: **0.8672**
- ✅ ROC-AUC > 0.95: ✅ (see test set results)
- ✅ F1 > 0.75: **0.7770**
- ✅ Precision > 0.70: **0.7287**
- ✅ Recall > 0.80: **0.8322**

**Test Set Evaluation (Final Model Retrained on Train+Val)**:
- **Training Data**: 239,756 samples (train+val combined, +33.3% more data)
- **Test Set**: 59,939 samples (20% of original dataset)
- **Final Performance**:
  - **PR-AUC**: **0.8658** ✅ (target: >0.85)
  - **ROC-AUC**: **0.9766** ✅ (target: >0.95)
  - **F1 Score**: **0.7774** ✅ (target: >0.75)
  - **Precision**: **73.36%** ✅ (target: >70%)
  - **Recall**: **82.68%** ✅ (target: >80%)
- **Result**: **All performance targets exceeded on held-out test set** - excellent generalization!

#### 7. Feature Importance Analysis
**Top 10 Features by XGBoost Gain Metric**:

| Rank | Feature | Importance (Gain) | Percentage |
|------|---------|-------------------|------------|
| 1 | account_age_days | 0.2052 | 20.5% |
| 2 | avs_match | 0.1814 | 18.1% |
| 3 | security_score | 0.1275 | 12.8% |
| 4 | shipping_distance_km | 0.0893 | 8.9% |
| 5 | high_risk_distance | 0.0569 | 5.7% |
| 6 | amount | 0.0556 | 5.6% |
| 7 | transaction_velocity | 0.0503 | 5.0% |
| 8 | amount_deviation | 0.0439 | 4.4% |
| 9 | country_mismatch | 0.0393 | 3.9% |
| 10 | amount_vs_avg_ratio | 0.0388 | 3.9% |

**Key Insights**:
- **Account age** is the strongest predictor (20.5%) - newer accounts are higher risk
- **Security features** (avs_match, security_score) account for 30.9% of importance
- **Geographic features** (shipping_distance_km, high_risk_distance, country_mismatch) contribute 18.5%
- **User behavior** (transaction_velocity, amount patterns) adds 13.3%
- Top 10 features account for **85.1% of total importance**

#### 8. Threshold Optimization
**Three Pre-Calibrated Strategies** (calibrated on validation set):

| Strategy | Target Recall | Threshold | Precision | Recall | Use Case |
|----------|---------------|-----------|-----------|--------|----------|
| **Conservative (90%)** | 90% | 0.2624 | 45.54% | 90.02% | Minimize fraud escapes (high false positives acceptable) |
| **Balanced (85%)** | 85% | 0.4462 | 68.89% | 85.03% | Production default - good precision-recall balance |
| **Aggressive (80%)** | 80% | 0.7291 | 85.53% | 79.97% | Minimize false positives (may miss some fraud) |

**Implementation**:
- Thresholds saved to `models/threshold_config.json`
- API allows selection via `threshold_strategy` query parameter
- Default strategy: `balanced_85pct_recall`

#### 9. Final Model Selection
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

#### 8. Save Best Parameters
**Output artifacts for next notebook**:
1. **models/best_params.json** - Optimal hyperparameters from GridSearchCV
2. **models/validation_metrics.json** - Validation set performance metrics

**Handoff Mechanism**: These JSON files enable the next notebook to load the best model configuration without re-running hyperparameter tuning.

### Notebook 3: fd3_model_evaluation_deployment.ipynb (Final Evaluation & Deployment)

#### 1. Load Best Parameters
- Loads `models/best_params.json` from notebook 2
- Loads `models/validation_metrics.json` for comparison

#### 2. Data Preparation
- Recreates same train/val/test splits (same random seed)
- Applies FraudFeatureTransformer pipeline

#### 3. Retrain Best Model
- Creates model with optimal hyperparameters from notebook 2
- Trains on combined train+val data (239,756 samples, 33% more data)

#### 4. Test Set Evaluation
**Final model performance on held-out test set**:
- **PR-AUC**: 0.8655 ✅ (target: >0.85)
- **ROC-AUC**: 0.9768 ✅ (target: >0.95)
- **F1 Score**: 0.7754 ✅ (target: >0.75)
- **Precision**: 72.72% ✅ (target: >70%)
- **Recall**: 83.06% ✅ (target: >80%)

#### 5. Performance Visualization
- ROC and PR curves on test set
- Comparison with validation performance

#### 6. Feature Importance Analysis
- **SHAP-based importance** using XGBoost native `pred_contribs=True`
- SHAP beeswarm plot showing value distributions colored by feature values
- Comparison of XGBoost gain vs SHAP importance rankings
- **Important**: Global SHAP direction (mean_shap) can be misleading for continuous features with non-linear relationships
  - Example: `shipping_distance_km` shows ↓ fraud globally, but per-sample SHAP correctly shows: short distance → negative SHAP, long distance → positive SHAP
  - Always use beeswarm plots to understand true feature-SHAP relationships
  - Per-sample SHAP (used in API) always shows correct direction for individual predictions

#### 7. Threshold Optimization
- Three threshold strategies calibrated on validation set
- Precision-recall trade-offs for different use cases

#### 8. Model Deployment Preparation
**All deployment artifacts generated and saved to `models/` directory**:

1. **xgb_fraud_detector.joblib** (gitignored)
   - Final XGBoost model retrained on train+val (239,756 samples)
   - Includes full sklearn Pipeline with preprocessing steps

2. **transformer_config.json** (tracked in git)
   - FeatureConfig for FraudFeatureTransformer
   - Quantile thresholds calculated from training data
   - Timezone mapping for local time conversion

3. **threshold_config.json** (tracked in git)
   - Three pre-calibrated threshold strategies (conservative, balanced, aggressive)
   - Precision-recall trade-offs for different use cases

4. **model_metadata.json** (tracked in git)
   - Model version, training date, algorithm details
   - Test set performance metrics (PR-AUC, ROC-AUC, F1, Precision, Recall)
   - Hyperparameters used for training
   - Top 10 feature importance scores
   - Dataset information (sample counts, feature counts)

5. **feature_lists.json** (tracked in git, created in fd1 notebook)
   - Feature categorization by dtype for preprocessing pipeline
   - Categories: categorical (1), continuous_numeric (12), binary (17)
   - Used by fd2 and fd3 notebooks for consistent feature handling

**Status**: ✅ Model fully trained, evaluated, and ready for production deployment via FastAPI

### Notebook Workflow Summary

```
notebooks/fd1_EDA_FE.ipynb
    ↓ (saves models/transformer_config.json + models/feature_lists.json)
notebooks/fd2_model_selection_tuning.ipynb
    ↓ (loads feature_lists.json, saves models/xgb_fraud_detector.joblib + models/best_params.json + models/validation_metrics.json)
notebooks/fd3_model_evaluation_deployment.ipynb
    ↓ (loads feature_lists.json + xgb_fraud_detector.joblib, saves models/threshold_config.json + models/model_metadata.json)
Production Deployment
```

**Benefits of Split Workflow**:
- **Clear separation**: EDA → Selection → Deployment stages
- **Focused notebooks**: Each notebook manageable size with single purpose
- **Proper holdout**: Test set only evaluated in final notebook
- **Reproducibility**: JSON files enable independent notebook execution
- **Modularity**: Easy to re-run specific stages without full pipeline
- **Collaboration**: Different team members can work on different stages

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

# Local imports - notebook utilities
from src.fd1_nb.data_utils import load_data, split_train_val_test, analyze_target_stats
from src.fd1_nb.eda_utils import analyze_vif, analyze_correlations, analyze_mutual_information
from src.fd1_nb.feature_engineering import convert_utc_to_local_time, create_temporal_features

# Local imports - production modules (if needed)
from src.deployment.preprocessing import FeatureConfig, FraudFeatureTransformer
```

### Extract Functions to Source Modules

**Core Principle**: Move medium and long functions out of notebooks into dedicated source modules (`src/fd*_nb/`). This keeps notebooks focused on storytelling and analysis rather than implementation details.

#### Why Extract Functions?

**Notebooks should tell a story**, not be cluttered with function definitions. When a notebook contains many functions:
- The narrative flow is broken by large code blocks
- Functions can't be unit tested
- Code can't be reused across notebooks
- Version control diffs become noisy
- The notebook becomes harder to read and maintain

#### The Pattern

1. **Create a module** in `src/fd*_nb/` for each notebook's utilities
2. **Move functions** that are reusable or longer than ~10 lines
3. **Import and call** the functions in notebook cells
4. **Keep the notebook** focused on parameters, function calls, and analysis

#### Example Workflow

**Before (cluttered notebook):**
```python
# Cell 1: 50-line function definition
def analyze_vif(df, numeric_features, threshold=5.0, verbose=True):
    """Calculate VIF and visualize results."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # ... 45 more lines of code
    plt.show()

# Cell 2: Call the function
analyze_vif(train_df, numeric_features)
```

**After (clean notebook):**
```python
# In src/fd1_nb/eda_utils.py - the function lives here with full implementation

# In notebook - just import and call
from src.fd1_nb.eda_utils import analyze_vif

analyze_vif(train_df, numeric_features)
```

#### What to Extract

**Move to source modules:**
- Functions longer than ~10-15 lines
- Visualization functions with complex matplotlib/seaborn logic
- Data transformation functions used multiple times
- Analysis functions that could be unit tested
- Utility functions reusable across notebooks

**Keep in notebook:**
- Short helper functions (<10 lines) specific to one analysis
- One-off calculations that don't need reuse
- Simple wrappers around library functions

#### Module Organization

Each notebook should have a corresponding module in `src/`:

| Notebook | Source Module | Purpose |
|----------|---------------|---------|
| `notebooks/fd1_EDA_FE.ipynb` | `src/fd1_nb/` | EDA utilities, feature engineering |
| `notebooks/fd2_model_selection_tuning.ipynb` | `src/fd2_nb/` | Tuning utilities, comparison functions |
| `notebooks/fd3_model_evaluation_deployment.ipynb` | `src/fd3_nb/` | Evaluation, deployment prep |

#### Benefits

✅ **Readable notebooks**: Focus on the story, not implementation details
✅ **Testable code**: Functions in `src/` can be unit tested
✅ **Reusable utilities**: Functions available across notebooks and scripts
✅ **Clean diffs**: Changes to logic don't clutter notebook diffs
✅ **Better collaboration**: Easier to review and understand

### Keep Cells Clean with Function Calls

**Core Principle**: Notebook cells should contain minimal logic - ideally just a single function call. Complex logic should live in source modules (`src/fd*_nb/`), not inline in the notebook.

#### Cell Design Goals
1. **Readability**: Cells are easy to scan and understand at a glance
2. **Narrative focus**: The notebook tells a story with clear steps
3. **Minimal code**: Each cell has a clear, single purpose
4. **Self-documenting**: Function names describe what happens

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
# Cell with single function call (function defined in src/fd1_nb/)
from src.fd1_nb.eda_utils import plot_numeric_distributions

plot_numeric_distributions(train_df, numeric_features)
```

#### Function Naming Conventions
- **Action-based names**: `analyze_`, `plot_`, `calculate_`, `print_`
- **Descriptive**: Name should clearly indicate what the function does
- **Examples**:
  - `analyze_vif()` - Calculate and visualize VIF
  - `plot_categorical_fraud_rates()` - Visualize fraud rates by category
  - `analyze_temporal_patterns()` - Analyze time-based fraud patterns

#### Where Functions Should Live

| Function Type | Location | Example |
|--------------|----------|---------|
| **Long/reusable** (>10-15 lines) | `src/fd*_nb/` module | `analyze_vif()`, `create_temporal_features()` |
| **Short/one-off** (<10 lines) | Inline in notebook | Simple calculations, quick formatting |
| **Production/deployment** | `src/deployment/` | `FraudFeatureTransformer`, `evaluate_model()` |

### SHAP Interpretation Best Practices

#### Global vs Per-Sample SHAP

**Global SHAP** (mean across all samples) can be misleading for continuous features with non-linear relationships to the target:

- **Example**: `shipping_distance_km` - Mean SHAP direction shows "↓ fraud" (negative correlation with fraud globally)
- **Reality**: The correlation between `shipping_distance_km` and its SHAP values is **+0.87** (strong positive)
- **Explanation**: Most transactions have short distances (low fraud risk, negative SHAP), which dominates the mean. But long distances DO increase fraud risk (positive SHAP for individual samples).

**Per-Sample SHAP** (used in the API explainability feature) always shows the correct direction for individual predictions:
- Short shipping distance → Negative SHAP (decreases fraud probability)
- Long shipping distance → Positive SHAP (increases fraud probability)

#### Best Practices

1. **Always use beeswarm plots** to visualize the relationship between feature values and SHAP values
2. **Check correlations** between features and their SHAP values to understand true relationships
3. **Don't rely solely on mean SHAP direction** for continuous features
4. **Per-sample explanations** (API `include_explanation=true`) are always reliable for individual predictions
5. **Binary features** (like `avs_match`, `promo_used`) don't have this issue - their global direction is typically reliable

#### Implementation

The project uses **XGBoost's native SHAP computation** via `pred_contribs=True`:
- Fast (~1-5ms per prediction)
- Exact TreeSHAP algorithm (not approximation)
- Consistent with global SHAP analysis in notebook

## Key Functions

Functions are organized by module location for easy reference.

### Notebook Utilities (`src/fd1_nb/`)

#### Data Loading & Preprocessing (`src/fd1_nb/data_utils.py`)
- `load_data(data_dir, csv_file, verbose)`: Load CSV efficiently
- `split_train_val_test(df, val_ratio, test_ratio, stratify, r_seed)`: Create train/val/test splits with stratification
- `analyze_target_stats(df, target_col)`: Target distribution and imbalance detection with visualization
- `analyze_feature_stats(df, id_cols, target_col, categorical_features, numeric_features)`: Feature summary statistics

#### EDA & Analysis Functions (`src/fd1_nb/eda_utils.py`)
- `calculate_mi_scores(df, categorical_features, target_col)`: Mutual information for categorical features
- `calculate_numeric_correlations(df, numeric_features, target_col)`: Pearson correlations
- `calculate_vif(df, numeric_features)`: Variance Inflation Factor for multicollinearity
- `analyze_vif(df, numeric_features)`: VIF calculation and visualization with interpretation
- `analyze_correlations(df, numeric_features, target_col)`: Correlation analysis with bar chart visualization
- `analyze_mutual_information(df, categorical_features, target_col)`: MI score calculation and visualization

#### Feature Engineering (`src/fd1_nb/feature_engineering.py`)
- `convert_utc_to_local_time(df, date_col, country_col)`: Convert UTC to local time with timezone validation
- `create_temporal_features(df, date_col, suffix)`: Generate temporal features (hour, day_of_week, is_late_night, etc.)
- `create_interaction_features(df, feature_pairs)`: Fraud scenario-specific interaction features
- `create_percentile_based_features(df, feature_configs)`: Threshold-based binary flags

### Production Deployment (`src/deployment/`)

#### Data Loading (`src/deployment/data/loader.py`)
- `load_and_split_data(data_path, random_seed, verbose)`: Load CSV and perform stratified train/val/test splits

#### Feature Engineering (`src/deployment/preprocessing/`)
- `FraudFeatureTransformer`: Sklearn-compatible transformer for production feature engineering
- `FeatureConfig`: Dataclass for storing training-time thresholds
- `PreprocessingPipelineFactory`: Create model-specific preprocessing pipelines

#### Evaluation (`src/deployment/evaluation/`)
- `evaluate_model(model, X, y, model_name, dataset_name)`: Comprehensive model evaluation
- `optimize_thresholds(model, X_val, y_val)`: Find optimal thresholds for recall targets
- `calculate_metrics(y_true, y_pred, y_pred_proba)`: Calculate PR-AUC, ROC-AUC, F1, Precision, Recall

### Notebook 2 Utilities (`src/fd2_nb/`)

#### Model Comparison (`src/fd2_nb/model_comparison.py`)
- `compare_models(results_dict)`: Side-by-side model performance comparison
- `get_best_model(results_dict, metric)`: Select best model by specified metric
- `plot_comprehensive_comparison(results_dict)`: Full visualization of all metrics
- `plot_model_comparison(results_dict)`: Simple bar chart comparison

#### Hyperparameter Tuning (`src/fd2_nb/hyperparameter_tuning.py`)
- `create_search_object()`: Creates GridSearchCV or RandomizedSearchCV
- `tune_with_logging()`: Executes search with detailed logging to `models/logs/`
- `extract_cv_metrics()`: Extract metrics from CV results
- `get_best_params_summary()`: Format best parameters for display

#### CV Analysis (`src/fd2_nb/cv_analysis.py`)
- `analyze_cv_results()`: Production-focused CV results analysis
- `analyze_cv_train_val_gap()`: Detect overfitting via train-val gap
- `analyze_iteration_performance()`: Analyze XGBoost iteration curves
- `get_cv_statistics()`: Extract statistics from CV results

#### Bias-Variance (`src/fd2_nb/bias_variance.py`)
- `analyze_cv_fold_variance()`: Analyze variance across CV folds

### Notebook 3 Utilities (`src/fd3_nb/`)

#### Evaluation (`src/fd3_nb/evaluation.py`)
- `evaluate_model(model, X, y)`: Comprehensive model evaluation
- `compare_val_test_performance()`: Compare validation vs test metrics

#### Visualization (`src/fd3_nb/visualization.py`)
- `plot_roc_pr_curves()`: ROC and PR curve visualization
- `plot_feature_importance()`: XGBoost gain-based feature importance bar chart
- `plot_shap_importance()`: SHAP-based feature importance bar chart
- `plot_shap_beeswarm()`: SHAP beeswarm plot (handles categorical features)
- `plot_importance_comparison()`: Side-by-side gain vs SHAP comparison
- `plot_threshold_optimization()`: Threshold vs metrics curves

#### Threshold Optimization (`src/fd3_nb/threshold_optimization.py`)
- `find_threshold_for_recall()`: Find threshold for target recall
- `find_optimal_f1_threshold()`: Find threshold maximizing F1
- `find_target_performance_threshold()`: Find threshold meeting precision constraint
- `optimize_thresholds()`: Complete threshold optimization pipeline
- `create_threshold_comparison_df()`: Create comparison DataFrame

#### Feature Importance (`src/fd3_nb/feature_importance.py`)
- `extract_feature_importance()`: Extract gain-based importance from XGBoost pipeline
- `print_feature_importance_summary()`: Print formatted importance summary
- `compute_shap_importance()`: Compute SHAP values using XGBoost native method
- `compare_importance_methods()`: Compare gain vs SHAP rankings (Spearman correlation)
- `print_shap_importance_summary()`: Print SHAP summary with direction indicators
- `print_importance_comparison()`: Print ranking comparison between methods

#### Deployment (`src/fd3_nb/deployment.py`)
- `save_production_model()`: Save trained model with joblib
- `save_threshold_config()`: Save threshold configuration JSON
- `save_model_metadata()`: Save model metadata JSON
- `print_deployment_summary()`: Print deployment summary

### Explainability (`src/deployment/explainability/`)

#### SHAP Explainer (`src/deployment/explainability/shap_explainer.py`)
- `FraudExplainer`: Class for computing per-prediction SHAP explanations
  - `__init__(model, feature_names)`: Initialize with trained XGBoost model
  - `explain(features_df, top_n=3)`: Returns top N risk-increasing feature contributions

#### Feature Descriptions (`src/deployment/explainability/feature_descriptions.py`)
- `FEATURE_DESCRIPTIONS`: Dict mapping 30 technical feature names to human-readable display names
  - Example: `"avs_match"` → `"Address Verification Match"`
  - Example: `"is_new_account"` → `"New Account (≤30 days)"`

## Important Notes

### Gitignore
The following are excluded from version control:
- `data/` directory (contains large CSV files)
- `.kaggle/` directory and `kaggle.json` (credentials)
- Jupyter checkpoints
- Python cache files
- Virtual environments
- Trained model binaries (`xgb_fraud_detector.joblib`)
- Model training logs (`models/logs/*.log`, `models/logs/*.csv`)

**Note**: Configuration files (JSON) are tracked in git for reproducibility:
- `transformer_config.json` - Feature engineering configuration
- `threshold_config.json` - Decision threshold strategies
- `model_metadata.json` - Model version and performance metrics
- `feature_lists.json` - Feature categorization

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

### Run notebooks
```bash
# Launch Jupyter Lab
uv run --with jupyter jupyter lab

# Run notebooks in sequence (in notebooks/ folder):
# 1. notebooks/fd1_EDA_FE.ipynb (EDA and feature engineering)
# 2. notebooks/fd2_model_selection_tuning.ipynb (model selection and tuning)
# 3. notebooks/fd3_model_evaluation_deployment.ipynb (final evaluation and deployment)
```

### Run API server
```bash
# Development mode (with auto-reload)
uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --reload

# Production mode (better performance)
uv run uvicorn predict:app --host 0.0.0.0 --port 8000
```

### Run tests
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/preprocessing --cov-report=html
```

### Run benchmarks
```bash
# Quick performance benchmark
uv run python benchmarks/benchmark.py --url http://localhost:8000

# Custom benchmark configuration
uv run python benchmarks/benchmark.py --iterations 500 --concurrent 20

# Load testing with Locust (web UI)
locust -f benchmarks/locustfile.py --host=http://localhost:8000

# Headless load test with report
locust -f benchmarks/locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 120s --headless \
  --html=benchmarks/results/locust_report.html
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
- ✅ **Model serialization**: Trained model saved as `xgb_fraud_detector.joblib` with metadata
- ✅ **Threshold optimization**: Three pre-calibrated threshold strategies calibrated on validation set
- ✅ **Test set evaluation**: Final model (retrained on train+val) evaluated on held-out test set - all targets exceeded
- ✅ **Feature importance analysis**: Top 10 features identified using XGBoost gain metric
- ✅ **Model deployment with FastAPI**: Production REST API with automatic feature engineering and real-time predictions
- ✅ **Containerization**: Multi-stage Docker image with docker-compose for local deployment, optimized for production
- ✅ **Performance benchmarking**: Comprehensive benchmarking suite (benchmark.py + locustfile.py) with baseline results documented

### Remaining 🚧
- **Model monitoring and drift detection**: Track performance degradation over time
- **Automated retraining workflow**: Pipeline for periodic model updates
- **Cloud deployment**: Deploy to cloud platform (Google Cloud Run/AWS/Azure)
- **CI/CD pipeline**: Automated testing and deployment (GitHub Actions)
- **API authentication**: Secure API endpoints with API keys or OAuth
