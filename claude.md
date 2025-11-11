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
├── ec_fraud_detection.ipynb  # Main analysis notebook
├── data/                      # Data directory (gitignored)
│   └── transactions.csv       # Downloaded dataset (~300k rows, 17 columns)
├── pyproject.toml             # Python dependencies
├── uv.lock                    # Locked dependency versions
├── .python-version            # Python version specification
├── .gitignore                 # Git exclusions
└── README.md                  # Project readme
```

## Dataset Information

### Size & Balance
- **Rows**: 299,695 transactions
- **Columns**: 17 features
- **Target**: `is_fraud` (binary: 0=normal, 1=fraud)
- **Class Distribution**:
  - Normal: 97.8%
  - Fraud: 2.2%
  - **Class Imbalance Ratio**: 44.3:1 (highly imbalanced!)

### Key Features
- **Transaction Identifiers**: `transaction_id`, `user_id`
- **User Behavior**: `account_age_days`, `total_transactions_user`, `avg_amount_user`
- **Transaction Details**: `amount`, `transaction_time`, `merchant_category`
- **Geographic**: `country`, `bin_country`, `shipping_distance_km`
- **Security Flags**: `avs_match`, `cvv_result`, `three_ds_flag`
- **Channel & Promotions**: `channel` (web/app), `promo_used`

### Data Quality
- No missing values
- No duplicate records
- Each row uniquely identified by `transaction_id` and `user_id`
- Memory usage: ~107 MB

## Technical Stack

### Core Dependencies (from pyproject.toml)
- **Python**: 3.12+
- **Data Science**: pandas, numpy, matplotlib, seaborn
- **ML Models**: scikit-learn, xgboost
- **Statistics**: statsmodels
- **Data Source**: kaggle (API client)
- **Notebook**: jupyter
- **API (future)**: fastapi, uvicorn

### Package Management
This project uses `uv` for fast, reliable Python dependency management.

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

### 1. Setup
- Parameter definitions (data paths, split ratios, column names)
- Package imports
- Utility function definitions

### 2. Data Loading
- `download_data_csv()`: Kaggle API download with caching
- `load_data()`: Efficient pandas CSV loading

### 3. Preprocessing
- Table grain verification
- Target class balance analysis
- Date type conversion
- Train/validation/test splits (stratified)

### 4. Exploratory Data Analysis (EDA)
- Target variable distribution
- Numeric feature analysis
- Multicollinearity detection (VIF)
- Bivariate analysis (features vs. target)
- Categorical feature fraud rates
- Feature selection recommendations

### 5. Model Training
- Baseline models (in progress)
- Model comparison and evaluation
- Hyperparameter tuning (planned)

## Key Functions

### Data Loading & Preprocessing
- `download_data_csv(kaggle_source, data_dir, csv_file)`: Download from Kaggle
- `load_data(data_dir, csv_file, verbose)`: Load CSV efficiently
- `split_train_val_test(df, val_ratio, test_ratio, stratify, r_seed)`: Create train/val/test splits

### Analysis Functions
- `analyze_target_stats(df, target_col)`: Target distribution and imbalance detection
- `analyze_feature_stats(df, id_cols, target_col)`: Feature summary statistics
- `calculate_mi_scores(df, categorical_features, target_col)`: Mutual information for categorical features
- `calculate_numeric_correlations(df, numeric_features, target_col)`: Pearson correlations
- `calculate_vif(df, numeric_features)`: Variance Inflation Factor for multicollinearity

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
