# E-Commerce Fraud Detection

A machine learning project to detect fraudulent e-commerce transactions using classification models. The goal is to develop, optimize, and deploy a production-ready fraud detection system.

## Project Overview

This project builds and deploys a classification model to identify fraudulent e-commerce transactions in real-time. Using a realistic synthetic dataset that models actual fraud patterns observed in 2024, the system aims to help e-commerce platforms prevent fraudulent activity while minimizing false positives that could impact legitimate customers.

**Project Goal**: Deploy an optimally trained classification model capable of identifying fraudulent transactions with high precision and recall, packaged as a REST API service.

### Example Fraud Patterns Detected
- Card testing with small-value purchases (e.g., $1 transactions at midnight)
- Geographic anomalies (e.g., gaming accessories shipped 5,000 km away)
- Promo code abuse from newly created accounts
- Mismatched verification signals (AVS, CVV, 3D Secure)

## About This Project

This project is being developed as part of the [DataTalksClub Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp), a comprehensive course covering:
- Machine learning fundamentals
- Model training and evaluation
- Deployment and MLOps practices
- Production-ready ML systems

## Dataset

**Source**: [Kaggle - E-Commerce Fraud Detection Dataset](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)

### Dataset Characteristics
- **Size**: 299,695 transactions
- **Features**: 17 columns (user behavior, transaction details, security flags, geographic data)
- **Target**: `is_fraud` (binary classification)
- **Class Distribution**:
  - Normal transactions: 97.8%
  - Fraudulent transactions: 2.2%
  - **Imbalance ratio**: 44:1 (significant class imbalance)
- **Quality**: No missing values, no duplicates

### Key Features
- **User Behavior**: Account age, transaction history, average spend patterns
- **Transaction Details**: Amount, timestamp, merchant category, channel (web/app)
- **Geographic Data**: User country, card-issuing bank country, shipping distance
- **Security Signals**: AVS match, CVV result, 3D Secure flag, promo code usage

## Technology Stack

### Data Science & ML
- **Python**: 3.12+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML Models**: scikit-learn, xgboost
- **Statistics**: statsmodels
- **Timezone Handling**: pytz (UTC to local time conversion)
- **Testing**: pytest (unit and integration tests)
- **Notebook Environment**: Jupyter

### Deployment
- **Feature Engineering**: Production-ready sklearn-compatible transformer
- **API Framework**: FastAPI (planned)
- **ASGI Server**: Uvicorn (planned)
- **Containerization**: Docker (planned)
- **Package Management**: uv (fast Python package installer)

## Project Structure

```
.
├── fraud_detection_EDA_FE.ipynb        # EDA & feature engineering notebook
├── fraud_detection_modeling.ipynb     # Model training & evaluation notebook
├── data/                               # Dataset directory (gitignored)
│   ├── transactions.csv                # Raw transaction data from Kaggle
│   ├── train_features.pkl              # Engineered training set (179,817 × 31)
│   ├── val_features.pkl                # Engineered validation set (59,939 × 31)
│   └── test_features.pkl               # Engineered test set (59,939 × 31)
├── src/                                # Production source code
│   └── preprocessing/                  # Feature engineering pipeline
│       ├── config.py                   # FeatureConfig dataclass (JSON serialization)
│       ├── features.py                 # Feature engineering functions
│       ├── transformer.py              # FraudFeatureTransformer (sklearn-compatible)
│       └── __init__.py                 # Package exports
├── tests/                              # Test suite (41 passing tests)
│   ├── conftest.py                     # Shared pytest fixtures
│   └── test_preprocessing/             # Preprocessing tests
│       ├── test_config.py              # FeatureConfig tests (8 tests)
│       ├── test_features.py            # Feature function tests (23 tests)
│       └── test_transformer.py         # Transformer integration tests (18 tests)
├── models/                             # Model artifacts
│   └── feature_config.json             # Training-time configuration (tracked in git)
├── pyproject.toml                      # Python dependencies
├── uv.lock                             # Locked dependency versions
├── .gitignore                          # Git exclusions
├── claude.md                           # Project context for Claude Code
└── README.md                           # This file
```

## Getting Started

### Prerequisites

1. **Python 3.12+**
   ```bash
   python --version  # Verify installation
   ```

2. **uv Package Manager**
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Kaggle API Credentials (Optional)**
   - Create an account at [kaggle.com](https://www.kaggle.com)
   - Go to Account settings → API → Create New Token
   - Place the downloaded `kaggle.json` in `~/.kaggle/`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/frederick-douglas-pearce/e-commerce-fraud-detection.git
   cd e-commerce-fraud-detection
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Launch Jupyter Notebook**
   ```bash
   uv run --with jupyter jupyter lab
   ```

4. **Run the notebooks**
   - Open `fraud_detection_EDA_FE.ipynb` for EDA and feature engineering
   - Run cells sequentially
   - Dataset will auto-download on first run if not present
   - Open `fraud_detection_modeling.ipynb` for model training (after EDA is complete)

## Development Workflow

### Data Analysis & Feature Engineering
The EDA notebook (`fraud_detection_EDA_FE.ipynb`) contains:
1. **Data Loading**: Automated Kaggle dataset download with caching
2. **Preprocessing**: Data cleaning, type conversion, train/val/test splits (60/20/20, stratified)
3. **EDA**: Comprehensive exploratory data analysis
   - Target distribution and class imbalance analysis (44:1 ratio)
   - Numeric feature distributions and correlations
   - Categorical feature fraud rates and mutual information
   - Temporal pattern analysis
   - Multicollinearity detection (VIF)
4. **Feature Engineering**: Created 32 engineered features
   - **Temporal**: UTC and local timezone features (hour, day_of_week, is_late_night, etc.)
   - **Amount**: Deviation, ratios, micro/large transaction flags
   - **User Behavior**: Transaction velocity, new account flags, frequency indicators
   - **Geographic**: Country mismatch, high-risk distance, zero distance
   - **Security**: Composite security score from verification flags
   - **Interaction**: Fraud scenario-specific combinations (e.g., new_account_with_promo)
5. **Feature Selection**: Final selection of **30 features** from 45 available
   - Removed redundant features (UTC features, duplicate country fields)
   - Excluded low-signal features (merchant_category)
   - Prioritized interpretability and fraud scenario alignment
6. **Dataset Persistence**: Saves engineered train/val/test sets as pickle files
7. **Config Export**: Automatically generates `feature_config.json` for deployment
   - Stores quantile thresholds from training data
   - 30 selected feature names
   - Timezone mappings for 10 countries

### Model Training & Evaluation
The modeling notebook (`fraud_detection_modeling.ipynb`) contains:
1. **Data Loading**: Load pre-engineered feature sets from pickle files
2. **Preprocessing**: Model-specific transformations (one-hot encoding, scaling)
3. **Baseline Models**: Logistic Regression, Random Forest, XGBoost (all trained)
4. **Hyperparameter Tuning**: Flexible GridSearchCV/RandomizedSearchCV with detailed logging
   - Random Forest: GridSearchCV over 8 parameter combinations
   - XGBoost: GridSearchCV over 108 combinations (tuned scale_pos_weight, gamma, learning_rate)
5. **CV Results Analysis**: Production-focused evaluation of model stability and timing
   - Comprehensive CSV logging of all CV results
   - Stability analysis (std_test_score across folds)
   - Timing measurements with appropriate caveats for parallel processing
6. **Evaluation**: ROC-AUC, PR-AUC, F1, Precision-Recall metrics (appropriate for imbalanced data)
7. **Model Selection**: XGBoost (Tuned) selected as best performer (PR-AUC: 0.8679)

### Model Training Strategy
Given the 44:1 class imbalance, the project employs:
- **Stratified sampling** to maintain class distribution across splits
- **Class weighting** in model training (class_weight='balanced', scale_pos_weight)
- **Appropriate metrics**: PR-AUC (primary), ROC-AUC, F1, Precision-Recall (not accuracy)
- **Threshold tuning** to optimize precision/recall trade-offs
- **4-fold Stratified CV** for hyperparameter optimization

### Hyperparameter Tuning Features
The modeling pipeline includes production-ready tuning capabilities:

**Flexible Search Strategy:**
- Switch between GridSearchCV and RandomizedSearchCV with a single parameter
- Automatic calculation of total parameter combinations
- Support for both exhaustive and random search approaches

**Comprehensive Logging:**
- Detailed CV results exported to timestamped CSV files
- Verbose output captured to log files
- All parameter combinations and scores preserved for analysis

**Production-Focused Analysis:**
- Model stability evaluation (std_test_score across CV folds)
- Timing measurements with appropriate caveats for parallel processing
- Top N candidates comparison for trade-off analysis
- Automated recommendations for model selection
- Visual analysis of performance vs stability trade-offs

**Key Insights:**
- Timing metrics are unreliable with parallel CV (measurement artifacts)
- Focus on PR-AUC and stability for model selection
- Production API latency testing provides definitive performance numbers

## Production Feature Engineering Pipeline

The project includes a production-ready feature engineering pipeline (`src/preprocessing/`) designed for deployment. This sklearn-compatible transformer ensures consistent feature engineering between training and inference.

### Architecture Overview

**Design Pattern**: Hybrid Class + Config (sklearn-compatible transformer with JSON configuration)

**Key Components**:
1. **`FraudFeatureTransformer`** - Sklearn-compatible transformer class
   - `fit(X)` - Calculates quantile thresholds from training data
   - `transform(X)` - Applies feature engineering pipeline
   - `save(path)` / `load(path)` - Persists configuration as JSON

2. **`FeatureConfig`** - Type-safe configuration dataclass
   - Stores training-time statistics (95th/75th percentile thresholds)
   - Timezone mappings for 10 countries
   - List of 30 final selected features
   - JSON serialization for version control

3. **Feature Engineering Functions** - Modular, testable functions
   - Timezone conversion (UTC → local time by country)
   - Temporal, amount, behavior, geographic, security features
   - Fraud scenario-specific interaction features

### Usage

**Training Workflow**:
```python
from src.preprocessing import FraudFeatureTransformer

# Fit transformer on training data
transformer = FraudFeatureTransformer()
transformer.fit(train_df)  # Calculates quantile thresholds
X_train = transformer.transform(train_df)

# Save configuration for deployment
transformer.save("models/feature_config.json")
```

**Inference Workflow**:
```python
# Load transformer with saved configuration
transformer = FraudFeatureTransformer.load("models/feature_config.json")
X_new = transformer.transform(new_df)
```

**Sklearn Pipeline Integration**:
```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('feature_engineering', FraudFeatureTransformer()),
    ('classifier', LogisticRegression())
])
pipeline.fit(train_df, y_train)
predictions = pipeline.predict(test_df)
```

### Benefits

✅ **Sklearn Pipeline compatible** - Standard fit/transform API
✅ **Lightweight** - JSON config (not pickled Python objects)
✅ **Version control friendly** - Config changes visible in diffs
✅ **Type-safe** - Dataclass with validation
✅ **Fully tested** - 41 passing tests with edge case coverage
✅ **Production-ready** - Industry standard pattern

### Configuration File

The `feature_config.json` file stores:
```json
{
  "amount_95th_percentile": 595.97,
  "total_transactions_75th_percentile": 56,
  "shipping_distance_75th_percentile": 408.9,
  "timezone_mapping": { "US": "America/New_York", ... },
  "final_features": [ "account_age_days", "amount", ... ],
  "date_col": "transaction_time",
  "country_col": "country"
}
```

## Testing

The project includes comprehensive test coverage for the feature engineering pipeline.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage report
uv run pytest --cov=src/preprocessing --cov-report=html

# Run specific test file
uv run pytest tests/test_preprocessing/test_transformer.py

# Run tests in verbose mode
uv run pytest -v
```

### Test Suite Overview

**Total: 41 passing tests**

- **`test_config.py`** (8 tests)
  - Configuration creation and validation
  - Save/load round-trip testing
  - JSON structure verification
  - Quantile calculation from training data

- **`test_features.py`** (23 tests)
  - Individual feature function testing
  - Edge case handling (zero values, division by zero)
  - Timezone validation (strict UTC enforcement)
  - Binary feature output verification

- **`test_transformer.py`** (18 tests)
  - Full pipeline integration
  - Output shape verification (30 features)
  - Sklearn Pipeline compatibility
  - Save/load consistency
  - Multiple transform consistency

### Development Commands

```bash
# Add new dependencies
uv add <package-name>

# Update dependencies
uv sync

# Run Jupyter notebook
uv run --with jupyter jupyter lab
```

## Feature Engineering Summary

The project implements comprehensive feature engineering targeting the three specific fraud scenarios:

### Engineered Features (30 selected from 32 created)

**1. Temporal Features (6) - Local Timezone**
- `hour_local`, `day_of_week_local`, `month_local`
- `is_weekend_local`, `is_late_night_local` (11 PM - 4 AM), `is_business_hours_local`
- **Why local time?** Better captures human behavior patterns. Fraud at "2 AM local" is suspicious regardless of UTC time.

**2. Transaction Amount Features (4)**
- `amount_deviation` - Absolute deviation from user's average
- `amount_vs_avg_ratio` - Ratio of transaction to user average
- `is_micro_transaction` - Flags amounts ≤$5 (card testing pattern)
- `is_large_transaction` - Flags 95th percentile+ amounts

**3. User Behavior Features (3)**
- `transaction_velocity` - Transactions per day of account age
- `is_new_account` - Accounts <30 days old (promo abuse pattern)
- `is_high_frequency_user` - 75th percentile+ transaction count

**4. Geographic Features (3)**
- `country_mismatch` - User country ≠ card issuing country (replaces separate fields)
- `high_risk_distance` - Shipping distance >75th percentile
- `zero_distance` - Billing = shipping address (lower risk)

**5. Security Features (1)**
- `security_score` - Composite score: avs_match + cvv_result + three_ds_flag (0-3)

**6. Interaction Features (3) - Fraud Scenario Specific**
- `new_account_with_promo` → **Scenario #3**: Promo abuse from fresh accounts
- `late_night_micro_transaction` → **Scenario #1**: Card testing at midnight
- `high_value_long_distance` → **Scenario #2 variant**: Large amounts shipped far

**Original Features Retained (10)**
- Numeric (5): account_age_days, total_transactions_user, avg_amount_user, amount, shipping_distance_km
- Categorical (5): channel, promo_used, avs_match, cvv_result, three_ds_flag

**Total: 30 features + 1 target = 31 columns**

## Deployment Plan

### Phase 1: Model Development & Feature Engineering ✅ (100% Complete)
- [x] Dataset acquisition and exploration
- [x] Initial EDA and data quality checks
- [x] Preprocessing pipeline setup (stratified splits, type conversion)
- [x] Comprehensive exploratory data analysis
- [x] Feature engineering (32 features created)
- [x] Final feature selection (30 features selected)
- [x] Dataset persistence for modeling
- [x] **Production feature engineering pipeline** (sklearn-compatible)
- [x] **Comprehensive test suite** (41 passing tests)
- [x] **Configuration management** (JSON-based FeatureConfig)
- [x] **Baseline model training** (Logistic Regression, Random Forest, XGBoost)
- [x] **Hyperparameter tuning** (Random Forest and XGBoost optimized)
- [x] **CV analysis tooling** (Production-focused stability and timing evaluation)
- [x] **Model selection** (XGBoost Tuned - PR-AUC: 0.8679)
- [x] **Test set evaluation** (PR-AUC: 0.8679, excellent generalization)
- [x] **Feature importance analysis** (XGBoost built-in + SHAP values)
- [x] **Threshold optimization** (Multiple recall targets: 80%, 85%, 90%)
- [x] **Model persistence and deployment package** (Model, metadata, thresholds, model card)

### Phase 2: API Development
- [ ] Create FastAPI application structure
- [ ] Implement prediction endpoint
- [ ] Add input validation and error handling
- [ ] Create health check and monitoring endpoints
- [ ] Write API documentation (OpenAPI/Swagger)

### Phase 3: Containerization
- [ ] Create Dockerfile
- [ ] Optimize container image size
- [ ] Add docker-compose for local development
- [ ] Test containerized application

### Phase 4: Production Deployment
- [ ] Set up CI/CD pipeline
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
- [ ] Implement logging and monitoring
- [ ] Set up model performance tracking
- [ ] Create alerting for model drift

## API Design (Planned)

### Prediction Endpoint
```
POST /predict
Content-Type: application/json

{
  "user_id": 12345,
  "account_age_days": 150,
  "amount": 99.99,
  "country": "US",
  "bin_country": "US",
  "channel": "web",
  "merchant_category": "electronics",
  "promo_used": 0,
  "avs_match": 1,
  "cvv_result": 1,
  "three_ds_flag": 1,
  "shipping_distance_km": 250.5,
  "total_transactions_user": 45,
  "avg_amount_user": 125.50
}

Response:
{
  "is_fraud": false,
  "fraud_probability": 0.03,
  "risk_level": "low",
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Health Check
```
GET /health

Response:
{
  "status": "healthy",
  "model_version": "1.0.0",
  "uptime": 3600
}
```

## Model Performance

### Target Metrics (Production Deployment)
- **PR-AUC**: > 0.85
- **ROC-AUC**: > 0.95
- **F1 Score**: > 0.75
- **Recall**: > 0.80 (prioritize catching fraud)
- **Precision**: > 0.70 (minimize false positives)
- **Inference Time**: < 100ms per prediction

### Achieved Results (XGBoost Tuned - Validation Set)
- **PR-AUC**: 0.8679 ✅ (Target: > 0.85)
- **ROC-AUC**: 0.9790 ✅ (Target: > 0.95)
- **F1 Score**: 0.7756 ✅ (Target: > 0.75)
- **Recall**: 0.8360 ✅ (Target: > 0.80)
- **Precision**: 0.7233 ✅ (Target: > 0.70)
- **Inference Time**: TBD (to be measured in production API)

**Model Details:**
- Best hyperparameters: n_estimators=90, max_depth=5, learning_rate=0.08, scale_pos_weight=8
- Confusion Matrix: TN=58,193 | FP=423 | FN=217 | TP=1,106
- Excellent precision-recall balance for fraud detection
- Significant improvement over baseline (+32.1% precision, +2.6% PR-AUC)

## Contributing

This is a personal learning project, but suggestions and feedback are welcome! Feel free to:
- Open issues for bugs or suggestions
- Submit pull requests with improvements
- Share ideas for model improvements or deployment strategies

## License

This project is developed for educational purposes as part of the ML Zoomcamp course.

## Acknowledgments

- [DataTalksClub](https://github.com/DataTalksClub) for the excellent Machine Learning Zoomcamp
- [Kaggle](https://www.kaggle.com) and the dataset creator for providing realistic fraud detection data
- The open-source ML community for the amazing tools and libraries

## Contact

Frederick Douglas Pearce
- GitHub: [@frederick-douglas-pearce](https://github.com/frederick-douglas-pearce)

## Resources

- [ML Zoomcamp Course](https://github.com/DataTalksClub/machine-learning-zoomcamp)
- [Dataset on Kaggle](https://www.kaggle.com/datasets/umuttuygurr/e-commerce-fraud-detection-dataset)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
