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
- **API Framework**: FastAPI
- **ASGI Server**: Uvicorn
- **Containerization**: Docker
- **Package Management**: uv (fast Python package installer)

## Project Structure

```
.
├── fraud_detection_EDA_FE.ipynb        # EDA & feature engineering notebook
├── fraud_detection_modeling.ipynb     # Model training & evaluation notebook
├── data/                               # Dataset directory (gitignored)
│   └── transactions.csv                # Raw transaction data from Kaggle
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
│   ├── test_api.py                     # Integration tests for Fraud Detection API.
├── models/                             # Model artifacts
│   └── feature_config.json             # Training-time configuration (tracked in git)
├── benchmark.py                        # Performance benchmarking script
├── locustfile.py                       # Load testing configuration (Locust)
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
   - Run cells sequentially

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
6. **Production Configuration**: Generates `FraudFeatureTransformer` configuration for deployment
   - Automatically creates `transformer_config.json` from training data
   - Stores quantile thresholds (95th/75th percentiles) for feature engineering
   - Saves 30 selected feature names with categorical groupings
   - Includes timezone mappings for 10 countries
   - Ensures consistent feature engineering between training and inference

### Model Training & Evaluation
The modeling notebook (`fraud_detection_modeling.ipynb`) contains:
1. **Data Loading**: Loads raw transaction data and applies `FraudFeatureTransformer` pipeline
   - Applies production feature engineering consistently across train/val/test splits
   - Generates 30 engineered features from 15 raw transaction fields
   - Uses same transformer configuration as deployment API
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

**Two-Stage Tuning Approach:**
- **Stage 1 (Exploration)**: RandomizedSearchCV with broad parameter ranges identifies stable parameters that remain unchanged across top-performing models
- **Stage 2 (Refinement)**: GridSearchCV on a focused parameter subset with narrow ranges, guided by Stage 1 insights
- Applied to Random Forest and XGBoost for efficient, thorough hyperparameter optimization

**Critical Hyperparameter Finding:**
- XGBoost's `scale_pos_weight` primarily controls the recall/precision trade-off and was included in the hyperparameter search space
- Using the actual class imbalance ratio (44:1) produced excessive false positives
- Optimal value of 8 (5.5× lower than class imbalance) achieved performance targets for both metrics
- Key tuning parameter for adapting model behavior to changing business requirements

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

### Phase 2: API Development ✅ (100% Complete)
- [x] Create FastAPI application structure
- [x] Implement prediction endpoint with Pydantic validation
- [x] Add input validation and comprehensive error handling
- [x] Create health check and monitoring endpoints
- [x] Write API documentation (OpenAPI/Swagger)
- [x] Multiple threshold strategies (conservative/balanced/aggressive)
- [x] Request logging and structured error responses
- [x] Comprehensive API integration tests (41 test cases)

### Phase 3: Containerization ✅ (100% Complete)
- [x] Create Dockerfile with multi-stage build
- [x] Optimize container image size (<500MB target)
- [x] Add docker compose for local development
- [x] Test containerized application
- [x] Security hardening (non-root user, health checks)
- [x] Build context optimization (.dockerignore)

### Phase 4: Production Deployment 🚧 (In Progress)
- [x] Implement logging and monitoring endpoints
- [x] Model artifact management and versioning
- [x] Automated testing (pytest integration)
- [ ] Deploy to cloud platform (Google Cloud Run/AWS/Azure)
- [ ] Set up CI/CD pipeline (GitHub Actions)
- [ ] Production monitoring dashboard
- [ ] Model performance tracking and alerting
- [ ] Model drift detection

## Model Training

Train the fraud detection model using the provided training script.

### Prerequisites
1. Raw transaction data must exist in `data/` directory:
   - `transactions.csv` (download from Kaggle)

**Note:** The training script uses raw transaction data and applies the production `FraudFeatureTransformer` pipeline, ensuring consistency between training and inference. All feature engineering is performed on-the-fly using the same transformer configuration deployed in the API.

### Training the Model

```bash
# Basic training (uses optimal hyperparameters, skips tuning for speed)
uv run python train.py --skip-tuning

# Full training with hyperparameter tuning (takes longer)
uv run python train.py

# Custom training options
uv run python train.py \
  --data-dir data \
  --output-dir models \
  --random-seed 42 \
  --verbose
```

**Output artifacts** (saved to `models/` directory):
- `xgb_fraud_detector.joblib` - Trained XGBoost model pipeline
- `transformer_config.json` - Feature transformer configuration (quantile thresholds)
- `threshold_config.json` - Optimized decision thresholds
- `model_metadata.json` - Model info, hyperparameters, performance
- `feature_lists.json` - Feature categorization
- `training_report.txt` - Detailed training summary

## API Deployment

Deploy the fraud detection model as a production REST API.

### Option 1: Local Development (FastAPI + Uvicorn)

#### 1. Install Dependencies
```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

#### 2. Train Model (if not already trained)
```bash
uv run python train.py --skip-tuning
```

#### 3. Start API Server
```bash
# Development mode with auto-reload
uv run uvicorn predict:app --reload --host 0.0.0.0 --port 8000

# Production mode
uv run uvicorn predict:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 4. Access API
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### Option 2: Docker Deployment (Recommended for Production)

#### 1. Build Docker Image
```bash
# Build the image (1.5 - 2 minutes)
docker build -t fraud-detection-api .

# Check image size (~1.5GB)
docker images fraud-detection-api
```

#### 2. Run Container
```bash
# Run with docker
docker run -d \
  --name fraud-api \
  -p 8000:8000 \
  fraud-detection-api

# Or use docker compose (easier)
docker compose up -d
```

#### 3. Verify Deployment
```bash
# Check health
curl http://localhost:8000/health

# View logs
docker logs fraud-api

# Stop container
docker compose down
```

### Option 3: Cloud Deployment

#### Google Cloud Run (Recommended - Free Tier Available)

```bash
# 1. Install Google Cloud SDK
# 2. Authenticate
gcloud auth login

# 3. Set project
gcloud config set project YOUR_PROJECT_ID

# 4. Build and deploy
gcloud run deploy fraud-detection-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# 5. Get deployment URL
gcloud run services describe fraud-detection-api --region us-central1 --format 'value(status.url)'
```

#### AWS Elastic Beanstalk

```bash
# 1. Initialize EB
eb init -p docker fraud-detection-api

# 2. Create environment
eb create fraud-api-prod

# 3. Deploy
eb deploy

# 4. Open in browser
eb open
```

## API Usage

### Prediction Endpoint

Make fraud predictions for transactions using the `/predict` endpoint.

The API accepts **raw transaction data** (15 fields) and automatically applies feature engineering using the production `FraudFeatureTransformer` pipeline before making predictions.

**Request:**
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

**Required Fields:**
- **User Information**: `user_id`, `account_age_days`, `total_transactions_user`, `avg_amount_user`
- **Transaction Details**: `amount`, `country`, `bin_country`, `channel`, `merchant_category`
- **Security Flags**: `promo_used`, `avs_match`, `cvv_result`, `three_ds_flag`
- **Geographic/Temporal**: `shipping_distance_km`, `transaction_time` (ISO format: `YYYY-MM-DD HH:MM:SS`)

**Note:** The API automatically generates 30 engineered features from these 15 raw fields using the production feature engineering pipeline.

**Response:**
```json
{
  "transaction_id": "550e8400-e29b-41d4-a716-446655440000",
  "is_fraud": false,
  "fraud_probability": 0.12,
  "risk_level": "low",
  "threshold_used": "balanced_85pct_recall",
  "threshold_value": 0.35,
  "model_version": "1.0",
  "processing_time_ms": 15.3
}
```

### Threshold Strategies

Choose different risk tolerance levels:

| Strategy | Target Recall | Use Case |
|----------|---------------|----------|
| `conservative_90pct_recall` | 90% | Catch maximum fraud (more false positives) |
| `balanced_85pct_recall` | 85% | Balanced approach (default) |
| `aggressive_80pct_recall` | 80% | Minimize false positives |

**Example - Conservative Strategy:**
```bash
curl -X POST "http://localhost:8000/predict?threshold_strategy=conservative_90pct_recall" \
  -H "Content-Type: application/json" \
  -d @transaction.json
```

### Health Check
```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0",
  "uptime_seconds": 3600.5,
  "timestamp": "2025-01-14T10:30:00Z"
}
```

### Model Information
```bash
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "model_name": "XGBoost Fraud Detector",
  "version": "1.0",
  "training_date": "2025-01-14T09:00:00",
  "algorithm": "XGBoost Gradient Boosting",
  "performance": {
    "pr_auc": 0.8679,
    "precision": 0.7233,
    "recall": 0.8360,
    "f1": 0.7756
  },
  "threshold_strategies": {
    "balanced_85pct_recall": {
      "threshold": 0.35,
      "precision": 0.72,
      "recall": 0.85
    }
  },
  "raw_features_required": [
    "user_id", "account_age_days", "total_transactions_user", "avg_amount_user",
    "amount", "country", "bin_country", "channel", "merchant_category",
    "promo_used", "avs_match", "cvv_result", "three_ds_flag",
    "shipping_distance_km", "transaction_time"
  ],
  "engineered_features_count": 30
}
```

## Testing

Run comprehensive test suite to verify model and API functionality:

### Unit Tests (Preprocessing Pipeline)
```bash
# Run all preprocessing tests
pytest tests/test_preprocessing/ -v

# Run specific test file
pytest tests/test_preprocessing/test_transformer.py -v

# Run with coverage
pytest tests/test_preprocessing/ --cov=src/preprocessing --cov-report=html
```

### Integration Tests (API)
```bash
# Run all API tests
pytest tests/test_api.py -v

# Run specific test class
pytest tests/test_api.py::TestPredictEndpoint -v

# Run with detailed output
pytest tests/test_api.py -v --tb=short
```

### All Tests
```bash
# Run entire test suite
pytest tests/ -v

# Quick smoke test
pytest tests/ -x  # Stop on first failure
```

**Test Coverage:**
- 41 preprocessing tests (100% passing)
- 25+ API integration tests
- Request/response validation
- Error handling scenarios
- Threshold strategies
- Performance validation

## Performance Benchmarking

Comprehensive performance testing suite to measure API latency and throughput.

### Running Benchmarks

#### Quick Benchmark (Python Script)

```bash
# Default benchmark (100 requests, 10 concurrent users)
uv run python benchmark.py --url http://localhost:8000

# Custom configuration
uv run python benchmark.py \
  --url http://localhost:8000 \
  --iterations 500 \
  --concurrent 20 \
  --output benchmark_results.json
```

**Metrics Measured:**
- **Cold Start**: First request latency (includes model loading overhead)
- **Single Request**: Sequential request latency (P50, P95, P99)
- **Concurrent Load**: Multi-user throughput (requests/second)
- **Server vs E2E**: Processing time vs total latency (network overhead)

#### Load Testing (Locust)

```bash
# Start Locust web UI
uv run locust -f locustfile.py --host=http://localhost:8000

# Headless mode with 50 users, run for 60 seconds
uv run locust -f locustfile.py \
  --host=http://localhost:8000 \
  --users 50 \
  --spawn-rate 10 \
  --run-time 60s \
  --headless

# Generate HTML report
uv run locust -f locustfile.py \
  --host=http://localhost:8000 \
  --users 100 \
  --spawn-rate 10 \
  --run-time 120s \
  --headless \
  --html=locust_report.html
```

**User Scenarios:**
- **Normal Transactions** (70% of traffic): Established accounts with typical amounts
- **Suspicious Transactions** (30% of traffic): New accounts with high amounts
- **Stress Test**: Rapid-fire requests with minimal wait time

### Benchmark Results

**Environment:**
- Platform: Linux (Ubuntu 24.04)
- Python: 3.12
- Deployment: Docker (docker compose)
- Date: 2025-11-15

**Single Request Performance (500 iterations):**

| Metric | Mean | Median | P95 | P99 |
|--------|------|--------|-----|-----|
| **Server Processing** | 19.94 ms | 18.51 ms | 33.84 ms | 39.54 ms |
| **End-to-End Latency** | 22.35 ms | 20.73 ms | 37.13 ms | 44.36 ms |
| **Network Overhead** | 2.41 ms | 2.10 ms | - | - |

**Concurrent Load Performance (20 concurrent users, 500 requests):**

| Metric | Value |
|--------|-------|
| **Throughput** | 48.16 requests/second |
| **Success Rate** | 100% |
| **Total Time** | 10.38 seconds |
| **Server P95** | 32.16 ms |
| **Server P99** | 37.10 ms |
| **E2E P95** | 557.64 ms |
| **E2E P99** | 603.04 ms |

**Cold Start Performance:**

| Metric | Latency |
|--------|---------|
| **Server Processing** | 52.37 ms |
| **End-to-End** | 54.98 ms |
| **Network Overhead** | 2.61 ms |

### Performance Analysis

✅ **Excellent Latency**: Sub-35ms P95 server processing (33.84ms - well below 50ms target)
✅ **Consistent Performance**: Minimal variance between P50 and P95 (18.5ms → 33.8ms)
✅ **Fast Cold Start**: ~52ms on first request (including container overhead)
✅ **High Reliability**: 100% success rate under load (500 requests)
✅ **Highly Scalable**: 48.16 RPS on single instance (extrapolates to 173k+ requests/hour)

**Comprehensive Testing:** Results based on 500 iterations with 20 concurrent users, providing more accurate performance characterization than typical benchmarks.

**Docker Overhead:** Cold start is ~52ms (vs ~26ms for local uvicorn), which is acceptable for production deployment. The containerization provides isolation and portability benefits with minimal performance impact.

**Network Overhead:** Average 2.4ms indicates local deployment. Production deployments will add 10-50ms depending on geographic distance.

**Concurrent Load:** E2E latency increases under concurrent load (397ms P50) due to request queueing, but server processing remains consistently fast (18.5ms P50).

### Performance Targets vs Achieved

| Target | Achieved (Docker) | Status |
|--------|-------------------|--------|
| Server P95 < 50ms | 33.84 ms | ✅ Pass (32% better) |
| Server P99 < 100ms | 39.54 ms | ✅ Pass (60% better) |
| Throughput > 20 RPS | 48.16 RPS | ✅ Pass (140% of target) |
| Success Rate 100% | 100% | ✅ Pass |

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
- **Inference Time (P95)**: 33.84ms ✅ (Target: < 50ms)
- **Inference Time (P99)**: 39.54ms ✅ (Target: < 100ms)

**Model Details:**
- Best hyperparameters: n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=8
- Confusion Matrix: TN=58,206 | FP=410 | FN=222 | TP=1,101
- Excellent precision-recall balance for fraud detection
- Significant improvement over baseline (+31.5% precision, +2.5% PR-AUC)

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
