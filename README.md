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
- **Notebook Environment**: Jupyter

### Deployment (Planned)
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
│   ├── transactions.csv                # Raw transaction data from Kaggle
│   ├── train_features.pkl              # Engineered training set
│   ├── val_features.pkl                # Engineered validation set
│   └── test_features.pkl               # Engineered test set
├── models/                             # Trained model artifacts (to be created)
├── src/                                # Source code (to be created)
│   ├── api/                            # FastAPI application
│   ├── preprocessing/                  # Data preprocessing pipelines
│   └── training/                       # Model training scripts
├── tests/                              # Unit tests (to be created)
├── Dockerfile                          # Docker configuration (to be created)
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

### Model Training & Evaluation
The modeling notebook (`fraud_detection_modeling.ipynb`) contains:
1. **Data Loading**: Load pre-engineered feature sets from pickle files
2. **Preprocessing**: Model-specific transformations (one-hot encoding, scaling)
3. **Baseline Models**: Logistic Regression, Random Forest, XGBoost
4. **Hyperparameter Tuning**: Grid search / randomized search
5. **Evaluation**: ROC-AUC, F1, Precision-Recall metrics (appropriate for imbalanced data)
6. **Model Selection**: Choose best performing model for deployment

### Model Training Strategy
Given the 44:1 class imbalance, the project employs:
- **Stratified sampling** to maintain class distribution across splits
- **Class weighting** in model training
- **Appropriate metrics**: F1, ROC-AUC, PR-AUC (not accuracy)
- **Threshold tuning** to optimize precision/recall trade-offs

### Adding Dependencies
```bash
uv add <package-name>
```

### Running Tests (Future)
```bash
pytest tests/
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

### Phase 1: Model Development (In Progress)
- [x] Dataset acquisition and exploration
- [x] Initial EDA and data quality checks
- [x] Preprocessing pipeline setup (stratified splits, type conversion)
- [x] Comprehensive exploratory data analysis
- [x] Feature engineering (32 features created)
- [x] Final feature selection (30 features selected)
- [x] Dataset persistence for modeling
- [ ] Baseline model training
- [ ] Model optimization and selection
- [ ] Final model evaluation on test set

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

## Model Performance Requirements

Target metrics for production deployment:
- **ROC-AUC**: > 0.90
- **F1 Score**: > 0.75
- **Precision**: > 0.80 (minimize false positives)
- **Recall**: > 0.70 (catch majority of fraud)
- **Inference Time**: < 100ms per prediction

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
