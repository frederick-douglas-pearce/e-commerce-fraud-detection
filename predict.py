#!/usr/bin/env python3
"""
E-Commerce Fraud Detection API

FastAPI web service for real-time fraud prediction using trained XGBoost model.

Usage:
    uvicorn predict:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /predict - Predict fraud for a transaction
    GET  /health  - Health check
    GET  /model/info - Model information and metadata
    GET  /         - API root with links to documentation
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Import production feature engineering pipeline
from src.deployment.preprocessing.transformer import FraudFeatureTransformer

# Import explainability module
from src.deployment.explainability import FraudExplainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for model and configuration
model = None
transformer = None
threshold_config = None
model_metadata = None
feature_lists = None
startup_time = None
explainer = None  # SHAP explainer for prediction explanations


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model, transformer, threshold_config, model_metadata, feature_lists, startup_time, explainer

    # Startup: Load model and configuration files
    startup_time = datetime.now()
    logger.info("Starting E-Commerce Fraud Detection API...")

    try:
        models_dir = Path("models")

        # Load feature transformer
        transformer_config_path = models_dir / "transformer_config.json"
        if not transformer_config_path.exists():
            raise FileNotFoundError(f"Transformer config not found: {transformer_config_path}")

        transformer = FraudFeatureTransformer.load(str(transformer_config_path))
        logger.info(f"✓ Feature transformer loaded from {transformer_config_path}")

        # Load model
        model_path = models_dir / "xgb_fraud_detector.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        logger.info(f"✓ Model loaded from {model_path}")

        # Load threshold configuration
        threshold_path = models_dir / "threshold_config.json"
        with open(threshold_path, "r") as f:
            threshold_config = json.load(f)
        logger.info(f"✓ Threshold config loaded: {list(threshold_config.keys())}")

        # Load model metadata
        metadata_path = models_dir / "model_metadata.json"
        with open(metadata_path, "r") as f:
            model_metadata = json.load(f)
        logger.info(f"✓ Model metadata loaded: v{model_metadata['model_info']['version']}")

        # Load feature lists
        feature_lists_path = models_dir / "feature_lists.json"
        with open(feature_lists_path, "r") as f:
            feature_lists = json.load(f)
        logger.info(f"✓ Feature lists loaded: {len(feature_lists['all_features'])} features")

        # Initialize SHAP explainer for prediction explanations
        # Get feature names in the order the preprocessor outputs them (after encoding)
        preprocessor = model.named_steps["preprocessor"]
        preprocessed_feature_names = [
            name.split("__", 1)[1] if "__" in name else name
            for name in preprocessor.get_feature_names_out()
        ]
        explainer = FraudExplainer(model, feature_names=preprocessed_feature_names)
        logger.info("✓ SHAP explainer initialized for prediction explanations")

        logger.info("=" * 80)
        logger.info("API READY FOR REQUESTS")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise

    yield

    # Shutdown: cleanup (if needed)
    logger.info("Shutting down E-Commerce Fraud Detection API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="E-Commerce Fraud Detection API",
    description="Real-time fraud prediction service using XGBoost machine learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# Pydantic models for request/response validation
class RawTransactionRequest(BaseModel):
    """Raw transaction data for fraud prediction.

    The API accepts raw transaction data and automatically applies feature engineering
    using the production FraudFeatureTransformer pipeline.
    """

    # User features
    user_id: int = Field(..., description="User ID", ge=0)
    account_age_days: int = Field(..., description="Account age in days", ge=0)
    total_transactions_user: int = Field(
        ..., description="Total number of transactions by user", ge=0
    )
    avg_amount_user: float = Field(..., description="Average transaction amount for user", ge=0.0)

    # Transaction features
    amount: float = Field(..., description="Transaction amount in USD", ge=0.0)
    country: str = Field(..., description="Transaction country (2-letter code)", min_length=2, max_length=2)
    bin_country: str = Field(..., description="Card BIN country (2-letter code)", min_length=2, max_length=2)
    channel: str = Field(..., description="Transaction channel", pattern="^(web|app)$")
    merchant_category: str = Field(..., description="Merchant category code")

    # Security flags
    promo_used: int = Field(..., description="Promo code used (0 or 1)", ge=0, le=1)
    avs_match: int = Field(..., description="AVS match result (0 or 1)", ge=0, le=1)
    cvv_result: int = Field(..., description="CVV verification result (0 or 1)", ge=0, le=1)
    three_ds_flag: int = Field(..., description="3D Secure flag (0 or 1)", ge=0, le=1)

    # Geographic and temporal features
    shipping_distance_km: float = Field(..., description="Shipping distance in km", ge=0.0)
    transaction_time: str = Field(
        ...,
        description="Transaction timestamp in ISO format (e.g., '2024-01-15 14:30:00')",
        pattern=r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    )

    @field_validator("country", "bin_country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase."""
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
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
            }
        }
    }


class PredictionResponse(BaseModel):
    """Fraud prediction response."""

    transaction_id: str = Field(..., description="Unique transaction ID")
    is_fraud: bool = Field(..., description="Fraud prediction (True/False)")
    fraud_probability: float = Field(..., description="Fraud probability (0.0-1.0)")
    risk_level: Literal["low", "medium", "high"] = Field(..., description="Risk level classification")
    threshold_used: str = Field(..., description="Threshold strategy applied")
    threshold_value: float = Field(..., description="Threshold value used")
    model_version: str = Field(..., description="Model version")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    explanation: Optional["ExplanationResponse"] = Field(
        None, description="Explanation of the fraud score (when include_explanation=true)"
    )

    model_config = {"json_schema_extra": {"example": {"transaction_id": "550e8400-e29b-41d4-a716-446655440000", "is_fraud": False, "fraud_probability": 0.12, "risk_level": "low", "threshold_used": "balanced_85pct_recall", "threshold_value": 0.35, "model_version": "1.0", "processing_time_ms": 15.3, "explanation": None}}}


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy"] = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Model loaded status")
    model_version: Optional[str] = Field(None, description="Model version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str
    version: str
    training_date: str
    algorithm: str
    performance: dict
    threshold_strategies: dict
    raw_features_required: list[str]
    engineered_features_count: int


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str
    timestamp: str


class FeatureContributionResponse(BaseModel):
    """A single feature's contribution to the fraud prediction.

    SHAP values are computed in log-odds space where they are additive.
    The probability contribution is an approximation for interpretability.
    """

    feature: str = Field(..., description="Technical feature name")
    display_name: str = Field(..., description="Human-readable feature name")
    value: float = Field(..., description="Actual feature value for this prediction")
    contribution_log_odds: float = Field(
        ..., description="SHAP value in log-odds space (additive, positive = increases fraud risk)"
    )
    contribution_probability: float = Field(
        ..., description="Approximate probability shift caused by this feature (e.g., +0.15 means +15%)"
    )


class ExplanationResponse(BaseModel):
    """Explanation of a fraud prediction showing top contributing features.

    Understanding SHAP Values:
    - SHAP values are computed in LOG-ODDS space, not probability space
    - In log-odds space, contributions are additive: base + sum(SHAP) = final_log_odds
    - The final probability = 1 / (1 + exp(-final_log_odds))
    - contribution_probability shows the approximate impact on the final probability
    """

    top_contributors: list[FeatureContributionResponse] = Field(
        ..., description="Top features contributing to the fraud score"
    )
    base_fraud_rate: float = Field(
        ..., description="Training data fraud rate (~2.2%), the prior probability before considering features"
    )
    final_fraud_probability: float = Field(
        ..., description="Model's predicted fraud probability for this transaction"
    )
    explanation_method: str = Field(default="shap", description="Method used for explanation")

    model_config = {
        "json_schema_extra": {
            "example": {
                "top_contributors": [
                    {
                        "feature": "account_age_days",
                        "display_name": "Account Age (days)",
                        "value": 5,
                        "contribution_log_odds": 2.60,
                        "contribution_probability": 0.18,
                    },
                    {
                        "feature": "security_score",
                        "display_name": "Security Score (0-3)",
                        "value": 0,
                        "contribution_log_odds": 1.08,
                        "contribution_probability": 0.06,
                    },
                ],
                "base_fraud_rate": 0.022,
                "final_fraud_probability": 0.917,
                "explanation_method": "shap",
            }
        }
    }


# API endpoints
@app.get("/", tags=["root"])
async def root():
    """API root endpoint with links to documentation."""
    return {
        "name": "E-Commerce Fraud Detection API",
        "version": "1.0.0",
        "status": "online",
        "endpoints": {
            "predict": "/predict (POST)",
            "health": "/health (GET)",
            "model_info": "/model/info (GET)",
            "docs": "/docs (GET)",
            "redoc": "/redoc (GET)",
        },
        "documentation": "Visit /docs for interactive API documentation",
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """
    Health check endpoint for monitoring service status.

    Returns service health, model status, and uptime information.
    """
    uptime = (datetime.now() - startup_time).total_seconds()

    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model_metadata["model_info"]["version"] if model_metadata else None,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
async def get_model_info():
    """
    Get model information and metadata.

    Returns model version, training date, performance metrics, and configuration.
    The API accepts raw transaction data and automatically applies feature engineering.
    """
    if model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not loaded",
        )

    # List of raw features required for prediction
    raw_features = [
        "user_id",
        "account_age_days",
        "total_transactions_user",
        "avg_amount_user",
        "amount",
        "country",
        "bin_country",
        "channel",
        "merchant_category",
        "promo_used",
        "avs_match",
        "cvv_result",
        "three_ds_flag",
        "shipping_distance_km",
        "transaction_time"
    ]

    return ModelInfoResponse(
        model_name=model_metadata["model_info"]["model_name"],
        version=model_metadata["model_info"]["version"],
        training_date=model_metadata["model_info"]["training_date"],
        algorithm=model_metadata["model_info"]["model_type"],
        performance=model_metadata["performance"]["test_set"],
        threshold_strategies=threshold_config["optimized_thresholds"],
        raw_features_required=raw_features,
        engineered_features_count=len(feature_lists["all_features"]),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_fraud(
    transaction: RawTransactionRequest,
    threshold_strategy: Literal[
        "optimal_f1", "target_performance", "conservative_90pct_recall", "balanced_85pct_recall", "aggressive_80pct_recall"
    ] = "optimal_f1",
    include_explanation: bool = Query(
        False,
        description="Include SHAP-based explanation of top contributing features",
    ),
    top_n: int = Query(
        3,
        ge=1,
        le=10,
        description="Number of top contributing features to include in explanation (1-10)",
    ),
):
    """
    Predict fraud for a raw transaction.

    The API accepts raw transaction data and automatically applies feature engineering
    using the production FraudFeatureTransformer pipeline before making predictions.

    **Threshold Strategies:**
    - `optimal_f1`: Best precision-recall balance (F1 score optimized) - **RECOMMENDED DEFAULT**
    - `target_performance`: Max recall while maintaining >=70% precision - **RECOMMENDED FOR PRODUCTION**
    - `conservative_90pct_recall`: Catches 90% of fraud (more false positives)
    - `balanced_85pct_recall`: Targets 85% recall with maximized precision
    - `aggressive_80pct_recall`: Targets 80% recall with highest precision (fewer false positives)

    **Explanation Parameters:**
    - `include_explanation`: Set to `true` to include SHAP-based explanation
    - `top_n`: Number of top risk-increasing features to return (default: 3, range: 1-10)

    **Understanding SHAP Explanations:**

    SHAP values are computed in **log-odds space**, not probability space. This means:
    - `contribution_log_odds`: The raw SHAP value (additive in log-odds space)
    - `contribution_probability`: Approximate probability shift for interpretability

    Example interpretation:
    - `contribution_log_odds: 2.6` means this feature adds 2.6 to the log-odds
    - `contribution_probability: 0.18` means this feature increases probability by ~18%

    The math: `base_log_odds + sum(all SHAP values) = final_log_odds`, then
    `final_probability = 1 / (1 + exp(-final_log_odds))`

    **Returns:**
    - Fraud prediction (True/False)
    - Fraud probability (0.0-1.0)
    - Risk level (low/medium/high)
    - Processing metadata
    - Explanation (optional): Top features with log-odds and probability contributions
    """
    start_time = time.time()

    try:
        # Validate model and transformer are loaded
        if model is None or transformer is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model or transformer not loaded. Please check server logs.",
            )

        # Generate transaction ID
        transaction_id = str(uuid.uuid4())

        # Convert request to DataFrame for transformer
        transaction_dict = transaction.model_dump()
        transaction_df = pd.DataFrame([transaction_dict])

        # Apply feature engineering transformer
        engineered_features = transformer.transform(transaction_df)

        # Get prediction probability
        fraud_probability = float(model.predict_proba(engineered_features)[0, 1])

        # Apply threshold strategy
        threshold_info = threshold_config["optimized_thresholds"][threshold_strategy]
        threshold_value = threshold_info["threshold"]
        is_fraud = fraud_probability >= threshold_value

        # Determine risk level
        if fraud_probability >= 0.7:
            risk_level = "high"
        elif fraud_probability >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Generate explanation if requested
        explanation_response = None
        if include_explanation and explainer is not None:
            # Apply the pipeline's preprocessor to convert categorical features to numeric
            # This is needed because XGBoost's DMatrix requires numeric input
            preprocessor = model.named_steps["preprocessor"]
            preprocessed_features = preprocessor.transform(engineered_features)
            explanation_result = explainer.explain(
                preprocessed_features,
                top_n=top_n,
                only_positive=True,  # Only show features that increase fraud risk
            )
            explanation_response = ExplanationResponse(
                top_contributors=[
                    FeatureContributionResponse(
                        feature=contrib.feature,
                        display_name=contrib.display_name,
                        value=contrib.value,
                        contribution_log_odds=contrib.contribution_log_odds,
                        contribution_probability=contrib.contribution_probability,
                    )
                    for contrib in explanation_result.top_contributors
                ],
                base_fraud_rate=explanation_result.base_fraud_rate,
                final_fraud_probability=explanation_result.final_fraud_probability,
                explanation_method=explanation_result.explanation_method,
            )

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Log prediction
        logger.info(
            f"Prediction: ID={transaction_id}, "
            f"Fraud={is_fraud}, "
            f"Prob={fraud_probability:.4f}, "
            f"Risk={risk_level}, "
            f"Explained={include_explanation}, "
            f"Time={processing_time:.2f}ms"
        )

        return PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=is_fraud,
            fraud_probability=fraud_probability,
            risk_level=risk_level,
            threshold_used=threshold_strategy,
            threshold_value=threshold_value,
            model_version=model_metadata["model_info"]["version"],
            processing_time_ms=processing_time,
            explanation=explanation_response,
        )

    except KeyError as e:
        logger.error(f"Missing feature in request: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required feature: {e}",
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, detail=str(exc), timestamp=datetime.now().isoformat()
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
        ).model_dump(),
    )


if __name__ == "__main__":
    import os
    import uvicorn

    # Use PORT from environment (Cloud Run) or default to 8000 (local)
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
