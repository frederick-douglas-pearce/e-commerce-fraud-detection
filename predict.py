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
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Fraud Detection API",
    description="Real-time fraud prediction service using XGBoost machine learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global variables for model and configuration
model = None
threshold_config = None
model_metadata = None
feature_lists = None
startup_time = None


# Pydantic models for request/response validation
class TransactionRequest(BaseModel):
    """Transaction data for fraud prediction."""

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

    # Binary flags
    promo_used: int = Field(..., description="Promo code used (0 or 1)", ge=0, le=1)
    avs_match: int = Field(..., description="AVS match result (0 or 1)", ge=0, le=1)
    cvv_result: int = Field(..., description="CVV verification result (0 or 1)", ge=0, le=1)
    three_ds_flag: int = Field(..., description="3D Secure flag (0 or 1)", ge=0, le=1)

    # Geographic and temporal features
    shipping_distance_km: float = Field(..., description="Shipping distance in km", ge=0.0)
    transaction_hour: int = Field(..., description="Transaction hour (0-23)", ge=0, le=23)
    transaction_day_of_week: int = Field(..., description="Day of week (0-6, Mon-Sun)", ge=0, le=6)
    is_weekend: int = Field(..., description="Weekend indicator (0 or 1)", ge=0, le=1)
    is_night: int = Field(..., description="Night transaction indicator (0 or 1)", ge=0, le=1)

    # Derived features (computed in preprocessing)
    country_mismatch: int = Field(..., description="Country mismatch indicator (0 or 1)", ge=0, le=1)
    high_value_txn: int = Field(..., description="High value transaction indicator (0 or 1)", ge=0, le=1)
    new_user: int = Field(..., description="New user indicator (0 or 1)", ge=0, le=1)
    low_security: int = Field(..., description="Low security indicator (0 or 1)", ge=0, le=1)

    # Risk scores (if available from preprocessing)
    amount_z_user: float = Field(
        ..., description="Amount z-score relative to user's history"
    )
    txn_velocity_1h: int = Field(
        ..., description="Number of transactions in last hour", ge=0
    )
    txn_velocity_24h: int = Field(
        ..., description="Number of transactions in last 24 hours", ge=0
    )

    # Card and device features
    card_type: str = Field(..., description="Card type (e.g., credit, debit)")
    device_type: str = Field(..., description="Device type (e.g., mobile, desktop)")
    ip_country: str = Field(..., description="IP country (2-letter code)", min_length=2, max_length=2)
    email_domain: str = Field(..., description="Email domain")

    @field_validator("country", "bin_country", "ip_country")
    @classmethod
    def validate_country_code(cls, v: str) -> str:
        """Validate country code is uppercase."""
        return v.upper()

    model_config = {"json_schema_extra": {"example": {"user_id": 12345, "account_age_days": 180, "total_transactions_user": 25, "avg_amount_user": 250.50, "amount": 850.75, "country": "US", "bin_country": "US", "channel": "web", "merchant_category": "retail", "promo_used": 0, "avs_match": 1, "cvv_result": 1, "three_ds_flag": 1, "shipping_distance_km": 12.5, "transaction_hour": 14, "transaction_day_of_week": 2, "is_weekend": 0, "is_night": 0, "country_mismatch": 0, "high_value_txn": 1, "new_user": 0, "low_security": 0, "amount_z_user": 2.3, "txn_velocity_1h": 1, "txn_velocity_24h": 3, "card_type": "credit", "device_type": "desktop", "ip_country": "US", "email_domain": "gmail.com"}}}


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

    model_config = {"json_schema_extra": {"example": {"transaction_id": "550e8400-e29b-41d4-a716-446655440000", "is_fraud": False, "fraud_probability": 0.12, "risk_level": "low", "threshold_used": "balanced_85pct_recall", "threshold_value": 0.35, "model_version": "1.0", "processing_time_ms": 15.3}}}


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
    features_required: list[str]


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str
    timestamp: str


# Model loading and initialization
@app.on_event("startup")
async def load_model_artifacts():
    """Load model and configuration files on startup."""
    global model, threshold_config, model_metadata, feature_lists, startup_time

    startup_time = datetime.now()
    logger.info("Starting E-Commerce Fraud Detection API...")

    try:
        models_dir = Path("models")

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

        logger.info("=" * 80)
        logger.info("API READY FOR REQUESTS")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Failed to load model artifacts: {e}")
        raise


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
    """
    if model_metadata is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata not loaded",
        )

    return ModelInfoResponse(
        model_name=model_metadata["model_info"]["name"],
        version=model_metadata["model_info"]["version"],
        training_date=model_metadata["model_info"]["training_date"],
        algorithm=model_metadata["model_info"]["algorithm"],
        performance=model_metadata["performance"]["test_set"],
        threshold_strategies=threshold_config,
        features_required=feature_lists["all_features"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict_fraud(
    transaction: TransactionRequest,
    threshold_strategy: Literal[
        "conservative_90pct_recall", "balanced_85pct_recall", "aggressive_80pct_recall"
    ] = "balanced_85pct_recall",
):
    """
    Predict fraud for a transaction.

    **Threshold Strategies:**
    - `conservative_90pct_recall`: Catches 90% of fraud (more false positives)
    - `balanced_85pct_recall`: Balanced approach (default)
    - `aggressive_80pct_recall`: Fewer false positives (may miss some fraud)

    **Returns:**
    - Fraud prediction (True/False)
    - Fraud probability (0.0-1.0)
    - Risk level (low/medium/high)
    - Processing metadata
    """
    start_time = time.time()

    try:
        # Validate model is loaded
        if model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please check server logs.",
            )

        # Generate transaction ID
        transaction_id = str(uuid.uuid4())

        # Convert transaction to feature array (must match training feature order)
        feature_dict = transaction.model_dump()

        # Create feature array in correct order
        feature_array = np.array(
            [[feature_dict[feat] for feat in feature_lists["all_features"]]]
        )

        # Get prediction probability
        fraud_probability = float(model.predict_proba(feature_array)[0, 1])

        # Apply threshold strategy
        threshold_info = threshold_config[threshold_strategy]
        threshold_value = threshold_info["threshold"]
        is_fraud = fraud_probability >= threshold_value

        # Determine risk level
        if fraud_probability >= 0.7:
            risk_level = "high"
        elif fraud_probability >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms

        # Log prediction
        logger.info(
            f"Prediction: ID={transaction_id}, "
            f"Fraud={is_fraud}, "
            f"Prob={fraud_probability:.4f}, "
            f"Risk={risk_level}, "
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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
