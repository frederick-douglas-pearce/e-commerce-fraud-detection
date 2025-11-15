"""
Integration tests for E-Commerce Fraud Detection API.

Tests all API endpoints, request/response validation, error handling, and model predictions.

The API accepts raw transaction data (15 fields) and automatically applies feature engineering
using the production FraudFeatureTransformer pipeline before making predictions.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app
from predict import app

# Create test client
client = TestClient(app)


class TestAPIRoot:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test GET / returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "E-Commerce Fraud Detection API" in data["name"]
        assert "endpoints" in data
        assert "predict" in data["endpoints"]

    def test_root_has_documentation_links(self):
        """Test root endpoint includes documentation links."""
        response = client.get("/")
        data = response.json()
        assert "documentation" in data
        assert "/docs" in data["documentation"]


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_has_model_version(self):
        """Test health endpoint returns model version."""
        response = client.get("/health")
        data = response.json()
        assert "model_version" in data
        assert data["model_version"] is not None


class TestModelInfoEndpoint:
    """Test model information endpoint."""

    def test_model_info_success(self):
        """Test model/info returns model metadata."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "performance" in data
        assert "threshold_strategies" in data

    def test_model_info_has_performance_metrics(self):
        """Test model info includes performance metrics."""
        response = client.get("/model/info")
        data = response.json()
        performance = data["performance"]
        assert "pr_auc" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert performance["pr_auc"] > 0

    def test_model_info_has_threshold_strategies(self):
        """Test model info includes all threshold strategies."""
        response = client.get("/model/info")
        data = response.json()
        strategies = data["threshold_strategies"]
        assert "conservative_90pct_recall" in strategies
        assert "balanced_85pct_recall" in strategies
        assert "aggressive_80pct_recall" in strategies


class TestPredictEndpoint:
    """Test fraud prediction endpoint."""

    @pytest.fixture
    def valid_transaction(self):
        """Sample valid raw transaction for testing.

        The API now accepts raw transaction data and automatically applies
        feature engineering using FraudFeatureTransformer.
        """
        return {
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

    def test_predict_success(self, valid_transaction):
        """Test successful fraud prediction."""
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()
        assert "transaction_id" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert data["fraud_probability"] >= 0.0
        assert data["fraud_probability"] <= 1.0

    def test_predict_has_required_fields(self, valid_transaction):
        """Test prediction response has all required fields."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        required_fields = [
            "transaction_id",
            "is_fraud",
            "fraud_probability",
            "risk_level",
            "threshold_used",
            "threshold_value",
            "model_version",
            "processing_time_ms",
        ]
        for field in required_fields:
            assert field in data

    def test_predict_risk_level_valid(self, valid_transaction):
        """Test risk level is one of the valid values."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        assert data["risk_level"] in ["low", "medium", "high"]

    def test_predict_with_balanced_threshold(self, valid_transaction):
        """Test prediction with balanced threshold strategy."""
        response = client.post(
            "/predict?threshold_strategy=balanced_85pct_recall",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "balanced_85pct_recall"

    def test_predict_with_conservative_threshold(self, valid_transaction):
        """Test prediction with conservative threshold strategy."""
        response = client.post(
            "/predict?threshold_strategy=conservative_90pct_recall",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "conservative_90pct_recall"

    def test_predict_with_aggressive_threshold(self, valid_transaction):
        """Test prediction with aggressive threshold strategy."""
        response = client.post(
            "/predict?threshold_strategy=aggressive_80pct_recall",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "aggressive_80pct_recall"

    def test_predict_missing_required_field(self, valid_transaction):
        """Test prediction fails with missing required field."""
        incomplete_transaction = valid_transaction.copy()
        del incomplete_transaction["amount"]
        response = client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_amount(self, valid_transaction):
        """Test prediction fails with negative amount."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["amount"] = -100.0
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_invalid_country_code(self, valid_transaction):
        """Test prediction fails with invalid country code."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["country"] = "USA"  # Should be 2-letter code
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_invalid_channel(self, valid_transaction):
        """Test prediction fails with invalid channel."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["channel"] = "mobile"  # Should be "web" or "app"
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_invalid_timestamp(self, valid_transaction):
        """Test prediction fails with invalid timestamp format."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["transaction_time"] = "2024-01-15"  # Missing time
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_processing_time_reasonable(self, valid_transaction):
        """Test prediction processing time is reasonable."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        # Processing should be under 1 second (1000ms)
        assert data["processing_time_ms"] < 1000


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_docs_accessible(self):
        """Test OpenAPI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self):
        """Test ReDoc documentation is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json_accessible(self):
        """Test OpenAPI JSON schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/predict" in data["paths"]


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_endpoint_404(self):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_wrong_method_405(self):
        """Test wrong HTTP method returns 405."""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
