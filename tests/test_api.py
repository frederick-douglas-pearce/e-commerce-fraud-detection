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


@pytest.fixture(scope="module")
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


class TestAPIRoot:
    """Test root endpoint."""

    def test_root_endpoint(self, client):
        """Test GET / returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "E-Commerce Fraud Detection API" in data["name"]
        assert "endpoints" in data
        assert "predict" in data["endpoints"]

    def test_root_has_documentation_links(self, client):
        """Test root endpoint includes documentation links."""
        response = client.get("/")
        data = response.json()
        assert "documentation" in data
        assert "/docs" in data["documentation"]


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_success(self, client):
        """Test health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_health_has_model_version(self, client):
        """Test health endpoint returns model version."""
        response = client.get("/health")
        data = response.json()
        assert "model_version" in data
        assert data["model_version"] is not None


class TestModelInfoEndpoint:
    """Test model information endpoint."""

    def test_model_info_success(self, client):
        """Test model/info returns model metadata."""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_name" in data
        assert "version" in data
        assert "performance" in data
        assert "threshold_strategies" in data

    def test_model_info_has_performance_metrics(self, client):
        """Test model info includes performance metrics."""
        response = client.get("/model/info")
        data = response.json()
        performance = data["performance"]
        assert "pr_auc" in performance
        assert "precision" in performance
        assert "recall" in performance
        assert performance["pr_auc"] > 0

    def test_model_info_has_threshold_strategies(self, client):
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

    def test_predict_success(self, client, valid_transaction):
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

    def test_predict_has_required_fields(self, client, valid_transaction):
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

    def test_predict_risk_level_valid(self, client, valid_transaction):
        """Test risk level is one of the valid values."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        assert data["risk_level"] in ["low", "medium", "high"]

    def test_predict_with_balanced_threshold(self, client, valid_transaction):
        """Test prediction with balanced threshold strategy."""
        response = client.post(
            "/predict?threshold_strategy=balanced_85pct_recall",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "balanced_85pct_recall"

    def test_predict_with_conservative_threshold(self, client, valid_transaction):
        """Test prediction with conservative threshold strategy."""
        response = client.post(
            "/predict?threshold_strategy=conservative_90pct_recall",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "conservative_90pct_recall"

    def test_predict_with_aggressive_threshold(self, client, valid_transaction):
        """Test prediction with aggressive threshold strategy."""
        response = client.post(
            "/predict?threshold_strategy=aggressive_80pct_recall",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "aggressive_80pct_recall"

    def test_predict_missing_required_field(self, client, valid_transaction):
        """Test prediction fails with missing required field."""
        incomplete_transaction = valid_transaction.copy()
        del incomplete_transaction["amount"]
        response = client.post("/predict", json=incomplete_transaction)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_invalid_amount(self, client, valid_transaction):
        """Test prediction fails with negative amount."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["amount"] = -100.0
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_invalid_country_code(self, client, valid_transaction):
        """Test prediction fails with invalid country code."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["country"] = "USA"  # Should be 2-letter code
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_invalid_channel(self, client, valid_transaction):
        """Test prediction fails with invalid channel."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["channel"] = "mobile"  # Should be "web" or "app"
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_invalid_timestamp(self, client, valid_transaction):
        """Test prediction fails with invalid timestamp format."""
        invalid_transaction = valid_transaction.copy()
        invalid_transaction["transaction_time"] = "2024-01-15"  # Missing time
        response = client.post("/predict", json=invalid_transaction)
        assert response.status_code == 422

    def test_predict_processing_time_reasonable(self, client, valid_transaction):
        """Test prediction processing time is reasonable."""
        response = client.post("/predict", json=valid_transaction)
        data = response.json()
        # Processing should be under 1 second (1000ms)
        assert data["processing_time_ms"] < 1000


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_docs_accessible(self, client):
        """Test OpenAPI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_accessible(self, client):
        """Test ReDoc documentation is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200

    def test_openapi_json_accessible(self, client):
        """Test OpenAPI JSON schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "paths" in data
        assert "/predict" in data["paths"]


class TestPredictExplanation:
    """Test fraud prediction explanation feature."""

    @pytest.fixture
    def valid_transaction(self):
        """Sample valid raw transaction for testing."""
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

    @pytest.fixture
    def suspicious_transaction(self):
        """Sample suspicious transaction likely to have positive SHAP contributions."""
        return {
            "user_id": 99999,
            "account_age_days": 5,  # new account
            "total_transactions_user": 2,
            "avg_amount_user": 50.0,
            "amount": 999.99,  # high amount
            "country": "US",
            "bin_country": "GB",  # country mismatch
            "channel": "web",
            "merchant_category": "electronics",
            "promo_used": 1,  # using promo
            "avs_match": 0,  # failed verification
            "cvv_result": 0,
            "three_ds_flag": 0,
            "shipping_distance_km": 500.0,  # long distance
            "transaction_time": "2024-01-15 02:30:00"  # late night
        }

    def test_predict_without_explanation(self, client, valid_transaction):
        """Test that explanation is not included by default."""
        response = client.post("/predict", json=valid_transaction)
        assert response.status_code == 200
        data = response.json()
        assert data.get("explanation") is None

    def test_predict_with_explanation_false(self, client, valid_transaction):
        """Test that explanation is not included when include_explanation=false."""
        response = client.post(
            "/predict?include_explanation=false",
            json=valid_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("explanation") is None

    def test_predict_with_explanation_true(self, client, suspicious_transaction):
        """Test that explanation is included when include_explanation=true."""
        response = client.post(
            "/predict?include_explanation=true",
            json=suspicious_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["explanation"] is not None
        assert "top_contributors" in data["explanation"]
        assert "base_fraud_rate" in data["explanation"]
        assert "explanation_method" in data["explanation"]

    def test_explanation_has_correct_structure(self, client, suspicious_transaction):
        """Test that explanation has the correct structure."""
        response = client.post(
            "/predict?include_explanation=true",
            json=suspicious_transaction,
        )
        data = response.json()
        explanation = data["explanation"]

        # Check base fields
        assert isinstance(explanation["base_fraud_rate"], float)
        assert explanation["explanation_method"] == "shap"

        # Check top_contributors structure
        assert isinstance(explanation["top_contributors"], list)
        if len(explanation["top_contributors"]) > 0:
            contrib = explanation["top_contributors"][0]
            assert "feature" in contrib
            assert "display_name" in contrib
            assert "value" in contrib
            assert "contribution" in contrib

    def test_explanation_default_top_n(self, client, suspicious_transaction):
        """Test that explanation returns default top 3 contributors."""
        response = client.post(
            "/predict?include_explanation=true",
            json=suspicious_transaction,
        )
        data = response.json()
        # Default is top 3, but may return fewer if not enough positive contributors
        assert len(data["explanation"]["top_contributors"]) <= 3

    def test_explanation_custom_top_n(self, client, suspicious_transaction):
        """Test that explanation respects custom top_n parameter."""
        response = client.post(
            "/predict?include_explanation=true&top_n=5",
            json=suspicious_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        # May return fewer if not enough positive contributors
        assert len(data["explanation"]["top_contributors"]) <= 5

    def test_explanation_top_n_min_value(self, client, suspicious_transaction):
        """Test that top_n=1 works correctly."""
        response = client.post(
            "/predict?include_explanation=true&top_n=1",
            json=suspicious_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["explanation"]["top_contributors"]) <= 1

    def test_explanation_top_n_max_value(self, client, suspicious_transaction):
        """Test that top_n=10 works correctly."""
        response = client.post(
            "/predict?include_explanation=true&top_n=10",
            json=suspicious_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["explanation"]["top_contributors"]) <= 10

    def test_explanation_top_n_too_small(self, client, valid_transaction):
        """Test that top_n < 1 is rejected."""
        response = client.post(
            "/predict?include_explanation=true&top_n=0",
            json=valid_transaction,
        )
        assert response.status_code == 422

    def test_explanation_top_n_too_large(self, client, valid_transaction):
        """Test that top_n > 10 is rejected."""
        response = client.post(
            "/predict?include_explanation=true&top_n=11",
            json=valid_transaction,
        )
        assert response.status_code == 422

    def test_explanation_contributions_are_positive(self, client, suspicious_transaction):
        """Test that all contributions are positive (increase fraud risk)."""
        response = client.post(
            "/predict?include_explanation=true&top_n=5",
            json=suspicious_transaction,
        )
        data = response.json()
        for contrib in data["explanation"]["top_contributors"]:
            assert contrib["contribution"] > 0, (
                f"Expected positive contribution for {contrib['feature']}, "
                f"got {contrib['contribution']}"
            )

    def test_explanation_with_threshold_strategy(self, client, suspicious_transaction):
        """Test that explanation works with different threshold strategies."""
        response = client.post(
            "/predict?include_explanation=true&threshold_strategy=conservative_90pct_recall",
            json=suspicious_transaction,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["threshold_used"] == "conservative_90pct_recall"
        assert data["explanation"] is not None

    def test_explanation_processing_time_reasonable(self, client, suspicious_transaction):
        """Test that explanation doesn't add too much processing time."""
        # Get baseline without explanation
        response_no_explain = client.post("/predict", json=suspicious_transaction)
        time_no_explain = response_no_explain.json()["processing_time_ms"]

        # Get time with explanation
        response_with_explain = client.post(
            "/predict?include_explanation=true",
            json=suspicious_transaction,
        )
        time_with_explain = response_with_explain.json()["processing_time_ms"]

        # Explanation should add reasonable overhead (less than 100ms typically)
        # Allow generous margin for CI environments
        assert time_with_explain < time_no_explain + 500


class TestErrorHandling:
    """Test API error handling."""

    def test_invalid_endpoint_404(self, client):
        """Test invalid endpoint returns 404."""
        response = client.get("/invalid-endpoint")
        assert response.status_code == 404

    def test_wrong_method_405(self, client):
        """Test wrong HTTP method returns 405."""
        response = client.get("/predict")  # Should be POST
        assert response.status_code == 405


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
