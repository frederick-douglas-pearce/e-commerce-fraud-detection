"""
Locust Load Testing Configuration for Fraud Detection API

Simulates realistic user behavior with randomized transaction data.

Usage:
    # Start Locust web UI
    locust -f locustfile.py --host=http://localhost:8000

    # Run headless with 50 users, spawn rate 10/sec, run for 60 seconds
    locust -f locustfile.py --host=http://localhost:8000 --users 50 --spawn-rate 10 --run-time 60s --headless

    # Generate HTML report
    locust -f locustfile.py --host=http://localhost:8000 --users 100 --spawn-rate 10 --run-time 120s --headless --html=locust_report.html

Requirements:
    pip install locust
"""

import random
from locust import HttpUser, task, between


class FraudDetectionUser(HttpUser):
    """
    Simulated user for Fraud Detection API load testing.

    Behavior:
    - Waits 0.1-0.5 seconds between requests (realistic think time)
    - Sends prediction requests with randomized transaction data
    - Simulates real-world variation in transaction attributes
    - Tests all threshold strategies and explainability feature

    Task Weights (relative frequency):
    - Normal transactions: 10 (most common)
    - Suspicious transactions: 3
    - Predictions with SHAP explanation: 2
    - Health checks: 1
    - Model info requests: 1
    """

    wait_time = between(0.1, 0.5)  # Wait 100-500ms between requests

    # Realistic data distributions
    COUNTRIES = ["US", "GB", "CA", "AU", "DE", "FR", "JP", "SG", "BR", "MX"]
    CHANNELS = ["web", "app"]
    MERCHANT_CATEGORIES = [
        "retail",
        "travel",
        "digital",
        "grocery",
        "entertainment",
        "utilities",
    ]
    THRESHOLD_STRATEGIES = [
        "target_performance",
        "optimal_f1",
        "conservative_90pct_recall",
        "balanced_85pct_recall",
        "aggressive_80pct_recall",
    ]

    def _generate_transaction_time(self) -> str:
        """Generate a realistic transaction timestamp."""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return f"2025-01-15 {hour:02d}:{minute:02d}:{second:02d}"

    def on_start(self):
        """Called when a simulated user starts (optional: check health)."""
        # Verify API is healthy before starting load test
        response = self.client.get("/health")
        if response.status_code != 200:
            print("⚠️  Warning: API health check failed")

    def _build_normal_payload(self) -> dict:
        """Build payload for a typical legitimate transaction."""
        return {
            "user_id": random.randint(1000, 99999),
            "account_age_days": random.randint(30, 365),  # Established accounts
            "total_transactions_user": random.randint(10, 100),
            "avg_amount_user": round(random.uniform(50, 500), 2),
            "amount": round(random.uniform(20, 800), 2),
            "country": random.choice(self.COUNTRIES),
            "bin_country": random.choice(self.COUNTRIES),
            "channel": random.choice(self.CHANNELS),
            "merchant_category": random.choice(self.MERCHANT_CATEGORIES),
            "promo_used": random.randint(0, 1),
            "avs_match": 1,  # Legitimate transactions usually match
            "cvv_result": 1,
            "three_ds_flag": random.randint(0, 1),
            "shipping_distance_km": round(random.uniform(0, 100), 2),
            "transaction_time": self._generate_transaction_time(),
        }

    @task(10)  # Weight 10: Most common task
    def predict_normal_transaction(self):
        """Simulate a typical legitimate transaction."""
        payload = self._build_normal_payload()
        strategy = random.choice(self.THRESHOLD_STRATEGIES)

        with self.client.post(
            f"/predict?threshold_strategy={strategy}",
            json=payload,
            catch_response=True,
            name="/predict [normal]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate response structure
                if "fraud_probability" in data and "is_fraud" in data:
                    response.success()
                else:
                    response.failure("Invalid response structure")
            else:
                response.failure(f"HTTP {response.status_code}")

    def _build_suspicious_payload(self) -> dict:
        """Build payload for a potentially suspicious transaction."""
        return {
            "user_id": random.randint(1000, 99999),
            "account_age_days": random.randint(1, 30),  # New accounts
            "total_transactions_user": random.randint(1, 5),  # Few transactions
            "avg_amount_user": round(random.uniform(50, 200), 2),
            "amount": round(random.uniform(800, 2000), 2),  # High amount
            "country": random.choice(self.COUNTRIES),
            "bin_country": random.choice(self.COUNTRIES),  # Potentially mismatched
            "channel": random.choice(self.CHANNELS),
            "merchant_category": random.choice(self.MERCHANT_CATEGORIES),
            "promo_used": random.randint(0, 1),
            "avs_match": random.randint(0, 1),  # May not match
            "cvv_result": random.randint(0, 1),
            "three_ds_flag": 0,  # No 3D Secure
            "shipping_distance_km": round(random.uniform(100, 500), 2),  # Far shipping
            "transaction_time": self._generate_transaction_time(),
        }

    @task(3)  # Weight 3: Less common
    def predict_suspicious_transaction(self):
        """Simulate a potentially suspicious transaction."""
        payload = self._build_suspicious_payload()
        strategy = random.choice(self.THRESHOLD_STRATEGIES)

        with self.client.post(
            f"/predict?threshold_strategy={strategy}",
            json=payload,
            catch_response=True,
            name="/predict [suspicious]",
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Weight 1: Occasional health checks
    def check_health(self):
        """Check API health status."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("API unhealthy")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)  # Weight 1: Occasional model info requests
    def get_model_info(self):
        """Retrieve model information."""
        with self.client.get("/model/info", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)  # Weight 2: Some users request explanations
    def predict_with_explanation(self):
        """Request prediction with SHAP explanation (slower, tests explainability)."""
        payload = self._build_suspicious_payload()  # Use suspicious for more interesting explanations
        top_n = random.choice([3, 5])

        with self.client.post(
            f"/predict?include_explanation=true&top_n={top_n}",
            json=payload,
            catch_response=True,
            name="/predict [with explanation]",
        ) as response:
            if response.status_code == 200:
                data = response.json()
                # Validate explanation is present
                if data.get("explanation") and "top_contributors" in data["explanation"]:
                    response.success()
                else:
                    response.failure("Missing explanation in response")
            else:
                response.failure(f"HTTP {response.status_code}")


class StressTestUser(HttpUser):
    """
    Stress testing user with no wait time between requests.

    Use this class to test maximum throughput and identify breaking points.

    Usage:
        locust -f locustfile.py --user-classes StressTestUser --host=http://localhost:8000
    """

    wait_time = between(0, 0.1)  # Minimal wait time

    def _generate_transaction_time(self) -> str:
        """Generate a realistic transaction timestamp."""
        hour = random.randint(0, 23)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return f"2025-01-15 {hour:02d}:{minute:02d}:{second:02d}"

    @task
    def predict_rapid_fire(self):
        """Send rapid-fire prediction requests."""
        payload = {
            "user_id": random.randint(1000, 99999),
            "account_age_days": random.randint(1, 365),
            "total_transactions_user": random.randint(1, 100),
            "avg_amount_user": round(random.uniform(50, 500), 2),
            "amount": round(random.uniform(20, 2000), 2),
            "country": random.choice(["US", "GB", "CA"]),
            "bin_country": random.choice(["US", "GB", "CA"]),
            "channel": random.choice(["web", "app"]),
            "merchant_category": random.choice(["retail", "travel", "digital"]),
            "promo_used": random.randint(0, 1),
            "avs_match": random.randint(0, 1),
            "cvv_result": random.randint(0, 1),
            "three_ds_flag": random.randint(0, 1),
            "shipping_distance_km": round(random.uniform(0, 500), 2),
            "transaction_time": self._generate_transaction_time(),
        }

        self.client.post("/predict", json=payload, name="/predict [stress]")
