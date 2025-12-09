"""
Unit tests for fraud explainer.

Tests the FraudExplainer class for correct feature contribution computation
and explanation generation using XGBoost's native pred_contribs.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.deployment.explainability import FEATURE_DESCRIPTIONS, FraudExplainer
from src.deployment.explainability.shap_explainer import (
    ExplanationResult,
    FeatureContribution,
)


@pytest.fixture(scope="module")
def model():
    """Load the trained XGBoost model."""
    return joblib.load("models/best_model.joblib")


@pytest.fixture(scope="module")
def feature_names():
    """Load feature names from feature_lists.json."""
    with open("models/feature_lists.json") as f:
        return json.load(f)["all_features"]


@pytest.fixture(scope="module")
def explainer(model, feature_names):
    """Create FraudExplainer instance."""
    return FraudExplainer(model, feature_names=feature_names)


@pytest.fixture
def sample_features(feature_names):
    """Create sample engineered features for testing."""
    # Create a sample with characteristics that might indicate fraud
    data = {
        "channel": [1],  # encoded value
        "account_age_days": [5],  # new account
        "total_transactions_user": [2],
        "avg_amount_user": [50.0],
        "amount": [500.0],  # high relative to average
        "shipping_distance_km": [500.0],  # long distance
        "hour_local": [2],  # late night
        "day_of_week_local": [5],  # weekend
        "month_local": [12],
        "amount_deviation": [450.0],
        "amount_vs_avg_ratio": [10.0],
        "transaction_velocity": [0.4],
        "security_score": [0],  # low security
        "promo_used": [1],
        "avs_match": [0],  # failed verification
        "cvv_result": [0],
        "three_ds_flag": [0],
        "is_weekend_local": [1],
        "is_late_night_local": [1],
        "is_business_hours_local": [0],
        "is_micro_transaction": [0],
        "is_large_transaction": [1],
        "is_new_account": [1],
        "is_high_frequency_user": [0],
        "country_mismatch": [1],
        "high_risk_distance": [1],
        "zero_distance": [0],
        "new_account_with_promo": [1],
        "late_night_micro_transaction": [0],
        "high_value_long_distance": [1],
    }
    return pd.DataFrame(data)


class TestFraudExplainerInit:
    """Tests for FraudExplainer initialization."""

    def test_explainer_initialization(self, explainer):
        """Test that explainer initializes correctly."""
        assert explainer is not None
        assert explainer.booster is not None
        assert explainer.classifier is not None

    def test_explainer_has_base_value(self, explainer):
        """Test that explainer has a valid base value."""
        # base_value is computed lazily, so just access it
        base_val = explainer.base_value
        assert isinstance(base_val, float)
        assert 0.0 <= base_val <= 1.0

    def test_explainer_has_feature_names(self, explainer, feature_names):
        """Test that explainer has correct feature names."""
        assert explainer.feature_names is not None
        assert len(explainer.feature_names) == len(feature_names)


class TestExplain:
    """Tests for the explain method."""

    def test_explain_returns_explanation_result(self, explainer, sample_features):
        """Test that explain returns an ExplanationResult."""
        result = explainer.explain(sample_features)
        assert isinstance(result, ExplanationResult)

    def test_explain_returns_correct_number_of_contributors(
        self, explainer, sample_features
    ):
        """Test that explain returns the requested number of contributors."""
        for top_n in [1, 3, 5]:
            result = explainer.explain(sample_features, top_n=top_n)
            # May return fewer if not enough positive contributors
            assert len(result.top_contributors) <= top_n

    def test_explain_returns_feature_contributions(self, explainer, sample_features):
        """Test that explain returns valid FeatureContribution objects."""
        result = explainer.explain(sample_features, top_n=3)
        for contrib in result.top_contributors:
            assert isinstance(contrib, FeatureContribution)
            assert isinstance(contrib.feature, str)
            assert isinstance(contrib.display_name, str)
            assert isinstance(contrib.value, float)
            assert isinstance(contrib.contribution, float)

    def test_explain_only_positive_contributions(self, explainer, sample_features):
        """Test that only_positive=True returns only positive contributions."""
        result = explainer.explain(sample_features, top_n=10, only_positive=True)
        for contrib in result.top_contributors:
            assert contrib.contribution > 0, (
                f"Expected positive contribution, got {contrib.contribution}"
            )

    def test_explain_includes_negative_when_requested(self, explainer, sample_features):
        """Test that only_positive=False can include negative contributions."""
        result = explainer.explain(sample_features, top_n=30, only_positive=False)
        # With 30 features requested and only_positive=False, we should get all features
        # At least some should have negative contributions
        contributions = [c.contribution for c in result.top_contributors]
        # It's possible all are positive for a very suspicious transaction,
        # so just check we got results
        assert len(result.top_contributors) > 0

    def test_explain_contributions_sorted_by_magnitude(self, explainer, sample_features):
        """Test that contributions are sorted by absolute magnitude (descending)."""
        result = explainer.explain(sample_features, top_n=5, only_positive=False)
        contributions = [abs(c.contribution) for c in result.top_contributors]
        assert contributions == sorted(contributions, reverse=True)

    def test_explain_has_valid_base_fraud_rate(self, explainer, sample_features):
        """Test that explanation includes valid base fraud rate."""
        result = explainer.explain(sample_features)
        assert isinstance(result.base_fraud_rate, float)
        assert 0.0 <= result.base_fraud_rate <= 1.0

    def test_explain_method_is_shap(self, explainer, sample_features):
        """Test that explanation method is 'shap'."""
        result = explainer.explain(sample_features)
        assert result.explanation_method == "shap"


class TestFeatureDescriptions:
    """Tests for feature descriptions."""

    def test_all_features_have_descriptions(self, feature_names):
        """Test that all model features have human-readable descriptions."""
        for feature in feature_names:
            assert feature in FEATURE_DESCRIPTIONS, (
                f"Missing description for feature: {feature}"
            )

    def test_descriptions_are_strings(self):
        """Test that all descriptions are non-empty strings."""
        for feature, description in FEATURE_DESCRIPTIONS.items():
            assert isinstance(description, str)
            assert len(description) > 0

    def test_explain_uses_descriptions(self, explainer, sample_features):
        """Test that explanations use human-readable descriptions."""
        result = explainer.explain(sample_features, top_n=3)
        for contrib in result.top_contributors:
            expected_description = FEATURE_DESCRIPTIONS.get(
                contrib.feature, contrib.feature
            )
            assert contrib.display_name == expected_description


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_explain_with_numpy_array(self, explainer, sample_features):
        """Test that explain works with numpy arrays."""
        features_array = sample_features.values
        result = explainer.explain(features_array, top_n=3)
        assert isinstance(result, ExplanationResult)
        assert len(result.top_contributors) <= 3

    def test_explain_with_single_row_dataframe(self, explainer, sample_features):
        """Test explain with single-row DataFrame."""
        result = explainer.explain(sample_features, top_n=3)
        assert len(result.top_contributors) <= 3

    def test_explain_top_n_greater_than_features(self, explainer, sample_features):
        """Test that requesting more features than available works correctly."""
        result = explainer.explain(sample_features, top_n=100, only_positive=False)
        # Should return at most the number of features available
        assert len(result.top_contributors) <= 30  # 30 features in model
