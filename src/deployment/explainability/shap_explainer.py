"""
Explainer for fraud detection model predictions.

Uses XGBoost's native pred_contribs feature to provide local (per-prediction)
explanations of fraud scores. This approach is compatible with all XGBoost
versions and provides SHAP-like explanations efficiently.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .feature_descriptions import get_feature_description


@dataclass
class FeatureContribution:
    """A single feature's contribution to the fraud prediction.

    SHAP values are computed in log-odds space, where they are additive:
        base_log_odds + sum(all SHAP values) = final_log_odds
        final_probability = 1 / (1 + exp(-final_log_odds))

    The contribution_log_odds is the raw SHAP value. The contribution_probability
    is an approximation of how much this feature shifts the probability.
    """

    feature: str  # Technical feature name
    display_name: str  # Human-readable name
    value: float  # Actual feature value for this prediction
    contribution_log_odds: float  # SHAP value in log-odds space (additive)
    contribution_probability: float  # Approximate probability shift (for interpretability)


@dataclass
class ExplanationResult:
    """Result of explaining a fraud prediction.

    The base_fraud_rate is the actual fraud rate in the training data (~2.2%),
    representing the prior probability before considering transaction features.
    """

    top_contributors: list[FeatureContribution]
    base_fraud_rate: float  # Training data fraud rate (prior probability)
    final_fraud_probability: float  # Model's predicted fraud probability
    explanation_method: str = "shap"


class FraudExplainer:
    """
    Explainer for fraud detection predictions using XGBoost native contributions.

    Uses XGBoost's pred_contribs feature which computes SHAP values natively.
    This is more stable across XGBoost versions than the SHAP library's TreeExplainer.

    SHAP Value Interpretation:
    --------------------------
    SHAP values from XGBoost are in LOG-ODDS space, not probability space.
    This means they are additive in log-odds:

        base_log_odds + sum(all SHAP values) = final_log_odds
        final_probability = sigmoid(final_log_odds) = 1 / (1 + exp(-final_log_odds))

    A SHAP value of +2.0 in log-odds means the feature increases the log-odds by 2,
    which could shift probability from 10% to 60% (not a +200% change in probability).

    For interpretability, we also compute an approximate probability contribution
    using the marginal effect at the current prediction point.
    """

    # Actual fraud rate in training data (from model_metadata.json)
    TRAINING_FRAUD_RATE = 0.022

    def __init__(self, model: Pipeline, feature_names: list[str] | None = None):
        """
        Initialize the explainer with a trained model.

        Args:
            model: Trained sklearn Pipeline with XGBClassifier
            feature_names: List of feature names in order. If None, will
                          attempt to extract from model.
        """
        self.model = model

        # Extract the XGBoost classifier from the pipeline
        if hasattr(model, "named_steps") and "classifier" in model.named_steps:
            self.classifier = model.named_steps["classifier"]
        else:
            # Assume model is the classifier directly
            self.classifier = model

        # Get the underlying booster for native SHAP computation
        self.booster = self.classifier.get_booster()

        # Get feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(self.classifier, "feature_names_in_"):
            self.feature_names = list(self.classifier.feature_names_in_)
        else:
            self.feature_names = None

        # Store base log-odds for probability conversion
        self._base_log_odds = None

    @property
    def base_value(self) -> float:
        """Get the base fraud rate (training data fraud rate)."""
        return self.TRAINING_FRAUD_RATE

    @property
    def base_log_odds(self) -> float:
        """Get the model's base log-odds value."""
        if self._base_log_odds is None:
            import xgboost as xgb
            dummy_data = np.zeros((1, len(self.feature_names) if self.feature_names else 30))
            dmatrix = xgb.DMatrix(dummy_data)
            contribs = self.booster.predict(dmatrix, pred_contribs=True)
            self._base_log_odds = float(contribs[0, -1])
        return self._base_log_odds

    def _log_odds_to_probability(self, log_odds: float) -> float:
        """Convert log-odds to probability using sigmoid function."""
        return float(1 / (1 + np.exp(-log_odds)))

    def _compute_probability_contribution(
        self, shap_log_odds: float, final_log_odds: float
    ) -> float:
        """
        Compute approximate probability contribution for a single feature.

        Uses the marginal method: compute probability with and without this feature's
        contribution to estimate its impact on the final probability.

        Args:
            shap_log_odds: SHAP value in log-odds space
            final_log_odds: Final prediction in log-odds space

        Returns:
            Approximate probability shift caused by this feature
        """
        prob_with = self._log_odds_to_probability(final_log_odds)
        prob_without = self._log_odds_to_probability(final_log_odds - shap_log_odds)
        return prob_with - prob_without

    def explain(
        self,
        features: pd.DataFrame | np.ndarray,
        top_n: int = 3,
        only_positive: bool = True,
    ) -> ExplanationResult:
        """
        Explain a single prediction.

        Args:
            features: Engineered features for one transaction (1 row).
                     Should be the output of the FraudFeatureTransformer.
            top_n: Number of top contributing features to return
            only_positive: If True, only return features that increase fraud risk

        Returns:
            ExplanationResult with top contributing features, including both
            log-odds contributions (additive) and probability contributions
            (approximate, for interpretability)
        """
        import xgboost as xgb

        # Ensure we have a 2D array
        if isinstance(features, pd.DataFrame):
            feature_values = features.values
            feature_names = list(features.columns)
        else:
            feature_values = np.atleast_2d(features)
            feature_names = self.feature_names or [f"feature_{i}" for i in range(feature_values.shape[1])]

        # Create DMatrix for XGBoost
        dmatrix = xgb.DMatrix(feature_values)

        # Get SHAP values using XGBoost's native pred_contribs
        # This returns an array of shape (n_samples, n_features + 1)
        # where the last column is the bias term
        contribs = self.booster.predict(dmatrix, pred_contribs=True)

        # Extract feature contributions (excluding bias which is last column)
        shap_vals = contribs[0, :-1]
        bias = contribs[0, -1]

        # Compute final log-odds and probability
        final_log_odds = float(bias + np.sum(shap_vals))
        final_probability = self._log_odds_to_probability(final_log_odds)

        # Create list of (feature_name, value, shap_log_odds, shap_prob) tuples
        contributions = []
        for i, (name, shap_val) in enumerate(zip(feature_names, shap_vals)):
            feat_value = float(feature_values[0, i])
            shap_log_odds = float(shap_val)
            shap_prob = self._compute_probability_contribution(shap_log_odds, final_log_odds)
            contributions.append((name, feat_value, shap_log_odds, shap_prob))

        # Filter to positive contributions if requested (based on log-odds)
        if only_positive:
            contributions = [(n, v, s_lo, s_p) for n, v, s_lo, s_p in contributions if s_lo > 0]

        # Sort by absolute log-odds contribution (descending)
        contributions.sort(key=lambda x: abs(x[2]), reverse=True)

        # Take top N
        top_contributions = contributions[:top_n]

        # Convert to FeatureContribution objects
        result_contributions = [
            FeatureContribution(
                feature=name,
                display_name=get_feature_description(name),
                value=value,
                contribution_log_odds=shap_log_odds,
                contribution_probability=shap_prob,
            )
            for name, value, shap_log_odds, shap_prob in top_contributions
        ]

        return ExplanationResult(
            top_contributors=result_contributions,
            base_fraud_rate=self.base_value,
            final_fraud_probability=final_probability,
            explanation_method="shap",
        )

    def explain_preprocessed(
        self,
        raw_features: pd.DataFrame,
        preprocessor: Any,
        top_n: int = 3,
        only_positive: bool = True,
    ) -> ExplanationResult:
        """
        Explain a prediction by first applying the preprocessor.

        This is useful when you have raw input that needs to go through
        the pipeline's preprocessor before explanation.

        Args:
            raw_features: Raw features (before preprocessing)
            preprocessor: The preprocessor step from the pipeline
            top_n: Number of top contributing features to return
            only_positive: If True, only return features that increase fraud risk

        Returns:
            ExplanationResult with top contributing features
        """
        # Apply preprocessor
        processed = preprocessor.transform(raw_features)
        if isinstance(processed, np.ndarray):
            processed = pd.DataFrame(processed, columns=self.feature_names)
        return self.explain(processed, top_n=top_n, only_positive=only_positive)
