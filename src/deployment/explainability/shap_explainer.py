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
    """A single feature's contribution to the fraud prediction."""

    feature: str  # Technical feature name
    display_name: str  # Human-readable name
    value: float  # Actual feature value for this prediction
    contribution: float  # Contribution value (positive = increases fraud risk)


@dataclass
class ExplanationResult:
    """Result of explaining a fraud prediction."""

    top_contributors: list[FeatureContribution]
    base_fraud_rate: float  # Expected value (baseline fraud probability)
    explanation_method: str = "shap"


class FraudExplainer:
    """
    Explainer for fraud detection predictions using XGBoost native contributions.

    Uses XGBoost's pred_contribs feature which computes SHAP values natively.
    This is more stable across XGBoost versions than the SHAP library's TreeExplainer.
    """

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

        # Compute base value (expected value / bias term)
        # The base value is the last column in pred_contribs output
        # We'll compute it lazily on first explanation
        self._base_value = None

    @property
    def base_value(self) -> float:
        """Get the base value (expected prediction before any features)."""
        if self._base_value is None:
            # Compute base value by getting contributions for a dummy input
            # The bias term is constant regardless of input
            import xgboost as xgb

            dummy_data = np.zeros((1, len(self.feature_names) if self.feature_names else 30))
            dmatrix = xgb.DMatrix(dummy_data)
            contribs = self.booster.predict(dmatrix, pred_contribs=True)
            # Last column is the bias (base value) in log-odds space
            # Convert from log-odds to probability
            bias_log_odds = contribs[0, -1]
            self._base_value = float(1 / (1 + np.exp(-bias_log_odds)))
        return self._base_value

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
            ExplanationResult with top contributing features
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

        # Create list of (feature_name, value, shap_value) tuples
        contributions = []
        for i, (name, shap_val) in enumerate(zip(feature_names, shap_vals)):
            feat_value = float(feature_values[0, i])
            contributions.append((name, feat_value, float(shap_val)))

        # Filter to positive contributions if requested
        if only_positive:
            contributions = [(n, v, s) for n, v, s in contributions if s > 0]

        # Sort by absolute contribution (descending)
        contributions.sort(key=lambda x: abs(x[2]), reverse=True)

        # Take top N
        top_contributions = contributions[:top_n]

        # Convert to FeatureContribution objects
        result_contributions = [
            FeatureContribution(
                feature=name,
                display_name=get_feature_description(name),
                value=value,
                contribution=shap_val,
            )
            for name, value, shap_val in top_contributions
        ]

        return ExplanationResult(
            top_contributors=result_contributions,
            base_fraud_rate=self.base_value,
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
