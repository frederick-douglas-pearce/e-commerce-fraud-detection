"""
Explainability module for fraud detection model.

Provides SHAP-based explanations for fraud predictions.
"""

from .feature_descriptions import FEATURE_DESCRIPTIONS
from .shap_explainer import FraudExplainer

__all__ = ["FraudExplainer", "FEATURE_DESCRIPTIONS"]
