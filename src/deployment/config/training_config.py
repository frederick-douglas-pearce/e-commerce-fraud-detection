"""
Training configuration for fraud detection project.

Centralizes training parameters like CV strategy and threshold optimization
to ensure consistency across all scripts (bias_variance_analysis.py, train.py).

All default values are loaded from deployment_defaults.json (single source of truth).
"""

import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


# Load defaults from JSON file (single source of truth)
_CONFIG_PATH = Path(__file__).parent / "deployment_defaults.json"
with open(_CONFIG_PATH) as f:
    _DEFAULTS = json.load(f)

_TRAINING_DEFAULTS = _DEFAULTS["training"]
_DEPLOYMENT_DEFAULTS = _DEFAULTS["deployment"]


class TrainingConfig:
    """Configuration for model training."""

    # Cross-validation strategy
    CV_FOLDS: int = _TRAINING_DEFAULTS["cv_folds"]
    CV_SHUFFLE: bool = _TRAINING_DEFAULTS["cv_shuffle"]

    # Threshold optimization targets (recall levels) - from deployment section
    THRESHOLD_TARGETS: dict = _DEPLOYMENT_DEFAULTS["threshold_targets"].copy()

    @classmethod
    def get_cv_strategy(cls, random_seed: int = 1):
        """Get the cross-validation strategy.

        Args:
            random_seed: Random seed for reproducibility

        Returns:
            StratifiedKFold object configured for cross-validation
        """
        return StratifiedKFold(
            n_splits=cls.CV_FOLDS,
            shuffle=cls.CV_SHUFFLE,
            random_state=random_seed
        )

    @classmethod
    def get_threshold_targets(cls) -> dict:
        """Get threshold optimization targets.

        Returns:
            Dictionary mapping threshold names to target recall values
        """
        return cls.THRESHOLD_TARGETS.copy()

    @classmethod
    def get_cv_config(cls) -> dict:
        """Get cross-validation configuration as a dictionary.

        Returns:
            Dictionary with CV parameters
        """
        return {
            'n_splits': cls.CV_FOLDS,
            'shuffle': cls.CV_SHUFFLE,
            'strategy': 'StratifiedKFold'
        }
