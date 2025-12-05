"""
Training configuration for fraud detection project.

Centralizes training parameters like CV strategy and threshold optimization
to ensure consistency across all scripts (bias_variance_analysis.py, train.py).
"""

from sklearn.model_selection import StratifiedKFold


class TrainingConfig:
    """Configuration for model training."""

    # Cross-validation strategy
    CV_FOLDS = 4
    CV_SHUFFLE = True

    # Threshold optimization targets (recall levels)
    THRESHOLD_TARGETS = {
        "conservative_90pct_recall": 0.90,
        "balanced_85pct_recall": 0.85,
        "aggressive_80pct_recall": 0.80
    }

    # Early stopping configuration (for XGBoost)
    EARLY_STOPPING_ROUNDS = 50
    EVAL_METRIC = "aucpr"

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
