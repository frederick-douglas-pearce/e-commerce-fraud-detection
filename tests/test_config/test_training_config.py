"""
Tests for src/config/training_config.py

Tests TrainingConfig class
"""

import pytest
from sklearn.model_selection import StratifiedKFold
from src.config.training_config import TrainingConfig


class TestTrainingConfig:
    """Tests for TrainingConfig class"""

    def test_get_cv_strategy_returns_stratified_kfold(self):
        """Test that get_cv_strategy returns StratifiedKFold"""
        cv = TrainingConfig.get_cv_strategy()
        assert isinstance(cv, StratifiedKFold)

    def test_get_cv_strategy_correct_n_splits(self):
        """Test that CV strategy has correct number of splits"""
        cv = TrainingConfig.get_cv_strategy()
        assert cv.n_splits == 4

    def test_get_cv_strategy_with_random_seed(self):
        """Test that CV strategy uses provided random seed"""
        cv = TrainingConfig.get_cv_strategy(random_seed=42)
        assert cv.random_state == 42

    def test_get_cv_strategy_default_random_seed(self):
        """Test that CV strategy uses default random seed when not provided"""
        cv = TrainingConfig.get_cv_strategy()
        assert cv.random_state == 1

    def test_get_cv_strategy_shuffle_enabled(self):
        """Test that CV strategy has shuffle enabled"""
        cv = TrainingConfig.get_cv_strategy()
        assert cv.shuffle is True

    def test_get_threshold_targets_returns_dict(self):
        """Test that get_threshold_targets returns a dictionary"""
        targets = TrainingConfig.get_threshold_targets()
        assert isinstance(targets, dict)

    def test_get_threshold_targets_has_expected_keys(self):
        """Test that threshold targets dict has expected keys"""
        targets = TrainingConfig.get_threshold_targets()
        assert 'conservative_90pct_recall' in targets
        assert 'balanced_85pct_recall' in targets
        assert 'aggressive_80pct_recall' in targets

    def test_get_threshold_targets_has_correct_values(self):
        """Test that threshold targets have correct recall values"""
        targets = TrainingConfig.get_threshold_targets()
        assert targets['conservative_90pct_recall'] == 0.90
        assert targets['balanced_85pct_recall'] == 0.85
        assert targets['aggressive_80pct_recall'] == 0.80

    def test_cv_folds_constant(self):
        """Test that CV_FOLDS constant is defined"""
        assert hasattr(TrainingConfig, 'CV_FOLDS')
        assert TrainingConfig.CV_FOLDS == 4
