"""
Tests for src/evaluation/thresholds.py

Tests optimize_thresholds function for threshold optimization.
"""

import pytest
import numpy as np
from unittest.mock import Mock
from src.evaluation.thresholds import optimize_thresholds
from src.config.training_config import TrainingConfig


@pytest.fixture
def sample_classification_data():
    """Create sample classification data for threshold optimization"""
    np.random.seed(42)

    # Generate synthetic data: 100 samples, 20% positive class
    n_samples = 100
    y_val = np.array([0] * 80 + [1] * 20)

    # Generate realistic probabilities
    # For true positives: higher probabilities
    # For true negatives: lower probabilities
    y_proba = np.zeros(n_samples)
    y_proba[:80] = np.random.beta(2, 5, 80)  # Negatives: lower probs
    y_proba[80:] = np.random.beta(5, 2, 20)  # Positives: higher probs

    return y_val, y_proba


@pytest.fixture
def mock_model_with_probabilities(sample_classification_data):
    """Create a mock model that returns probabilities"""
    y_val, y_proba = sample_classification_data

    model = Mock()
    model.predict_proba.return_value = np.column_stack([1 - y_proba, y_proba])

    return model


class TestOptimizeThresholds:
    """Tests for optimize_thresholds function"""

    def test_optimize_thresholds_returns_dict(self, mock_model_with_probabilities,
                                               sample_classification_data):
        """Test that optimize_thresholds returns a dictionary"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)  # Dummy features

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        assert isinstance(config, dict)

    def test_optimize_thresholds_uses_default_targets(self, mock_model_with_probabilities,
                                                       sample_classification_data):
        """Test that default recall targets from TrainingConfig are used plus optimal_f1"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        # Should have optimal_f1 + default thresholds from TrainingConfig
        default_targets = TrainingConfig.get_threshold_targets()
        expected_keys = set(['optimal_f1']) | set(default_targets.keys())
        assert set(config.keys()) == expected_keys

    def test_optimize_thresholds_custom_recall_targets(self, mock_model_with_probabilities,
                                                        sample_classification_data):
        """Test that custom recall targets are respected plus optimal_f1"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        custom_targets = {
            'custom_95pct': 0.95,
            'custom_80pct': 0.80
        }

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val,
            recall_targets=custom_targets,
            verbose=False
        )

        # Should have optimal_f1 + custom targets
        expected_keys = set(['optimal_f1']) | set(custom_targets.keys())
        assert set(config.keys()) == expected_keys

    def test_optimize_thresholds_config_structure(self, mock_model_with_probabilities,
                                                   sample_classification_data):
        """Test that each threshold config has expected structure"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        for threshold_name, threshold_config in config.items():
            assert 'threshold' in threshold_config
            assert 'precision' in threshold_config
            # optimal_f1 has 'recall' and 'f1', recall-targeted have 'achieved_recall' and 'target_recall'
            if threshold_name == 'optimal_f1':
                assert 'recall' in threshold_config
                assert 'f1' in threshold_config
            else:
                assert 'achieved_recall' in threshold_config
                assert 'target_recall' in threshold_config

    def test_optimize_thresholds_values_are_floats(self, mock_model_with_probabilities,
                                                    sample_classification_data):
        """Test that all values in threshold configs are floats"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        for threshold_name, threshold_config in config.items():
            assert isinstance(threshold_config['threshold'], float)
            assert isinstance(threshold_config['precision'], float)
            # optimal_f1 has 'recall' and 'f1', recall-targeted have 'achieved_recall' and 'target_recall'
            if threshold_name == 'optimal_f1':
                assert isinstance(threshold_config['recall'], float)
                assert isinstance(threshold_config['f1'], float)
            else:
                assert isinstance(threshold_config['achieved_recall'], float)
                assert isinstance(threshold_config['target_recall'], float)

    def test_optimize_thresholds_values_in_valid_range(self, mock_model_with_probabilities,
                                                        sample_classification_data):
        """Test that threshold, precision, recall are in valid range [0, 1]"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        for threshold_name, threshold_config in config.items():
            assert 0 <= threshold_config['threshold'] <= 1
            assert 0 <= threshold_config['precision'] <= 1
            # optimal_f1 has 'recall', recall-targeted have 'achieved_recall' and 'target_recall'
            if threshold_name == 'optimal_f1':
                assert 0 <= threshold_config['recall'] <= 1
            else:
                assert 0 <= threshold_config['achieved_recall'] <= 1
                assert 0 <= threshold_config['target_recall'] <= 1

    def test_optimize_thresholds_recall_close_to_target(self, mock_model_with_probabilities,
                                                         sample_classification_data):
        """Test that actual recall is close to target recall"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        custom_targets = {'test_90pct': 0.90}

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val,
            recall_targets=custom_targets,
            verbose=False
        )

        # Actual recall should be close to target (within 10% tolerance)
        # Skip optimal_f1 since it doesn't have target_recall
        for threshold_name, threshold_config in config.items():
            if threshold_name == 'optimal_f1':
                continue
            target = threshold_config['target_recall']
            actual = threshold_config['achieved_recall']
            assert abs(actual - target) < 0.10, \
                f"Recall {actual} too far from target {target}"

    def test_optimize_thresholds_calls_predict_proba(self, mock_model_with_probabilities,
                                                      sample_classification_data):
        """Test that optimize_thresholds calls model.predict_proba()"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        mock_model_with_probabilities.predict_proba.assert_called_once()

    def test_optimize_thresholds_verbose_false_no_print(self, mock_model_with_probabilities,
                                                         sample_classification_data,
                                                         capsys):
        """Test that verbose=False doesn't print output"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_optimize_thresholds_verbose_true_prints_info(self, mock_model_with_probabilities,
                                                           sample_classification_data,
                                                           capsys):
        """Test that verbose=True prints optimization information"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=True
        )

        captured = capsys.readouterr()
        assert "THRESHOLD OPTIMIZATION" in captured.out
        assert "Target Recall:" in captured.out
        assert "Achieved Recall:" in captured.out  # Changed from "Actual Recall:"
        assert "Precision:" in captured.out
        assert "Threshold:" in captured.out

    def test_optimize_thresholds_default_config_has_three_targets(self,
                                                                   mock_model_with_probabilities,
                                                                   sample_classification_data):
        """Test that default config includes all four threshold strategies"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val, verbose=False
        )

        # Should have 4 strategies: optimal_f1 + 3 recall-targeted thresholds
        assert len(config) == 4
        assert 'optimal_f1' in config
        assert 'conservative_90pct_recall' in config
        assert 'balanced_85pct_recall' in config
        assert 'aggressive_80pct_recall' in config

    def test_optimize_thresholds_target_recall_matches_input(self,
                                                              mock_model_with_probabilities,
                                                              sample_classification_data):
        """Test that target_recall in output matches input recall_targets"""
        y_val, _ = sample_classification_data
        X_val = np.random.randn(len(y_val), 5)

        custom_targets = {
            'target_1': 0.95,
            'target_2': 0.85,
            'target_3': 0.75
        }

        config = optimize_thresholds(
            mock_model_with_probabilities, X_val, y_val,
            recall_targets=custom_targets,
            verbose=False
        )

        for name, target_value in custom_targets.items():
            assert config[name]['target_recall'] == target_value
