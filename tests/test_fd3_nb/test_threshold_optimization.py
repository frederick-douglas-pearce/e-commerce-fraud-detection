"""Tests for threshold_optimization module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import precision_recall_curve

from src.fd3_nb.threshold_optimization import (
    find_threshold_for_recall,
    find_optimal_f1_threshold,
    find_target_performance_threshold,
    optimize_thresholds,
    create_threshold_comparison_df,
)


@pytest.fixture
def sample_predictions():
    """Create sample y_true and y_pred_proba for testing."""
    np.random.seed(42)
    n_samples = 1000

    # Create imbalanced dataset (10% positive class)
    y_true = np.zeros(n_samples)
    y_true[:100] = 1
    np.random.shuffle(y_true)

    # Create realistic predictions - higher for positive class
    y_pred_proba = np.where(
        y_true == 1,
        np.random.beta(5, 2, n_samples),  # Higher proba for fraud
        np.random.beta(2, 5, n_samples)   # Lower proba for non-fraud
    )

    return pd.Series(y_true), y_pred_proba


@pytest.fixture
def pr_curve_data(sample_predictions):
    """Get precision-recall curve data from sample predictions."""
    y_true, y_pred_proba = sample_predictions
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    return precisions, recalls, thresholds


class TestFindThresholdForRecall:
    """Tests for find_threshold_for_recall function."""

    def test_returns_tuple_of_four(self, pr_curve_data):
        """Test that function returns tuple of four values."""
        precisions, recalls, thresholds = pr_curve_data
        result = find_threshold_for_recall(0.80, precisions, recalls, thresholds)
        assert len(result) == 4

    def test_achieves_target_recall(self, pr_curve_data):
        """Test that returned threshold achieves target recall."""
        precisions, recalls, thresholds = pr_curve_data
        target_recall = 0.80
        threshold, precision, recall, f1 = find_threshold_for_recall(
            target_recall, precisions, recalls, thresholds
        )
        assert recall >= target_recall

    def test_returns_none_for_impossible_recall(self, pr_curve_data):
        """Test that function returns None for impossible target recall."""
        precisions, recalls, thresholds = pr_curve_data
        # Recall of 1.0+ is impossible
        result = find_threshold_for_recall(1.01, precisions, recalls, thresholds)
        assert result == (None, None, None, None)

    def test_f1_calculation_correct(self, pr_curve_data):
        """Test that F1 score is calculated correctly."""
        precisions, recalls, thresholds = pr_curve_data
        threshold, precision, recall, f1 = find_threshold_for_recall(
            0.80, precisions, recalls, thresholds
        )
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert abs(f1 - expected_f1) < 0.001


class TestFindOptimalF1Threshold:
    """Tests for find_optimal_f1_threshold function."""

    def test_returns_tuple_of_four(self, pr_curve_data):
        """Test that function returns tuple of four values."""
        precisions, recalls, thresholds = pr_curve_data
        result = find_optimal_f1_threshold(precisions, recalls, thresholds)
        assert len(result) == 4

    def test_f1_is_maximum(self, pr_curve_data):
        """Test that returned F1 is the maximum possible."""
        precisions, recalls, thresholds = pr_curve_data
        threshold, precision, recall, f1 = find_optimal_f1_threshold(
            precisions, recalls, thresholds
        )

        # Calculate F1 for all thresholds
        all_f1 = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        max_f1 = np.max(all_f1)

        assert abs(f1 - max_f1) < 0.001

    def test_threshold_in_valid_range(self, pr_curve_data):
        """Test that returned threshold is in [0, 1] range."""
        precisions, recalls, thresholds = pr_curve_data
        threshold, _, _, _ = find_optimal_f1_threshold(precisions, recalls, thresholds)
        assert 0 <= threshold <= 1

    def test_precision_recall_in_valid_range(self, pr_curve_data):
        """Test that precision and recall are in [0, 1] range."""
        precisions, recalls, thresholds = pr_curve_data
        _, precision, recall, _ = find_optimal_f1_threshold(precisions, recalls, thresholds)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1


class TestFindTargetPerformanceThreshold:
    """Tests for find_target_performance_threshold function."""

    def test_returns_tuple_of_four(self, pr_curve_data):
        """Test that function returns tuple of four values."""
        precisions, recalls, thresholds = pr_curve_data
        result = find_target_performance_threshold(precisions, recalls, thresholds, min_precision=0.5)
        assert len(result) == 4

    def test_meets_min_precision(self, pr_curve_data):
        """Test that returned precision meets minimum constraint."""
        precisions, recalls, thresholds = pr_curve_data
        min_precision = 0.5
        threshold, precision, recall, f1 = find_target_performance_threshold(
            precisions, recalls, thresholds, min_precision=min_precision
        )
        if threshold is not None:
            assert precision >= min_precision

    def test_maximizes_recall_within_constraint(self, pr_curve_data):
        """Test that recall is maximized among valid thresholds."""
        precisions, recalls, thresholds = pr_curve_data
        min_precision = 0.5
        threshold, precision, recall, f1 = find_target_performance_threshold(
            precisions, recalls, thresholds, min_precision=min_precision
        )

        if threshold is not None:
            # Check that recall is maximum among thresholds meeting precision constraint
            valid_mask = precisions[:-1] >= min_precision
            max_valid_recall = np.max(recalls[:-1][valid_mask])
            assert abs(recall - max_valid_recall) < 0.001

    def test_returns_none_for_impossible_precision(self, pr_curve_data):
        """Test that function returns None when min_precision cannot be met."""
        precisions, recalls, thresholds = pr_curve_data
        # Precision of 1.0 may not be achievable
        result = find_target_performance_threshold(
            precisions, recalls, thresholds, min_precision=1.01
        )
        assert result == (None, None, None, None)

    def test_f1_calculation_correct(self, pr_curve_data):
        """Test that F1 score is calculated correctly."""
        precisions, recalls, thresholds = pr_curve_data
        threshold, precision, recall, f1 = find_target_performance_threshold(
            precisions, recalls, thresholds, min_precision=0.5
        )
        if threshold is not None:
            expected_f1 = 2 * (precision * recall) / (precision + recall)
            assert abs(f1 - expected_f1) < 0.001


class TestOptimizeThresholds:
    """Tests for optimize_thresholds function."""

    def test_returns_tuple_of_three(self, sample_predictions):
        """Test that function returns tuple of three values."""
        y_true, y_pred_proba = sample_predictions
        result = optimize_thresholds(y_true, y_pred_proba, verbose=False)
        assert len(result) == 3

    def test_optimal_f1_result_has_required_keys(self, sample_predictions):
        """Test that optimal F1 result contains required keys."""
        y_true, y_pred_proba = sample_predictions
        optimal_f1_result, _, _ = optimize_thresholds(y_true, y_pred_proba, verbose=False)

        required_keys = ['name', 'threshold', 'precision', 'recall', 'f1', 'tn', 'fp', 'fn', 'tp']
        for key in required_keys:
            assert key in optimal_f1_result

    def test_target_performance_result_structure(self, sample_predictions):
        """Test that target performance result has correct structure."""
        y_true, y_pred_proba = sample_predictions
        _, target_perf_result, _ = optimize_thresholds(
            y_true, y_pred_proba, min_precision_target=0.5, verbose=False
        )

        if target_perf_result is not None:
            required_keys = ['name', 'threshold', 'precision', 'recall', 'f1', 'min_precision']
            for key in required_keys:
                assert key in target_perf_result

    def test_threshold_results_list_length(self, sample_predictions):
        """Test that threshold results list matches recall targets count."""
        y_true, y_pred_proba = sample_predictions
        recall_targets = [0.90, 0.85, 0.80]
        _, _, threshold_results = optimize_thresholds(
            y_true, y_pred_proba, recall_targets=recall_targets, verbose=False
        )

        # Should have at least some results (may not achieve all targets)
        assert len(threshold_results) <= len(recall_targets)

    def test_verbose_mode_runs(self, sample_predictions):
        """Test that verbose mode runs without error."""
        y_true, y_pred_proba = sample_predictions
        # Should not raise an exception
        optimize_thresholds(y_true, y_pred_proba, verbose=True)


class TestCreateThresholdComparisonDf:
    """Tests for create_threshold_comparison_df function."""

    @pytest.fixture
    def sample_results(self):
        """Create sample threshold results for testing."""
        optimal_f1_result = {
            'name': 'optimal_f1',
            'threshold': 0.45,
            'precision': 0.72,
            'recall': 0.85,
            'f1': 0.78,
            'fp': 100,
            'fn': 50
        }

        target_performance_result = {
            'name': 'target_performance',
            'threshold': 0.35,
            'precision': 0.70,
            'recall': 0.88,
            'f1': 0.78,
            'min_precision': 0.70,
            'fp': 120,
            'fn': 40
        }

        threshold_results = [
            {'target_recall': 0.90, 'threshold': 0.30, 'precision': 0.65, 'recall': 0.90, 'f1': 0.75, 'fp': 150, 'fn': 30},
            {'target_recall': 0.85, 'threshold': 0.40, 'precision': 0.70, 'recall': 0.86, 'f1': 0.77, 'fp': 110, 'fn': 45},
            {'target_recall': 0.80, 'threshold': 0.50, 'precision': 0.75, 'recall': 0.81, 'f1': 0.78, 'fp': 80, 'fn': 60},
        ]

        return optimal_f1_result, target_performance_result, threshold_results

    def test_returns_dataframe(self, sample_results):
        """Test that function returns a DataFrame."""
        optimal_f1, target_perf, threshold_results = sample_results
        result = create_threshold_comparison_df(optimal_f1, target_perf, threshold_results)
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, sample_results):
        """Test that DataFrame contains required columns."""
        optimal_f1, target_perf, threshold_results = sample_results
        result = create_threshold_comparison_df(optimal_f1, target_perf, threshold_results)

        required_cols = ['strategy', 'threshold', 'precision', 'recall', 'f1', 'fp', 'fn']
        for col in required_cols:
            assert col in result.columns

    def test_row_count_with_target_performance(self, sample_results):
        """Test correct row count when target_performance is provided."""
        optimal_f1, target_perf, threshold_results = sample_results
        result = create_threshold_comparison_df(optimal_f1, target_perf, threshold_results)

        # 1 optimal_f1 + 1 target_performance + 3 recall-targeted = 5 rows
        assert len(result) == 5

    def test_row_count_without_target_performance(self, sample_results):
        """Test correct row count when target_performance is None."""
        optimal_f1, _, threshold_results = sample_results
        result = create_threshold_comparison_df(optimal_f1, None, threshold_results)

        # 1 optimal_f1 + 3 recall-targeted = 4 rows
        assert len(result) == 4

    def test_strategy_names(self, sample_results):
        """Test that strategy names are correctly formatted."""
        optimal_f1, target_perf, threshold_results = sample_results
        result = create_threshold_comparison_df(optimal_f1, target_perf, threshold_results)

        strategies = result['strategy'].tolist()
        assert 'Optimal F1 (Best Balance)' in strategies
        assert any('Target Performance' in s for s in strategies)
        assert any('Conservative' in s or '90%' in s for s in strategies)
