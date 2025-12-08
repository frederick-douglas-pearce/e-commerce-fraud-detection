"""Tests for cv_analysis module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.fd2_nb.cv_analysis import (
    analyze_cv_results,
    analyze_cv_train_val_gap,
    analyze_iteration_performance,
    get_cv_statistics,
)


class TestAnalyzeCVResults:
    """Tests for analyze_cv_results function."""

    @pytest.fixture
    def sample_cv_results(self, tmp_path):
        """Create sample CV results CSV file."""
        # Create realistic CV results data
        n_candidates = 10
        np.random.seed(42)

        data = {
            'mean_test_score': np.random.uniform(0.75, 0.85, n_candidates),
            'std_test_score': np.random.uniform(0.01, 0.03, n_candidates),
            'mean_fit_time': np.random.uniform(1.0, 5.0, n_candidates),
            'std_fit_time': np.random.uniform(0.1, 0.5, n_candidates),
            'mean_score_time': np.random.uniform(0.01, 0.1, n_candidates),
            'std_score_time': np.random.uniform(0.001, 0.01, n_candidates),
            'rank_test_score': np.arange(1, n_candidates + 1),
            'param_C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0],
        }

        # Make rank match actual scores (highest score = rank 1)
        sorted_indices = np.argsort(-data['mean_test_score'])
        data['rank_test_score'] = np.zeros(n_candidates, dtype=int)
        for rank, idx in enumerate(sorted_indices, 1):
            data['rank_test_score'][idx] = rank

        df = pd.DataFrame(data)
        csv_path = tmp_path / 'cv_results.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_returns_dataframe(self, sample_cv_results):
        """Test that analyze_cv_results returns a DataFrame."""
        result = analyze_cv_results(sample_cv_results, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_returns_top_n(self, sample_cv_results):
        """Test that result contains top_n rows."""
        result = analyze_cv_results(sample_cv_results, top_n=3, verbose=False)
        assert len(result) == 3

    def test_sorted_by_score(self, sample_cv_results):
        """Test that results are sorted by mean_test_score descending."""
        result = analyze_cv_results(sample_cv_results, top_n=5, verbose=False)
        scores = result['mean_test_score'].values
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

    def test_contains_required_columns(self, sample_cv_results):
        """Test that result contains required columns."""
        result = analyze_cv_results(sample_cv_results, verbose=False)
        required_cols = ['mean_test_score', 'std_test_score', 'rank_test_score']
        for col in required_cols:
            assert col in result.columns


class TestGetCVStatistics:
    """Tests for get_cv_statistics function."""

    @pytest.fixture
    def sample_cv_results(self, tmp_path):
        """Create sample CV results CSV file."""
        data = {
            'mean_test_score': [0.85, 0.82, 0.80],
            'std_test_score': [0.02, 0.025, 0.03],
            'mean_fit_time': [2.0, 3.0, 4.0],
            'std_fit_time': [0.2, 0.3, 0.4],
            'mean_score_time': [0.05, 0.06, 0.07],
            'std_score_time': [0.005, 0.006, 0.007],
            'rank_test_score': [1, 2, 3],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'cv_results.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_returns_dict(self, sample_cv_results):
        """Test that get_cv_statistics returns a dictionary."""
        result = get_cv_statistics(sample_cv_results)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_cv_results):
        """Test that result contains required keys."""
        result = get_cv_statistics(sample_cv_results)
        required_keys = ['best_score', 'best_std', 'n_candidates', 'mean_score_all']
        for key in required_keys:
            assert key in result

    def test_best_score_is_highest(self, sample_cv_results):
        """Test that best_score is the highest score."""
        result = get_cv_statistics(sample_cv_results)
        assert result['best_score'] == 0.85

    def test_n_candidates_correct(self, sample_cv_results):
        """Test that n_candidates is correct."""
        result = get_cv_statistics(sample_cv_results)
        assert result['n_candidates'] == 3


class TestAnalyzeCVTrainValGap:
    """Tests for analyze_cv_train_val_gap function."""

    @pytest.fixture
    def sample_cv_results_with_train(self, tmp_path):
        """Create sample CV results with training scores (multi-metric format)."""
        data = {
            'mean_train_pr_auc': [0.95, 0.92, 0.90],
            'mean_test_pr_auc': [0.85, 0.84, 0.82],
            'std_train_pr_auc': [0.01, 0.015, 0.02],
            'std_test_pr_auc': [0.02, 0.025, 0.03],
            'rank_test_pr_auc': [1, 2, 3],
            'mean_fit_time': [2.0, 3.0, 4.0],
            'std_fit_time': [0.2, 0.3, 0.4],
            'mean_score_time': [0.05, 0.06, 0.07],
            'std_score_time': [0.005, 0.006, 0.007],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'cv_results_with_train.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def sample_cv_results_no_train(self, tmp_path):
        """Create sample CV results without training scores."""
        data = {
            'mean_test_pr_auc': [0.85, 0.84, 0.82],
            'std_test_pr_auc': [0.02, 0.025, 0.03],
            'rank_test_pr_auc': [1, 2, 3],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'cv_results_no_train.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def sample_cv_results_severe_overfitting(self, tmp_path):
        """Create sample CV results with severe overfitting."""
        data = {
            'mean_train_pr_auc': [0.98, 0.96, 0.94],
            'mean_test_pr_auc': [0.80, 0.78, 0.75],  # 18% gap for best model
            'std_train_pr_auc': [0.01, 0.015, 0.02],
            'std_test_pr_auc': [0.02, 0.025, 0.03],
            'rank_test_pr_auc': [1, 2, 3],
            'mean_fit_time': [2.0, 3.0, 4.0],
            'std_fit_time': [0.2, 0.3, 0.4],
            'mean_score_time': [0.05, 0.06, 0.07],
            'std_score_time': [0.005, 0.006, 0.007],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'cv_results_overfit.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_returns_dict(self, sample_cv_results_with_train):
        """Test that analyze_cv_train_val_gap returns a dictionary."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            verbose=False
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_cv_results_with_train):
        """Test that result contains required keys."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            verbose=False
        )
        required_keys = [
            'best_train_score', 'best_val_score', 'gap', 'gap_pct',
            'diagnosis', 'overfitting_detected', 'recommendation'
        ]
        for key in required_keys:
            assert key in result

    def test_gap_calculation_correct(self, sample_cv_results_with_train):
        """Test that gap is calculated correctly."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            verbose=False
        )
        expected_gap = result['best_train_score'] - result['best_val_score']
        assert abs(result['gap'] - expected_gap) < 0.001

    def test_gap_pct_calculation_correct(self, sample_cv_results_with_train):
        """Test that gap percentage is calculated correctly."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            verbose=False
        )
        expected_gap_pct = result['gap'] / result['best_train_score']
        assert abs(result['gap_pct'] - expected_gap_pct) < 0.001

    def test_good_fit_diagnosis(self, sample_cv_results_with_train):
        """Test good fit diagnosis for small gap."""
        # Gap is ~10.5% which is just above severe threshold
        # Let's check with higher thresholds
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            gap_threshold_warning=0.15,  # 15%
            gap_threshold_severe=0.20,   # 20%
            verbose=False
        )
        assert result['diagnosis'] == 'Good fit'
        assert result['overfitting_detected'] is False

    def test_moderate_overfitting_diagnosis(self, sample_cv_results_with_train):
        """Test moderate overfitting diagnosis."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            gap_threshold_warning=0.05,  # 5%
            gap_threshold_severe=0.15,   # 15%
            verbose=False
        )
        assert result['diagnosis'] == 'MODERATE OVERFITTING'
        assert result['overfitting_detected'] is True

    def test_severe_overfitting_diagnosis(self, sample_cv_results_severe_overfitting):
        """Test severe overfitting diagnosis."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_severe_overfitting,
            refit_metric='pr_auc',
            gap_threshold_warning=0.05,
            gap_threshold_severe=0.10,
            verbose=False
        )
        assert result['diagnosis'] == 'SEVERE OVERFITTING'
        assert result['overfitting_detected'] is True

    def test_raises_error_without_train_scores(self, sample_cv_results_no_train):
        """Test that error is raised when training scores are missing."""
        with pytest.raises(ValueError, match="Training scores not found"):
            analyze_cv_train_val_gap(
                sample_cv_results_no_train,
                refit_metric='pr_auc',
                verbose=False
            )

    def test_recommendation_contains_model_name(self, sample_cv_results_with_train):
        """Test that recommendation includes model name."""
        result = analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            model_name='Random Forest',
            verbose=False
        )
        assert 'Random Forest' in result['recommendation']

    def test_verbose_mode_runs(self, sample_cv_results_with_train):
        """Test that verbose mode runs without error."""
        # Should not raise an exception
        analyze_cv_train_val_gap(
            sample_cv_results_with_train,
            refit_metric='pr_auc',
            verbose=True
        )


class TestAnalyzeIterationPerformance:
    """Tests for analyze_iteration_performance function."""

    @pytest.fixture
    def sample_iteration_cv_results(self, tmp_path):
        """Create sample CV results for iteration analysis."""
        n_estimators_values = [50, 100, 150, 200, 250, 300]
        n_candidates = len(n_estimators_values)

        # Simulate typical learning curve - improvement then plateau
        train_scores = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
        val_scores = [0.82, 0.84, 0.85, 0.855, 0.854, 0.853]  # Peak at 200

        data = {
            'param_classifier__n_estimators': n_estimators_values,
            'mean_train_pr_auc': train_scores,
            'mean_test_pr_auc': val_scores,
            'std_train_pr_auc': [0.01] * n_candidates,
            'std_test_pr_auc': [0.02] * n_candidates,
            'rank_test_pr_auc': [4, 3, 2, 1, 5, 6],  # Rank 1 at n=200
            'mean_fit_time': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'std_fit_time': [0.1] * n_candidates,
            'mean_score_time': [0.05] * n_candidates,
            'std_score_time': [0.005] * n_candidates,
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'iteration_cv_results.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    @pytest.fixture
    def sample_iteration_cv_results_no_train(self, tmp_path):
        """Create sample CV results without training scores."""
        data = {
            'param_classifier__n_estimators': [50, 100, 150],
            'mean_test_pr_auc': [0.82, 0.84, 0.85],
            'std_test_pr_auc': [0.02, 0.02, 0.02],
            'rank_test_pr_auc': [3, 2, 1],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'iteration_cv_no_train.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_returns_dict(self, sample_iteration_cv_results):
        """Test that analyze_iteration_performance returns a dictionary."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        assert isinstance(result, dict)

    def test_contains_required_keys(self, sample_iteration_cv_results):
        """Test that result contains required keys."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        required_keys = [
            'optimal_n_estimators', 'optimal_train_score', 'optimal_val_score',
            'optimal_gap', 'optimal_gap_pct', 'tracking_df'
        ]
        for key in required_keys:
            assert key in result

    def test_finds_optimal_n_estimators(self, sample_iteration_cv_results):
        """Test that optimal n_estimators is found correctly."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        # Best validation score is at n_estimators=200
        assert result['optimal_n_estimators'] == 200

    def test_optimal_val_score_is_maximum(self, sample_iteration_cv_results):
        """Test that optimal_val_score is the maximum validation score."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        tracking_df = result['tracking_df']
        assert result['optimal_val_score'] == tracking_df['val_score_mean'].max()

    def test_tracking_df_has_correct_columns(self, sample_iteration_cv_results):
        """Test that tracking DataFrame has correct columns."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        tracking_df = result['tracking_df']
        required_cols = [
            'n_estimators', 'train_score_mean', 'val_score_mean',
            'gap', 'gap_pct'
        ]
        for col in required_cols:
            assert col in tracking_df.columns

    def test_tracking_df_row_count(self, sample_iteration_cv_results):
        """Test that tracking DataFrame has correct row count."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        # Should have 6 rows for 6 n_estimators values
        assert len(result['tracking_df']) == 6

    def test_gap_calculation_correct(self, sample_iteration_cv_results):
        """Test that gap is calculated correctly."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=False
        )
        expected_gap = result['optimal_train_score'] - result['optimal_val_score']
        assert abs(result['optimal_gap'] - expected_gap) < 0.001

    def test_tuned_n_estimators_stored(self, sample_iteration_cv_results):
        """Test that tuned_n_estimators is stored in result."""
        result = analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            tuned_n_estimators=150,
            verbose=False
        )
        assert result['tuned_n_estimators'] == 150

    def test_raises_error_without_train_scores(self, sample_iteration_cv_results_no_train):
        """Test that error is raised when training scores are missing."""
        with pytest.raises(ValueError, match="Training scores not found"):
            analyze_iteration_performance(
                sample_iteration_cv_results_no_train,
                refit_metric='pr_auc',
                verbose=False
            )

    def test_raises_error_without_n_estimators(self, tmp_path):
        """Test that error is raised when n_estimators column is missing."""
        data = {
            'mean_train_pr_auc': [0.90, 0.92],
            'mean_test_pr_auc': [0.82, 0.84],
            'rank_test_pr_auc': [2, 1],
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / 'no_n_estimators.csv'
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="n_estimators parameter not found"):
            analyze_iteration_performance(
                str(csv_path),
                refit_metric='pr_auc',
                verbose=False
            )

    def test_verbose_mode_runs(self, sample_iteration_cv_results):
        """Test that verbose mode runs without error."""
        # Should not raise an exception
        analyze_iteration_performance(
            sample_iteration_cv_results,
            refit_metric='pr_auc',
            verbose=True
        )
