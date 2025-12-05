"""Tests for cv_analysis module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.fd2_nb.cv_analysis import analyze_cv_results, get_cv_statistics


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
