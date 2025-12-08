"""Tests for bias_variance module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.fd2_nb.bias_variance import analyze_cv_fold_variance


class TestAnalyzeCVFoldVariance:
    """Tests for analyze_cv_fold_variance function."""

    @pytest.fixture
    def sample_cv_files(self, tmp_path):
        """Create sample CV results files."""
        # Random Forest CV results (single-metric format)
        rf_data = {
            'mean_test_score': [0.82, 0.80, 0.78],
            'std_test_score': [0.02, 0.025, 0.03],
            'rank_test_score': [1, 2, 3],
        }
        rf_path = tmp_path / 'rf_cv_results.csv'
        pd.DataFrame(rf_data).to_csv(rf_path, index=False)

        # XGBoost CV results (single-metric format)
        xgb_data = {
            'mean_test_score': [0.85, 0.83, 0.81],
            'std_test_score': [0.015, 0.02, 0.025],
            'rank_test_score': [1, 2, 3],
        }
        xgb_path = tmp_path / 'xgb_cv_results.csv'
        pd.DataFrame(xgb_data).to_csv(xgb_path, index=False)

        return {
            'Random Forest': str(rf_path),
            'XGBoost': str(xgb_path)
        }

    @pytest.fixture
    def sample_cv_files_multi_metric(self, tmp_path):
        """Create sample CV results files with multi-metric format."""
        # Random Forest CV results (multi-metric format)
        rf_data = {
            'mean_test_pr_auc': [0.82, 0.80, 0.78],
            'std_test_pr_auc': [0.02, 0.025, 0.03],
            'rank_test_pr_auc': [1, 2, 3],
            'mean_test_roc_auc': [0.90, 0.88, 0.86],
            'std_test_roc_auc': [0.01, 0.015, 0.02],
            'rank_test_roc_auc': [1, 2, 3],
        }
        rf_path = tmp_path / 'rf_cv_results_multi.csv'
        pd.DataFrame(rf_data).to_csv(rf_path, index=False)

        return {'Random Forest': str(rf_path)}

    def test_returns_dataframe(self, sample_cv_files):
        """Test that analyze_cv_fold_variance returns a DataFrame."""
        result = analyze_cv_fold_variance(sample_cv_files, refit_metric=None, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, sample_cv_files):
        """Test that result contains required columns."""
        result = analyze_cv_fold_variance(sample_cv_files, refit_metric=None, verbose=False)
        required_cols = ['model', 'mean_score', 'std_score', 'cv_coef_pct']
        for col in required_cols:
            assert col in result.columns

    def test_one_row_per_model(self, sample_cv_files):
        """Test that there's one row per model."""
        result = analyze_cv_fold_variance(sample_cv_files, refit_metric=None, verbose=False)
        assert len(result) == 2

    def test_cv_coefficient_calculation(self, sample_cv_files):
        """Test that CV coefficient is calculated correctly."""
        result = analyze_cv_fold_variance(sample_cv_files, refit_metric=None, verbose=False)
        rf_row = result[result['model'] == 'Random Forest'].iloc[0]
        expected_cv = (rf_row['std_score'] / rf_row['mean_score']) * 100
        assert abs(rf_row['cv_coef_pct'] - expected_cv) < 0.01

    def test_multi_metric_format(self, sample_cv_files_multi_metric):
        """Test with multi-metric CV results format."""
        result = analyze_cv_fold_variance(
            sample_cv_files_multi_metric,
            refit_metric='pr_auc',
            verbose=False
        )
        assert len(result) == 1
        assert result.iloc[0]['mean_score'] == 0.82  # Best pr_auc score

    def test_handles_missing_file(self, tmp_path):
        """Test that missing files are handled gracefully."""
        paths = {'Missing Model': str(tmp_path / 'nonexistent.csv')}
        result = analyze_cv_fold_variance(paths, refit_metric=None, verbose=False)
        assert len(result) == 0

    def test_handles_glob_pattern(self, tmp_path):
        """Test that glob patterns work correctly."""
        # Create files with timestamp pattern
        for i, suffix in enumerate(['001', '002']):
            data = {
                'mean_test_score': [0.80 + i * 0.02],
                'std_test_score': [0.02],
                'rank_test_score': [1],
            }
            path = tmp_path / f'model_cv_results_{suffix}.csv'
            pd.DataFrame(data).to_csv(path, index=False)

        paths = {'Model': str(tmp_path / 'model_cv_results_*.csv')}
        result = analyze_cv_fold_variance(paths, refit_metric=None, verbose=False)
        assert len(result) == 1
        # Should use the most recent file (002)
        assert abs(result.iloc[0]['mean_score'] - 0.82) < 0.001

    def test_selects_best_config(self, sample_cv_files):
        """Test that the best configuration (rank 1) is selected."""
        result = analyze_cv_fold_variance(sample_cv_files, refit_metric=None, verbose=False)
        rf_row = result[result['model'] == 'Random Forest'].iloc[0]
        # Best RF config has mean_test_score = 0.82
        assert rf_row['mean_score'] == 0.82
