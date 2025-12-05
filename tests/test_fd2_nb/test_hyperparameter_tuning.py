"""Tests for hyperparameter_tuning module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from src.fd2_nb.hyperparameter_tuning import (
    create_search_object,
    tune_with_logging,
    get_best_params_summary,
    _calculate_total_combinations,
)


class TestCreateSearchObject:
    """Tests for create_search_object function."""

    @pytest.fixture
    def simple_estimator(self):
        """Create simple estimator for testing."""
        return LogisticRegression(max_iter=100)

    @pytest.fixture
    def simple_param_grid(self):
        """Create simple parameter grid."""
        return {'C': [0.1, 1.0, 10.0]}

    @pytest.fixture
    def cv_strategy(self):
        """Create CV strategy."""
        return StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    def test_create_grid_search(self, simple_estimator, simple_param_grid, cv_strategy):
        """Test creating GridSearchCV object."""
        search = create_search_object(
            search_type='grid',
            estimator=simple_estimator,
            param_grid=simple_param_grid,
            cv=cv_strategy
        )
        assert search.__class__.__name__ == 'GridSearchCV'

    def test_create_random_search(self, simple_estimator, simple_param_grid, cv_strategy):
        """Test creating RandomizedSearchCV object."""
        search = create_search_object(
            search_type='random',
            estimator=simple_estimator,
            param_grid=simple_param_grid,
            cv=cv_strategy,
            n_iter=2
        )
        assert search.__class__.__name__ == 'RandomizedSearchCV'

    def test_invalid_search_type_raises(self, simple_estimator, simple_param_grid, cv_strategy):
        """Test that invalid search_type raises ValueError."""
        with pytest.raises(ValueError, match="must be 'grid' or 'random'"):
            create_search_object(
                search_type='invalid',
                estimator=simple_estimator,
                param_grid=simple_param_grid,
                cv=cv_strategy
            )

    def test_random_search_default_n_iter(self, simple_estimator, simple_param_grid, cv_strategy):
        """Test that RandomizedSearchCV defaults to n_iter=10."""
        search = create_search_object(
            search_type='random',
            estimator=simple_estimator,
            param_grid=simple_param_grid,
            cv=cv_strategy
        )
        assert search.n_iter == 10

    def test_search_object_scoring(self, simple_estimator, simple_param_grid, cv_strategy):
        """Test that scoring parameter is set correctly."""
        search = create_search_object(
            search_type='grid',
            estimator=simple_estimator,
            param_grid=simple_param_grid,
            cv=cv_strategy,
            scoring='roc_auc'
        )
        assert search.scoring == 'roc_auc'


class TestCalculateTotalCombinations:
    """Tests for _calculate_total_combinations helper."""

    def test_single_param(self):
        """Test with single parameter."""
        param_grid = {'C': [0.1, 1.0, 10.0]}
        assert _calculate_total_combinations(param_grid) == 3

    def test_multiple_params(self):
        """Test with multiple parameters."""
        param_grid = {
            'C': [0.1, 1.0],
            'penalty': ['l1', 'l2']
        }
        assert _calculate_total_combinations(param_grid) == 4

    def test_empty_grid(self):
        """Test with empty grid."""
        assert _calculate_total_combinations({}) == 1


class TestTuneWithLogging:
    """Tests for tune_with_logging function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple classification data."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        return pd.DataFrame(X), pd.Series(y)

    @pytest.fixture
    def simple_search(self):
        """Create simple search object."""
        estimator = LogisticRegression(max_iter=100)
        param_grid = {'C': [0.1, 1.0]}
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        return create_search_object(
            search_type='grid',
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=0
        )

    def test_tune_with_logging_returns_tuple(self, simple_search, simple_data):
        """Test that tune_with_logging returns correct tuple."""
        X, y = simple_data
        with tempfile.TemporaryDirectory() as tmpdir:
            result = tune_with_logging(
                simple_search, X, y, 'test_model',
                log_dir=tmpdir,
                verbose=False
            )
            assert isinstance(result, tuple)
            assert len(result) == 3

    def test_tune_with_logging_creates_files(self, simple_search, simple_data):
        """Test that log and CSV files are created."""
        X, y = simple_data
        with tempfile.TemporaryDirectory() as tmpdir:
            _, log_path, csv_path = tune_with_logging(
                simple_search, X, y, 'test_model',
                log_dir=tmpdir,
                verbose=False
            )
            assert Path(log_path).exists()
            assert Path(csv_path).exists()

    def test_tune_with_logging_csv_content(self, simple_search, simple_data):
        """Test that CSV contains CV results."""
        X, y = simple_data
        with tempfile.TemporaryDirectory() as tmpdir:
            _, _, csv_path = tune_with_logging(
                simple_search, X, y, 'test_model',
                log_dir=tmpdir,
                verbose=False
            )
            df = pd.read_csv(csv_path)
            assert 'mean_test_score' in df.columns
            assert len(df) == 2  # 2 parameter combinations


class TestGetBestParamsSummary:
    """Tests for get_best_params_summary function."""

    @pytest.fixture
    def fitted_search(self):
        """Create fitted search object."""
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42
        )
        estimator = LogisticRegression(max_iter=100)
        param_grid = {'C': [0.1, 1.0, 10.0]}
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        search = create_search_object(
            search_type='grid',
            estimator=estimator,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            verbose=0
        )
        search.fit(X, y)
        return search

    def test_returns_dict(self, fitted_search):
        """Test that function returns a dictionary."""
        result = get_best_params_summary(fitted_search, verbose=False)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, fitted_search):
        """Test that result contains required keys."""
        result = get_best_params_summary(fitted_search, verbose=False)
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'cv_results_summary' in result

    def test_best_params_is_dict(self, fitted_search):
        """Test that best_params is a dictionary."""
        result = get_best_params_summary(fitted_search, verbose=False)
        assert isinstance(result['best_params'], dict)

    def test_best_score_is_float(self, fitted_search):
        """Test that best_score is a float."""
        result = get_best_params_summary(fitted_search, verbose=False)
        assert isinstance(result['best_score'], float)
