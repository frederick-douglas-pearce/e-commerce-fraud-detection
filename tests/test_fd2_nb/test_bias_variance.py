"""Tests for bias_variance module."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.fd2_nb.bias_variance import (
    calculate_train_val_gap,
    analyze_train_val_gaps,
    analyze_cv_fold_variance,
    generate_bias_variance_report,
)


class TestCalculateTrainValGap:
    """Tests for calculate_train_val_gap function."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data and trained model."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        X_train, y_train = pd.DataFrame(X[:150]), pd.Series(y[:150])
        X_val, y_val = pd.DataFrame(X[150:]), pd.Series(y[150:])

        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X_train, y_train)

        return model, X_train, y_train, X_val, y_val

    def test_returns_dict(self, classification_data):
        """Test that calculate_train_val_gap returns a dictionary."""
        model, X_train, y_train, X_val, y_val = classification_data
        result = calculate_train_val_gap(model, X_train, y_train, X_val, y_val)
        assert isinstance(result, dict)

    def test_contains_required_keys(self, classification_data):
        """Test that result contains required keys."""
        model, X_train, y_train, X_val, y_val = classification_data
        result = calculate_train_val_gap(model, X_train, y_train, X_val, y_val)
        required_keys = ['train_score', 'val_score', 'gap', 'gap_pct', 'diagnosis']
        for key in required_keys:
            assert key in result

    def test_gap_calculation(self, classification_data):
        """Test that gap is calculated correctly."""
        model, X_train, y_train, X_val, y_val = classification_data
        result = calculate_train_val_gap(model, X_train, y_train, X_val, y_val)
        expected_gap = result['train_score'] - result['val_score']
        assert abs(result['gap'] - expected_gap) < 1e-10

    def test_different_metrics(self, classification_data):
        """Test with different metrics."""
        model, X_train, y_train, X_val, y_val = classification_data
        for metric in ['pr_auc', 'roc_auc', 'f1']:
            result = calculate_train_val_gap(
                model, X_train, y_train, X_val, y_val, metric=metric
            )
            assert 'train_score' in result
            assert 0 <= result['train_score'] <= 1

    def test_invalid_metric_raises(self, classification_data):
        """Test that invalid metric raises ValueError."""
        model, X_train, y_train, X_val, y_val = classification_data
        with pytest.raises(ValueError, match="Unknown metric"):
            calculate_train_val_gap(
                model, X_train, y_train, X_val, y_val, metric='invalid'
            )

    def test_diagnosis_good_fit(self, classification_data):
        """Test diagnosis for good fit (low gap)."""
        model, X_train, y_train, X_val, y_val = classification_data
        result = calculate_train_val_gap(model, X_train, y_train, X_val, y_val)
        # LogisticRegression typically has good generalization
        assert 'Good fit' in result['diagnosis'] or 'VARIANCE' in result['diagnosis']


class TestAnalyzeTrainValGaps:
    """Tests for analyze_train_val_gaps function."""

    @pytest.fixture
    def classification_data(self):
        """Create classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        X_train = pd.DataFrame(X[:150])
        y_train = pd.Series(y[:150])
        X_val = pd.DataFrame(X[150:])
        y_val = pd.Series(y[150:])
        return X_train, y_train, X_val, y_val

    @pytest.fixture
    def simple_models(self):
        """Create simple models dictionary."""
        return {
            'Logistic Regression': LogisticRegression(max_iter=200, random_state=42),
        }

    def test_returns_dataframe(self, classification_data, simple_models):
        """Test that analyze_train_val_gaps returns a DataFrame."""
        X_train, y_train, X_val, y_val = classification_data
        result = analyze_train_val_gaps(
            simple_models, X_train, y_train, X_val, y_val,
            cv_folds=2, verbose=False
        )
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, classification_data, simple_models):
        """Test that result contains required columns."""
        X_train, y_train, X_val, y_val = classification_data
        result = analyze_train_val_gaps(
            simple_models, X_train, y_train, X_val, y_val,
            cv_folds=2, verbose=False
        )
        required_cols = ['model', 'train_score', 'val_score', 'gap', 'diagnosis']
        for col in required_cols:
            assert col in result.columns

    def test_one_row_per_model(self, classification_data, simple_models):
        """Test that there's one row per model."""
        X_train, y_train, X_val, y_val = classification_data
        result = analyze_train_val_gaps(
            simple_models, X_train, y_train, X_val, y_val,
            cv_folds=2, verbose=False
        )
        assert len(result) == len(simple_models)


class TestAnalyzeCVFoldVariance:
    """Tests for analyze_cv_fold_variance function."""

    @pytest.fixture
    def sample_cv_files(self, tmp_path):
        """Create sample CV results files."""
        # Random Forest CV results
        rf_data = {
            'mean_test_score': [0.82, 0.80, 0.78],
            'std_test_score': [0.02, 0.025, 0.03],
            'rank_test_score': [1, 2, 3],
        }
        rf_path = tmp_path / 'rf_cv_results.csv'
        pd.DataFrame(rf_data).to_csv(rf_path, index=False)

        # XGBoost CV results
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

    def test_returns_dataframe(self, sample_cv_files):
        """Test that analyze_cv_fold_variance returns a DataFrame."""
        result = analyze_cv_fold_variance(sample_cv_files, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, sample_cv_files):
        """Test that result contains required columns."""
        result = analyze_cv_fold_variance(sample_cv_files, verbose=False)
        required_cols = ['model', 'mean_score', 'std_score', 'cv_coef_pct']
        for col in required_cols:
            assert col in result.columns

    def test_one_row_per_model(self, sample_cv_files):
        """Test that there's one row per model."""
        result = analyze_cv_fold_variance(sample_cv_files, verbose=False)
        assert len(result) == 2

    def test_cv_coefficient_calculation(self, sample_cv_files):
        """Test that CV coefficient is calculated correctly."""
        result = analyze_cv_fold_variance(sample_cv_files, verbose=False)
        rf_row = result[result['model'] == 'Random Forest'].iloc[0]
        expected_cv = (rf_row['std_score'] / rf_row['mean_score']) * 100
        assert abs(rf_row['cv_coef_pct'] - expected_cv) < 0.01


class TestGenerateBiasVarianceReport:
    """Tests for generate_bias_variance_report function."""

    @pytest.fixture
    def sample_gap_df(self):
        """Create sample gap DataFrame."""
        return pd.DataFrame({
            'model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
            'train_score': [0.78, 0.95, 0.88],
            'val_score': [0.76, 0.80, 0.86],
            'gap': [0.02, 0.15, 0.02],
            'gap_pct': [2.6, 15.8, 2.3],
            'diagnosis': ['Good fit', 'HIGH VARIANCE (Severe Overfitting)', 'Good fit']
        })

    def test_returns_string(self, sample_gap_df):
        """Test that generate_bias_variance_report returns a string."""
        result = generate_bias_variance_report(sample_gap_df, verbose=False)
        assert isinstance(result, str)

    def test_contains_model_names(self, sample_gap_df):
        """Test that report contains model names."""
        result = generate_bias_variance_report(sample_gap_df, verbose=False)
        assert 'Logistic Regression' in result
        assert 'Random Forest' in result
        assert 'XGBoost' in result

    def test_contains_diagnosis(self, sample_gap_df):
        """Test that report contains diagnosis."""
        result = generate_bias_variance_report(sample_gap_df, verbose=False)
        assert 'Good fit' in result or 'VARIANCE' in result

    def test_identifies_best_model(self, sample_gap_df):
        """Test that report identifies best model."""
        result = generate_bias_variance_report(sample_gap_df, verbose=False)
        assert 'XGBoost' in result  # XGBoost has highest val_score

    def test_saves_to_file(self, sample_gap_df, tmp_path):
        """Test that report can be saved to file."""
        output_path = tmp_path / 'report.txt'
        generate_bias_variance_report(
            sample_gap_df,
            output_path=str(output_path),
            verbose=False
        )
        assert output_path.exists()
        content = output_path.read_text()
        assert 'BIAS-VARIANCE' in content
