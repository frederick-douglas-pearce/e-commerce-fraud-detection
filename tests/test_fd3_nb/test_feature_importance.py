"""Tests for feature_importance module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

from src.fd3_nb.feature_importance import (
    extract_feature_importance,
    print_feature_importance_summary,
    compute_shap_importance,
    compare_importance_methods,
    print_shap_importance_summary,
    print_importance_comparison,
)


@pytest.fixture
def sample_xgb_pipeline():
    """Create a trained XGBoost pipeline for testing."""
    np.random.seed(42)
    n_samples = 500
    n_features = 10

    # Create features
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=feature_names
    )

    # Create target (10% positive)
    y = pd.Series(np.zeros(n_samples))
    y.iloc[:50] = 1
    y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    # Make first few features predictive
    X.loc[y == 1, 'feature_0'] += 2.0
    X.loc[y == 1, 'feature_1'] += 1.5
    X.loc[y == 1, 'feature_2'] += 1.0

    # Create and train pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            n_estimators=50,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    model.fit(X, y)

    return model, feature_names


class TestExtractFeatureImportance:
    """Tests for extract_feature_importance function."""

    def test_returns_dataframe(self, sample_xgb_pipeline):
        """Test that extract_feature_importance returns a DataFrame."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_contains_required_columns(self, sample_xgb_pipeline):
        """Test that result contains required columns."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert 'feature' in result.columns
        assert 'importance' in result.columns

    def test_correct_number_of_rows(self, sample_xgb_pipeline):
        """Test that result has correct number of rows."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert len(result) == len(feature_names)

    def test_sorted_by_importance_descending(self, sample_xgb_pipeline):
        """Test that results are sorted by importance descending."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        importance_values = result['importance'].values
        assert all(importance_values[i] >= importance_values[i+1]
                   for i in range(len(importance_values)-1))

    def test_importance_values_non_negative(self, sample_xgb_pipeline):
        """Test that all importance values are non-negative."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        assert (result['importance'] >= 0).all()

    def test_importance_values_sum_to_one(self, sample_xgb_pipeline):
        """Test that importance values sum to approximately 1."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        total_importance = result['importance'].sum()
        assert abs(total_importance - 1.0) < 0.01

    def test_top_features_are_predictive(self, sample_xgb_pipeline):
        """Test that most predictive features have high importance."""
        model, feature_names = sample_xgb_pipeline
        result = extract_feature_importance(model, feature_names, verbose=False)
        top_3_features = result.head(3)['feature'].tolist()
        # The predictive features (feature_0, feature_1, feature_2) should be in top 3
        predictive_features = ['feature_0', 'feature_1', 'feature_2']
        overlap = len(set(top_3_features) & set(predictive_features))
        assert overlap >= 2, "At least 2 of top 3 should be predictive features"

    def test_verbose_mode_runs(self, sample_xgb_pipeline):
        """Test that verbose mode runs without error."""
        model, feature_names = sample_xgb_pipeline
        # Should not raise an exception
        extract_feature_importance(model, feature_names, verbose=True)


class TestPrintFeatureImportanceSummary:
    """Tests for print_feature_importance_summary function."""

    @pytest.fixture
    def sample_importance_df(self):
        """Create sample feature importance DataFrame."""
        np.random.seed(42)
        n_features = 30

        # Create importance values that sum to 1
        importance_values = np.random.exponential(scale=0.1, size=n_features)
        importance_values = importance_values / importance_values.sum()
        importance_values = np.sort(importance_values)[::-1]  # Sort descending

        return pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(n_features)],
            'importance': importance_values
        })

    def test_runs_without_error(self, sample_importance_df, capsys):
        """Test that print_feature_importance_summary runs without error."""
        # Should not raise an exception
        print_feature_importance_summary(sample_importance_df)

        # Verify something was printed
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_prints_top_n_features(self, sample_importance_df, capsys):
        """Test that function prints top N features."""
        print_feature_importance_summary(sample_importance_df, top_n=10)
        captured = capsys.readouterr()

        # Should mention "Top 10"
        assert "Top 10" in captured.out

    def test_prints_cumulative_importance(self, sample_importance_df, capsys):
        """Test that function prints cumulative importance info."""
        print_feature_importance_summary(sample_importance_df)
        captured = capsys.readouterr()

        # Should mention model concentration
        assert "Top 5 features" in captured.out
        assert "Top 10 features" in captured.out

    def test_custom_top_n(self, sample_importance_df, capsys):
        """Test with custom top_n value."""
        print_feature_importance_summary(sample_importance_df, top_n=5)
        captured = capsys.readouterr()

        # Should mention "Top 5"
        assert "Top 5" in captured.out


@pytest.fixture
def sample_xgb_pipeline_with_preprocessor():
    """Create a trained XGBoost pipeline with ColumnTransformer for SHAP testing."""
    np.random.seed(42)
    n_samples = 200
    n_numeric_features = 8

    # Create numeric features
    numeric_cols = [f'num_feature_{i}' for i in range(n_numeric_features)]
    X_numeric = pd.DataFrame(
        np.random.randn(n_samples, n_numeric_features),
        columns=numeric_cols
    )

    # Create categorical feature
    X_cat = pd.DataFrame({
        'category': np.random.choice(['A', 'B'], n_samples)
    })

    X = pd.concat([X_numeric, X_cat], axis=1)

    # Create target (10% positive)
    y = pd.Series(np.zeros(n_samples))
    y.iloc[:20] = 1
    y = y.sample(frac=1, random_state=42).reset_index(drop=True)

    # Make first few features predictive
    X.loc[y == 1, 'num_feature_0'] += 2.0
    X.loc[y == 1, 'num_feature_1'] += 1.5

    # Create preprocessor (mimics the real pipeline structure)
    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), ['category']),
        ('rest', 'passthrough', numeric_cols)
    ])

    # Create and train pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(
            n_estimators=30,
            max_depth=3,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    model.fit(X, y)

    # Get feature names after preprocessing
    feature_names = [
        name.split('__', 1)[1] if '__' in name else name
        for name in preprocessor.get_feature_names_out()
    ]

    return model, X, feature_names


class TestComputeShapImportance:
    """Tests for compute_shap_importance function."""

    def test_returns_tuple(self, sample_xgb_pipeline_with_preprocessor):
        """Test that compute_shap_importance returns a tuple."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        result = compute_shap_importance(model, X, feature_names, verbose=False)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_dataframe_and_array(self, sample_xgb_pipeline_with_preprocessor):
        """Test that result contains DataFrame and ndarray."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        shap_df, shap_values = compute_shap_importance(model, X, feature_names, verbose=False)
        assert isinstance(shap_df, pd.DataFrame)
        assert isinstance(shap_values, np.ndarray)

    def test_dataframe_has_required_columns(self, sample_xgb_pipeline_with_preprocessor):
        """Test that DataFrame has required columns."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        shap_df, _ = compute_shap_importance(model, X, feature_names, verbose=False)
        assert 'feature' in shap_df.columns
        assert 'shap_importance' in shap_df.columns
        assert 'mean_shap' in shap_df.columns

    def test_shap_values_shape(self, sample_xgb_pipeline_with_preprocessor):
        """Test that SHAP values have correct shape."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        _, shap_values = compute_shap_importance(model, X, feature_names, verbose=False)
        assert shap_values.shape == (len(X), len(feature_names))

    def test_shap_importance_non_negative(self, sample_xgb_pipeline_with_preprocessor):
        """Test that SHAP importance (mean abs) is non-negative."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        shap_df, _ = compute_shap_importance(model, X, feature_names, verbose=False)
        assert (shap_df['shap_importance'] >= 0).all()

    def test_sorted_by_importance_descending(self, sample_xgb_pipeline_with_preprocessor):
        """Test that results are sorted by importance descending."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        shap_df, _ = compute_shap_importance(model, X, feature_names, verbose=False)
        importance_values = shap_df['shap_importance'].values
        assert all(importance_values[i] >= importance_values[i+1]
                   for i in range(len(importance_values)-1))

    def test_verbose_mode_runs(self, sample_xgb_pipeline_with_preprocessor, capsys):
        """Test that verbose mode runs and produces output."""
        model, X, feature_names = sample_xgb_pipeline_with_preprocessor
        compute_shap_importance(model, X, feature_names, verbose=True)
        captured = capsys.readouterr()
        assert "SHAP" in captured.out


class TestCompareImportanceMethods:
    """Tests for compare_importance_methods function."""

    @pytest.fixture
    def sample_importance_dfs(self):
        """Create sample importance DataFrames for comparison."""
        features = [f'feature_{i}' for i in range(10)]

        # Create gain importance (sorted)
        gain_df = pd.DataFrame({
            'feature': features,
            'importance': np.linspace(0.2, 0.02, 10)
        })

        # Create SHAP importance (slightly different order)
        shap_values = np.linspace(0.2, 0.02, 10)
        # Swap a couple of features
        shap_values[2], shap_values[5] = shap_values[5], shap_values[2]

        shap_df = pd.DataFrame({
            'feature': features,
            'shap_importance': shap_values,
            'mean_shap': np.random.randn(10) * 0.1
        }).sort_values('shap_importance', ascending=False)

        return gain_df, shap_df

    def test_returns_dataframe(self, sample_importance_dfs):
        """Test that compare_importance_methods returns a DataFrame."""
        gain_df, shap_df = sample_importance_dfs
        result = compare_importance_methods(gain_df, shap_df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self, sample_importance_dfs):
        """Test that result has required columns."""
        gain_df, shap_df = sample_importance_dfs
        result = compare_importance_methods(gain_df, shap_df)
        required_cols = ['feature', 'importance', 'shap_importance', 'gain_rank', 'shap_rank', 'rank_diff']
        for col in required_cols:
            assert col in result.columns

    def test_respects_top_n(self, sample_importance_dfs):
        """Test that top_n parameter is respected."""
        gain_df, shap_df = sample_importance_dfs
        result = compare_importance_methods(gain_df, shap_df, top_n=5)
        assert len(result) == 5


class TestPrintShapImportanceSummary:
    """Tests for print_shap_importance_summary function."""

    @pytest.fixture
    def sample_shap_data(self):
        """Create sample SHAP data for testing."""
        np.random.seed(42)
        n_features = 20
        n_samples = 100

        shap_df = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(n_features)],
            'shap_importance': np.linspace(0.5, 0.01, n_features),
            'mean_shap': np.random.randn(n_features) * 0.1
        })

        shap_values = np.random.randn(n_samples, n_features) * 0.1

        return shap_df, shap_values

    def test_runs_without_error(self, sample_shap_data, capsys):
        """Test that print_shap_importance_summary runs without error."""
        shap_df, shap_values = sample_shap_data
        print_shap_importance_summary(shap_df, shap_values)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_prints_direction_indicators(self, sample_shap_data, capsys):
        """Test that function prints direction indicators."""
        shap_df, shap_values = sample_shap_data
        print_shap_importance_summary(shap_df, shap_values)
        captured = capsys.readouterr()
        # Should have direction indicators
        assert "fraud" in captured.out


class TestPrintImportanceComparison:
    """Tests for print_importance_comparison function."""

    @pytest.fixture
    def sample_comparison_df(self):
        """Create sample comparison DataFrame."""
        return pd.DataFrame({
            'feature': ['feature_0', 'feature_1', 'feature_2'],
            'importance': [0.3, 0.2, 0.1],
            'shap_importance': [0.25, 0.22, 0.12],
            'mean_shap': [0.1, -0.05, 0.02],
            'gain_rank': [1, 2, 3],
            'shap_rank': [2, 1, 3],
            'rank_diff': [-1, 1, 0],
            'avg_rank': [1.5, 1.5, 3.0]
        })

    def test_runs_without_error(self, sample_comparison_df, capsys):
        """Test that print_importance_comparison runs without error."""
        print_importance_comparison(sample_comparison_df)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_prints_correlation(self, sample_comparison_df, capsys):
        """Test that function prints Spearman correlation."""
        print_importance_comparison(sample_comparison_df)
        captured = capsys.readouterr()
        assert "Spearman" in captured.out or "correlation" in captured.out
