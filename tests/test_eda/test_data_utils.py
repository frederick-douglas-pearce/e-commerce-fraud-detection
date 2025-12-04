"""
Tests for src/eda/data_utils.py

Tests data loading and preprocessing utilities.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
from src.eda.data_utils import (
    load_data,
    split_train_val_test,
    analyze_target_stats,
    analyze_feature_stats
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'id': range(n),
        'amount': np.random.uniform(10, 500, n),
        'age': np.random.randint(18, 70, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'channel': np.random.choice(['web', 'mobile', 'app'], n),
        'target': np.random.choice([0, 1], n, p=[0.95, 0.05])
    })


class TestSplitTrainValTest:
    """Tests for split_train_val_test function."""

    def test_split_ratios(self, sample_df):
        """Test that splits match specified ratios."""
        train, val, test = split_train_val_test(
            sample_df,
            target_col='target',
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            verbose=False
        )

        total = len(sample_df)
        assert len(train) == int(total * 0.6)
        assert len(val) == int(total * 0.2)
        # Test set gets the remainder
        assert len(train) + len(val) + len(test) == total

    def test_stratification(self, sample_df):
        """Test that stratification preserves target distribution."""
        train, val, test = split_train_val_test(
            sample_df,
            target_col='target',
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            verbose=False
        )

        orig_dist = sample_df['target'].mean()
        train_dist = train['target'].mean()
        val_dist = val['target'].mean()
        test_dist = test['target'].mean()

        # All distributions should be within 2% of original
        assert abs(train_dist - orig_dist) < 0.02
        assert abs(val_dist - orig_dist) < 0.02
        assert abs(test_dist - orig_dist) < 0.02

    def test_no_overlap(self, sample_df):
        """Test that splits have no overlapping rows."""
        train, val, test = split_train_val_test(
            sample_df,
            target_col='target',
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            verbose=False
        )

        # Check no overlap between sets using IDs
        train_ids = set(train['id'])
        val_ids = set(val['id'])
        test_ids = set(test['id'])

        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_without_stratification(self, sample_df):
        """Test split without stratification."""
        train, val, test = split_train_val_test(
            sample_df,
            target_col=None,  # No stratification
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            verbose=False
        )

        total = len(sample_df)
        assert len(train) + len(val) + len(test) == total

    def test_invalid_ratios(self, sample_df):
        """Test that invalid ratios raise ValueError."""
        with pytest.raises(ValueError):
            split_train_val_test(
                sample_df,
                target_col='target',
                train_ratio=0.5,
                val_ratio=0.3,
                test_ratio=0.3,  # Sum = 1.1
                random_state=42
            )


class TestAnalyzeTargetStats:
    """Tests for analyze_target_stats function."""

    def test_returns_dict(self, sample_df):
        """Test that function returns a dictionary."""
        result = analyze_target_stats(
            sample_df,
            target_col='target',
            plot=False
        )

        assert isinstance(result, dict)
        assert 'distribution' in result
        assert 'imbalance_ratio' in result
        assert 'is_imbalanced' in result

    def test_imbalance_detection(self, sample_df):
        """Test imbalance detection logic."""
        # Create highly imbalanced dataset
        imbalanced_df = pd.DataFrame({
            'target': [0] * 950 + [1] * 50
        })

        result = analyze_target_stats(
            imbalanced_df,
            target_col='target',
            imbalance_threshold=10.0,
            plot=False
        )

        assert result['is_imbalanced'] == True
        assert result['imbalance_ratio'] > 10.0

    def test_balanced_dataset(self):
        """Test with balanced dataset."""
        balanced_df = pd.DataFrame({
            'target': [0] * 500 + [1] * 500
        })

        result = analyze_target_stats(
            balanced_df,
            target_col='target',
            imbalance_threshold=2.0,
            plot=False
        )

        assert result['is_imbalanced'] == False
        assert result['imbalance_ratio'] == 1.0


class TestAnalyzeFeatureStats:
    """Tests for analyze_feature_stats function."""

    def test_categorical_analysis(self, sample_df, capsys):
        """Test categorical feature analysis."""
        analyze_feature_stats(
            sample_df,
            categorical_features=['category', 'channel'],
            numeric_features=[],
            top_n=3
        )

        captured = capsys.readouterr()
        assert 'Categorical Features' in captured.out
        assert 'category' in captured.out.lower()
        assert 'channel' in captured.out.lower()

    def test_numeric_analysis(self, sample_df, capsys):
        """Test numeric feature analysis."""
        analyze_feature_stats(
            sample_df,
            categorical_features=[],
            numeric_features=['amount', 'age'],
            top_n=3
        )

        captured = capsys.readouterr()
        assert 'Numeric Features' in captured.out
        assert 'amount' in captured.out.lower()
        assert 'age' in captured.out.lower()

    def test_empty_features(self, sample_df):
        """Test with empty feature lists."""
        # Should not raise error
        analyze_feature_stats(
            sample_df,
            categorical_features=[],
            numeric_features=[]
        )
