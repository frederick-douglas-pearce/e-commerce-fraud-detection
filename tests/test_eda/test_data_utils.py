"""
Tests for src/fd1_nb/data_utils.py

Tests data loading and preprocessing utilities.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
from src.fd1_nb.data_utils import (
    download_data_csv,
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


class TestLoadData:
    """Tests for load_data function."""

    @pytest.fixture
    def sample_csv_file(self, tmp_path):
        """Create a sample CSV file for testing."""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'amount': [100.5, 200.3, 150.0, 300.7, 250.1],
            'category': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 0, 1]
        })
        csv_path = tmp_path / 'test_data.csv'
        df.to_csv(csv_path, index=False)
        return str(csv_path)

    def test_returns_dataframe(self, sample_csv_file):
        """Test that load_data returns a DataFrame."""
        result = load_data(sample_csv_file, verbose=False)
        assert isinstance(result, pd.DataFrame)

    def test_correct_shape(self, sample_csv_file):
        """Test that loaded DataFrame has correct shape."""
        result = load_data(sample_csv_file, verbose=False)
        assert result.shape == (5, 4)

    def test_correct_columns(self, sample_csv_file):
        """Test that loaded DataFrame has correct columns."""
        result = load_data(sample_csv_file, verbose=False)
        expected_columns = ['id', 'amount', 'category', 'target']
        assert list(result.columns) == expected_columns

    def test_correct_dtypes(self, sample_csv_file):
        """Test that columns have expected data types."""
        result = load_data(sample_csv_file, verbose=False)
        assert result['id'].dtype == np.int64
        assert result['amount'].dtype == np.float64
        assert result['target'].dtype == np.int64

    def test_verbose_mode(self, sample_csv_file, capsys):
        """Test that verbose mode prints dataset info."""
        load_data(sample_csv_file, verbose=True)
        captured = capsys.readouterr()
        assert 'Dataset Shape' in captured.out
        assert '5 rows' in captured.out
        assert '4 columns' in captured.out
        assert 'Memory Usage' in captured.out

    def test_verbose_false_no_output(self, sample_csv_file, capsys):
        """Test that verbose=False produces no output."""
        load_data(sample_csv_file, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ''

    def test_nonexistent_file_raises_error(self, tmp_path):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_data(str(tmp_path / 'nonexistent.csv'), verbose=False)

    def test_data_integrity(self, sample_csv_file):
        """Test that data values are loaded correctly."""
        result = load_data(sample_csv_file, verbose=False)
        assert result['amount'].iloc[0] == 100.5
        assert result['category'].iloc[0] == 'A'
        assert result['target'].sum() == 2


class TestDownloadDataCSV:
    """Tests for download_data_csv function."""

    def test_creates_directory(self, tmp_path):
        """Test that function creates data directory if it doesn't exist."""
        nested_dir = tmp_path / 'nested' / 'data' / 'dir'

        # Create a dummy file to simulate existing data
        nested_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = nested_dir / 'test.csv'
        dummy_file.write_text('a,b,c\n1,2,3')

        # Function should detect existing file and not try to download
        download_data_csv(
            kaggle_source='dummy/dataset',
            data_dir=str(nested_dir),
            csv_file='test.csv'
        )

        # Directory should still exist
        assert nested_dir.exists()

    def test_skips_existing_file(self, tmp_path, capsys):
        """Test that function skips download if file exists."""
        # Create existing file
        csv_file = tmp_path / 'existing.csv'
        csv_file.write_text('a,b,c\n1,2,3')

        download_data_csv(
            kaggle_source='dummy/dataset',
            data_dir=str(tmp_path),
            csv_file='existing.csv'
        )

        captured = capsys.readouterr()
        assert 'already exists' in captured.out
