"""
Tests for src/data/loader.py

Tests load_and_split_data function for data loading and splitting.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from src.deployment.data.loader import load_and_split_data
from src.deployment.config.data_config import DataConfig


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing"""
    # Create a small dataset with fraud and non-fraud cases
    data = {
        'is_fraud': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1] * 10,  # 100 samples, 20% fraud
        'amount': [10.0 + i for i in range(100)],
        'feature_1': [i * 2 for i in range(100)],
        'feature_2': [i * 3 for i in range(100)],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        sample_csv_data.to_csv(f.name, index=False)
        yield f.name
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


class TestLoadAndSplitData:
    """Tests for load_and_split_data function"""

    def test_load_and_split_returns_three_dataframes(self, temp_csv_file):
        """Test that function returns three DataFrames"""
        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            verbose=False
        )

        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

    def test_load_and_split_correct_ratios(self, temp_csv_file):
        """Test that split ratios are correct (60/20/20 by default)"""
        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            verbose=False
        )

        total = len(train_df) + len(val_df) + len(test_df)

        # Allow 1% tolerance for rounding
        assert abs(len(train_df) / total - 0.6) < 0.01
        assert abs(len(val_df) / total - 0.2) < 0.01
        assert abs(len(test_df) / total - 0.2) < 0.01

    def test_load_and_split_custom_test_size(self, temp_csv_file):
        """Test that custom test_size is respected"""
        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            test_size=0.3,
            val_size=0.25,  # 25% of train+val
            verbose=False
        )

        total = len(train_df) + len(val_df) + len(test_df)

        # test_size=0.3 means 30% test, 70% train+val
        # val_size=0.25 means 25% of 70% = 17.5% val
        # train = 70% - 17.5% = 52.5%
        assert abs(len(test_df) / total - 0.3) < 0.01

    def test_load_and_split_custom_random_seed(self, temp_csv_file):
        """Test that custom random_seed produces consistent results"""
        train_df1, val_df1, test_df1 = load_and_split_data(
            data_path=temp_csv_file,
            random_seed=42,
            verbose=False
        )

        train_df2, val_df2, test_df2 = load_and_split_data(
            data_path=temp_csv_file,
            random_seed=42,
            verbose=False
        )

        # Same random seed should produce same splits
        pd.testing.assert_frame_equal(train_df1, train_df2)
        pd.testing.assert_frame_equal(val_df1, val_df2)
        pd.testing.assert_frame_equal(test_df1, test_df2)

    def test_load_and_split_different_random_seeds(self, temp_csv_file):
        """Test that different random seeds produce different splits"""
        train_df1, val_df1, test_df1 = load_and_split_data(
            data_path=temp_csv_file,
            random_seed=42,
            verbose=False
        )

        train_df2, val_df2, test_df2 = load_and_split_data(
            data_path=temp_csv_file,
            random_seed=123,
            verbose=False
        )

        # Different random seeds should produce different splits
        with pytest.raises(AssertionError):
            pd.testing.assert_frame_equal(train_df1, train_df2)

    def test_load_and_split_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_and_split_data(
                data_path="nonexistent_file.csv",
                verbose=False
            )

    def test_load_and_split_stratification_maintains_fraud_rate(self, temp_csv_file):
        """Test that stratification maintains similar fraud rates across splits"""
        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            verbose=False
        )

        # Get fraud rates for each split
        train_fraud_rate = train_df[DataConfig.TARGET_COLUMN].mean()
        val_fraud_rate = val_df[DataConfig.TARGET_COLUMN].mean()
        test_fraud_rate = test_df[DataConfig.TARGET_COLUMN].mean()

        # All splits should have similar fraud rates (within 5% tolerance)
        assert abs(train_fraud_rate - 0.2) < 0.05
        assert abs(val_fraud_rate - 0.2) < 0.05
        assert abs(test_fraud_rate - 0.2) < 0.05

    def test_load_and_split_no_data_leakage(self, temp_csv_file):
        """Test that there's no overlap between train/val/test sets"""
        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            verbose=False
        )

        # Reset index to compare by index values
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)

        # Check no overlap
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

    def test_load_and_split_all_data_accounted_for(self, temp_csv_file):
        """Test that all rows are included in one of the splits"""
        # Load original data
        original_df = pd.read_csv(temp_csv_file)

        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            verbose=False
        )

        # Total samples should match
        assert len(train_df) + len(val_df) + len(test_df) == len(original_df)

    def test_load_and_split_uses_default_config(self, temp_csv_file):
        """Test that function uses DataConfig defaults when parameters not provided"""
        train_df, val_df, test_df = load_and_split_data(
            data_path=temp_csv_file,
            verbose=False
        )

        # Should use default split ratios (TEST_SIZE=0.2, VAL_SIZE=0.25)
        total = len(train_df) + len(val_df) + len(test_df)
        assert abs(len(test_df) / total - DataConfig.TEST_SIZE) < 0.01

    def test_load_and_split_verbose_false_no_print(self, temp_csv_file, capsys):
        """Test that verbose=False doesn't print output"""
        load_and_split_data(data_path=temp_csv_file, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_load_and_split_verbose_true_prints_info(self, temp_csv_file, capsys):
        """Test that verbose=True prints loading information"""
        load_and_split_data(data_path=temp_csv_file, verbose=True)

        captured = capsys.readouterr()
        assert "Loading raw transaction data" in captured.out
        assert "Total samples:" in captured.out
        assert "Training set:" in captured.out
        assert "Validation set:" in captured.out
        assert "Test set:" in captured.out
