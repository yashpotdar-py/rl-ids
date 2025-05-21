"""
Test suite for the CICIDS2017 data extraction and preprocessing module.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

from rl_ids.features.extract import (
    load_dataset,
    preprocess_dataset,
    save_parquet,
    main,
    LABEL_COLUMN,
    app
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking CICIDS2017 data."""
    return pd.DataFrame({
        'Destination Port': [80, 443, 22, 8080],
        'Flow Duration': [100, 200, 300, 400],
        'Total Fwd Packets': [10, 20, 30, 40],
        'Total Backward Packets': [5, 10, 15, 20],
        'Total Length of Fwd Packets': [1000, 2000, 3000, 4000],
        'Total Length of Bwd Packets': [500, 600, 700, 800],
        'Fwd Packet Length Max': [100, 200, 300, 400],
        'Fwd Packet Length Min': [10, 20, 30, 40],
        'Fwd Packet Length Mean': [50, 60, 70, 80],
        'Fwd Packet Length Std': [5, 10, 15, 20],
        'Bwd Packet Length Max': [100, 200, 300, 400],
        'Bwd Packet Length Min': [10, 20, 30, 40],
        'Bwd Packet Length Mean': [50, 60, 70, 80],
        'Bwd Packet Length Std': [5, 10, 15, 20],
        'Flow Bytes/s': [1000, 2000, 3000, 4000],
        'Flow Packets/s': [10, 20, 30, 40],
        'Fwd PSH Flags': [0, 1, 0, 1],
        'Bwd PSH Flags': [0, 0, 1, 1],
        'Fwd URG Flags': [0, 0, 0, 1],
        'Bwd URG Flags': [0, 0, 0, 1],
        'Label': ['BENIGN', 'DoS', 'BENIGN', 'PortScan']
    })


@pytest.fixture
def temp_csv_path():
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tf:
        filepath = Path(tf.name)
    yield filepath
    # Clean up after test
    if filepath.exists():
        os.unlink(filepath)


@pytest.fixture
def temp_parquet_path():
    """Create a temporary parquet filepath for testing."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
        filepath = Path(tf.name)
    yield filepath
    # Clean up after test
    if filepath.exists():
        os.unlink(filepath)


class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_dataset_default_columns(self, sample_df, temp_csv_path):
        """Test loading dataset with default columns."""
        sample_df.to_csv(temp_csv_path, index=False)

        with patch('rl_ids.features.extract.DEFAULT_COLUMNS', list(sample_df.columns)) as mock_default_cols, \
                patch('rl_ids.features.extract.logger') as mock_logger:
            result = load_dataset(temp_csv_path)

            # Verify all default columns were loaded
            for col in sample_df.columns:
                assert col in result.columns

            # Verify logger was called
            mock_logger.info.assert_any_call(f"Loading data from {temp_csv_path}")

    def test_load_dataset_custom_columns(self, sample_df, temp_csv_path):
        """Test loading dataset with custom columns."""
        sample_df.to_csv(temp_csv_path, index=False)

        custom_columns = ['Destination Port', 'Flow Duration', 'Label']
        result = load_dataset(temp_csv_path, custom_columns)

        # Verify only specified columns were loaded
        assert list(result.columns) == custom_columns
        assert len(result.columns) == 3

    def test_file_not_found(self):
        """Test handling of non-existent file."""
        non_existent_path = Path("/tmp/non_existent_file.csv")

        with pytest.raises(FileNotFoundError):
            load_dataset(non_existent_path)

    def test_empty_file(self, temp_csv_path):
        """Test handling of empty file."""
        # Create empty file
        with open(temp_csv_path, 'w') as f:
            pass

        with pytest.raises(pd.errors.EmptyDataError):
            load_dataset(temp_csv_path)

    def test_missing_columns(self, temp_csv_path):
        """Test handling of missing required columns."""
        # Create a CSV with just a subset of required columns
        df = pd.DataFrame({
            'Destination Port': [80, 443],
            'Flow Duration': [100, 200]
            # Missing 'Label' column
        })
        df.to_csv(temp_csv_path, index=False)

        with pytest.raises(ValueError, match="Missing required columns"):
            load_dataset(temp_csv_path, ['Destination Port', 'Flow Duration', 'Label'])


class TestPreprocessDataset:
    """Tests for the preprocess_dataset function."""

    def test_successful_preprocessing(self, sample_df):
        """Test successful preprocessing with all necessary columns."""
        with patch('rl_ids.features.extract.logger') as mock_logger:
            result = preprocess_dataset(sample_df)

            # Verify labels were converted to binary
            assert set(result[LABEL_COLUMN].unique()) <= {0, 1}

            # Check binary conversion logic
            assert result.iloc[0][LABEL_COLUMN] == 0  # First row was 'BENIGN'
            assert result.iloc[1][LABEL_COLUMN] == 1  # Second row was 'DoS'

            # Verify no NaNs remain
            assert not result.isna().any().any()

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        with pytest.raises(ValueError, match="Cannot preprocess empty DataFrame"):
            preprocess_dataset(pd.DataFrame())

    def test_missing_label_column(self):
        """Test handling of DataFrame without Label column."""
        df = pd.DataFrame({'Destination Port': [80, 443]})

        with pytest.raises(ValueError, match=f"Required column '{LABEL_COLUMN}' missing"):
            preprocess_dataset(df)

    def test_handle_infinite_values(self):
        """Test handling of infinite values."""
        df = pd.DataFrame({
            'Destination Port': [80, 443],
            'Flow Duration': [100, np.inf],  # Infinite value
            'Label': ['BENIGN', 'DoS']
        })

        with patch('rl_ids.features.extract.logger'):
            result = preprocess_dataset(df)

            # Verify infinite value was handled (row dropped)
            assert len(result) == 1
            assert result['Destination Port'].iloc[0] == 80

    def test_handle_nan_values(self):
        """Test handling of NaN values."""
        df = pd.DataFrame({
            'Destination Port': [80, 443],
            'Flow Duration': [100, np.nan],  # NaN value
            'Label': ['BENIGN', 'DoS']
        })

        with patch('rl_ids.features.extract.logger'):
            result = preprocess_dataset(df)

            # Verify NaN value was handled (row dropped)
            assert len(result) == 1
            assert result['Destination Port'].iloc[0] == 80


class TestSaveParquet:
    """Tests for the save_parquet function."""

    def test_successful_save(self, sample_df, temp_parquet_path):
        """Test successful saving of DataFrame to Parquet."""
        with patch('rl_ids.features.extract.logger') as mock_logger:
            save_parquet(sample_df, temp_parquet_path)

            # Verify file was created
            assert temp_parquet_path.exists()

            # Verify data integrity
            saved_df = pd.read_parquet(temp_parquet_path)
            assert_frame_equal(sample_df, saved_df)

            # Verify logger was called
            mock_logger.success.assert_any_call(f"Successfully saved data to {temp_parquet_path}")

    def test_empty_dataframe(self, temp_parquet_path):
        """Test handling of empty DataFrame."""
        with pytest.raises(ValueError, match="Cannot save empty DataFrame"):
            save_parquet(pd.DataFrame(), temp_parquet_path)

    def test_create_parent_directories(self, sample_df):
        """Test parent directories are created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            complex_path = Path(tmpdir) / "nested" / "directories" / "file.parquet"

            with patch('rl_ids.features.extract.logger'):
                save_parquet(sample_df, complex_path)

                # Verify file was created with parent directories
                assert complex_path.exists()


class TestMain:
    """Tests for the main function."""

    def test_successful_execution(self, sample_df, temp_csv_path, temp_parquet_path):
        """Test successful end-to-end execution of main function."""
        # Set up sample data
        sample_df.to_csv(temp_csv_path, index=False)

        # Mock dependent functions to isolate test
        with patch('rl_ids.features.extract.load_dataset', return_value=sample_df) as mock_load, \
                patch('rl_ids.features.extract.preprocess_dataset', return_value=sample_df) as mock_preprocess, \
                patch('rl_ids.features.extract.save_parquet') as mock_save, \
                patch('rl_ids.features.extract.logger'):

            # Execute main function
            main(input_path=temp_csv_path, output_path=temp_parquet_path)

            # Verify function calls
            mock_load.assert_called_once_with(temp_csv_path, None)
            mock_preprocess.assert_called_once_with(sample_df)
            mock_save.assert_called_once_with(sample_df, temp_parquet_path)

    def test_error_handling(self, temp_csv_path, temp_parquet_path):
        """Test error handling in main function."""
        # Set up mocks to simulate error
        with patch('rl_ids.features.extract.load_dataset', side_effect=ValueError("Test error")), \
                patch('rl_ids.features.extract.logger') as mock_logger:

            # Execute with error expectation
            with pytest.raises(ValueError, match="Test error"):
                main(input_path=temp_csv_path, output_path=temp_parquet_path)

            # Verify logger recorded error
            mock_logger.error.assert_called_with("Preprocessing failed: Test error")


@pytest.mark.parametrize("label,expected", [
    ("BENIGN", 0),
    ("DoS", 1),
    ("PortScan", 1),
    ("FTP-Patator", 1),
    ("SSH-Patator", 1),
    ("benign", 1),  # Case matters
    ("", 1),        # Empty string isn't "BENIGN"
])
def test_label_conversion(label, expected):
    """Test label conversion logic for different attack types."""
    df = pd.DataFrame({'Label': [label]})

    with patch('rl_ids.features.extract.logger'):
        result = preprocess_dataset(df)
        assert result['Label'].iloc[0] == expected
