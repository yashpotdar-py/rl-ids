"""
Tests for the extract module that handles CICIDS2017 network traffic data preprocessing.
"""
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rl_ids.features.extract import (
    load_dataset,
    preprocess_dataset,
    save_parquet,
    DEFAULT_COLUMNS,
    LABEL_COLUMN,
)


@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Destination Port': [80, 443, 53],
        'Flow Duration': [100, 200, 300],
        'Total Fwd Packets': [5, 10, 15],
        'Total Length of Fwd Packets': [500, 1000, 1500],
        'Label': ['BENIGN', 'DoS', 'BENIGN']
    })


@pytest.fixture
def sample_csv(tmp_path, sample_data):
    """Create a sample CSV file with test data."""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_logger():
    """Mock logger to avoid console output during tests."""
    with patch('rl_ids.features.extract.logger') as mock:
        yield mock


class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_successful_load_default_columns(self, sample_csv, mock_logger):
        """Test loading dataset with default columns."""
        # Only use columns that exist in our test data
        test_columns = [col for col in DEFAULT_COLUMNS if col in ['Destination Port', 'Flow Duration', 
                                                                 'Total Fwd Packets', 'Total Length of Fwd Packets', 'Label']]
        
        with patch('rl_ids.features.extract.DEFAULT_COLUMNS', test_columns):
            df = load_dataset(sample_csv)
            
        assert len(df) == 3
        assert list(df.columns) == test_columns
        mock_logger.info.assert_called()

    def test_custom_columns(self, sample_csv, mock_logger):
        """Test loading dataset with custom columns."""
        custom_columns = ['Destination Port', 'Label']
        df = load_dataset(sample_csv, custom_columns)
        
        assert len(df) == 3
        assert list(df.columns) == custom_columns

    def test_file_not_found(self, tmp_path):
        """Test handling of missing file."""
        non_existent_path = tmp_path / "does_not_exist.csv"
        
        with pytest.raises(FileNotFoundError):
            load_dataset(non_existent_path)

    def test_missing_columns(self, sample_csv):
        """Test error when required columns are missing."""
        missing_columns = ['Missing Column 1', 'Missing Column 2']
        
        with pytest.raises(ValueError) as excinfo:
            load_dataset(sample_csv, missing_columns)
        
        assert "Missing required columns" in str(excinfo.value)


class TestPreprocessDataset:
    """Tests for the preprocess_dataset function."""

    def test_label_conversion(self, sample_data, mock_logger):
        """Test conversion of labels to binary format."""
        df = preprocess_dataset(sample_data)
        
        assert df[LABEL_COLUMN].tolist() == [0, 1, 0]
        mock_logger.info.assert_called()

    def test_handle_inf_values(self, mock_logger):
        """Test handling of infinite values."""
        test_data = pd.DataFrame({
            'Destination Port': [80, 443, 53],
            'Flow Duration': [100, np.inf, 300],
            'Total Fwd Packets': [5, 10, -np.inf],
            'Label': ['BENIGN', 'DoS', 'BENIGN']
        })
        
        df = preprocess_dataset(test_data)
        
        # Should drop rows with inf/-inf
        assert len(df) == 1
        assert df['Destination Port'].iloc[0] == 80

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError) as excinfo:
            preprocess_dataset(empty_df)
        
        assert "Cannot preprocess empty DataFrame" in str(excinfo.value)

    def test_missing_label_column(self):
        """Test handling of DataFrame without Label column."""
        df_no_label = pd.DataFrame({
            'Destination Port': [80, 443, 53],
            'Flow Duration': [100, 200, 300]
        })
        
        with pytest.raises(ValueError) as excinfo:
            preprocess_dataset(df_no_label)
        
        assert f"Required column '{LABEL_COLUMN}'" in str(excinfo.value)


class TestSaveParquet:
    """Tests for the save_parquet function."""

    def test_successful_save(self, sample_data, tmp_path, mock_logger):
        """Test successful saving to Parquet."""
        out_path = tmp_path / "test_output" / "data.parquet"
        
        save_parquet(sample_data, out_path)
        
        assert out_path.exists()
        mock_logger.success.assert_called_once()
        
        # Verify the saved data can be loaded back
        loaded_df = pd.read_parquet(out_path)
        assert_frame_equal(loaded_df, sample_data)

    def test_empty_dataframe(self, tmp_path):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame()
        out_path = tmp_path / "empty.parquet"
        
        with pytest.raises(ValueError) as excinfo:
            save_parquet(empty_df, out_path)
        
        assert "Cannot save empty DataFrame" in str(excinfo.value)
        assert not out_path.exists()

    def test_directory_creation_error(self, sample_data):
        """Test handling of directory creation errors."""
        # Create a path where directory creation will fail
        with tempfile.NamedTemporaryFile() as tmp_file:
            invalid_dir_path = Path(tmp_file.name) / "impossible_subdir" / "data.parquet"
            
            with pytest.raises(OSError):
                save_parquet(sample_data, invalid_dir_path)


@patch('rl_ids.features.extract.load_dataset')
@patch('rl_ids.features.extract.preprocess_dataset')
@patch('rl_ids.features.extract.save_parquet')
def test_main_integration(mock_save, mock_preprocess, mock_load, sample_data, mock_logger):
    """Integration test for the main function."""
    from rl_ids.features.extract import main
    
    # Configure mocks
    mock_load.return_value = sample_data
    mock_preprocess.return_value = sample_data
    mock_save.return_value = None
    
    # Call with test paths
    test_input = Path("test_input.csv")
    test_output = Path("test_output.parquet")
    
    # Run main function
    main(input_path=test_input, output_path=test_output)
    
    # Verify function calls
    mock_load.assert_called_once_with(test_input, None)
    mock_preprocess.assert_called_once_with(sample_data)
    mock_save.assert_called_once_with(sample_data, test_output)
    mock_logger.success.assert_called_with("Preprocessing completed successfully")


@pytest.mark.skip(reason="Integration test that requires actual files")
def test_actual_csv_processing():
    """
    Real integration test with actual CSV files.
    
    This test is skipped by default as it requires actual data files.
    Remove the skip decorator to run this test with real data.
    """
    from rl_ids.features.extract import main
    
    # Paths to actual test files (adjust as needed)
    test_input = Path("tests/test_data/sample_cicids.csv")
    test_output = Path("tests/test_data/output.parquet")
    
    # Run the main function
    main(input_path=test_input, output_path=test_output)
    
    # Verify output file exists and has expected properties
    assert test_output.exists()
    df = pd.read_parquet(test_output)
    assert not df.empty
    assert LABEL_COLUMN in df.columns
    assert set(df[LABEL_COLUMN].unique()).issubset({0, 1})