"""
CICIDS2017 Network Traffic Data Preprocessor

This script normalizes and cleans CICIDS2017 network traffic data,
extracting specific features and converting attack labels to binary format.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Default columns to keep in preprocessing
DEFAULT_COLUMNS = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
    'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward', 'Idle Mean', 'Idle Std', 'Idle Max',
    'Idle Min', 'Label'
]

LABEL_COLUMN = "Label"


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess CICIDS2017 network traffic data for machine learning'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=False,
        default="data/raw/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv",
        help='Path to input CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=False,
        default="data/processed/cleaned.parquet",
        help='Path to output Parquet file'
    )
    parser.add_argument(
        '--columns', '-c',
        type=str,
        nargs='+',
        help='Optional list of columns to keep (defaults to predefined set)'
    )
    return parser.parse_args()


def load_dataset(csv_path: str, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load CICIDS2017 CSV dataset with specified columns.

    Args:
        csv_path: Path to the CSV file
        columns: Optional list of columns to keep (defaults to predefined set)

    Returns:
        DataFrame with selected columns

    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
        ValueError: If required columns are missing from the CSV
    """
    # Use default columns if not provided
    if columns is None:
        columns = DEFAULT_COLUMNS

    # Validate input file
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    logger.info(f"Loading data from {csv_path}")

    try:
        # Read CSV with optimized settings
        df = pd.read_csv(
            csv_path,
            low_memory=False,  # Prevent mixed type inference warnings
            skipinitialspace=True,  # Handle potential whitespace in column names
            encoding='utf-8',  # Specify encoding
            on_bad_lines='warn'  # Warn on problematic lines
        )
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Input file is empty: {csv_path}")
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

    # Verify all required columns exist
    missing_cols = set(columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    # Select only the specified columns
    df = df[columns]

    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset by normalizing labels to binary and handling invalid values.

    Args:
        df: Input DataFrame to preprocess

    Returns:
        Preprocessed DataFrame

    Raises:
        ValueError: If the DataFrame is empty or Label column is missing
    """
    if df.empty:
        raise ValueError("Cannot preprocess empty DataFrame")

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Required column '{LABEL_COLUMN}' missing from DataFrame")

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Normalize labels to binary (0 for BENIGN, 1 for any attack)
    logger.info(f"Converting {LABEL_COLUMN} to binary format")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].apply(
        lambda x: 0 if isinstance(x, str) and x.strip() == "BENIGN" else 1
    )

    # Count attack distribution
    attack_count = df[LABEL_COLUMN].value_counts()
    logger.info(f"Attack distribution - Benign: {attack_count.get(0, 0)}, Attack: {attack_count.get(1, 0)}")

    # Handle infinite values
    logger.info("Replacing infinite values with NaN")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Check for and report missing values before dropping
    missing_before = df.isna().sum().sum()
    if missing_before > 0:
        logger.warning(f"Found {missing_before} missing values in the dataset")

    # Drop rows with NaN values
    df.dropna(inplace=True)

    # Report data size after cleaning
    logger.info(f"Dataset after cleaning: {len(df)} rows")

    return df


def save_parquet(df: pd.DataFrame, out_path: str) -> None:
    """
    Write DataFrame to Parquet for fast downstream loading.

    Args:
        df: DataFrame to save
        out_path: Output Parquet file path

    Raises:
        OSError: If directory creation fails
        ValueError: If DataFrame is empty
    """
    if df.empty:
        raise ValueError("Cannot save empty DataFrame to Parquet")

    # Create output directory if it doesn't exist
    out_path = Path(out_path)
    try:
        os.makedirs(out_path.parent, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory: {e}")
        raise

    # Save to Parquet with compression
    try:
        logger.info(f"Saving {len(df)} rows to {out_path}")
        df.to_parquet(
            out_path,
            index=False,
            compression='snappy'  # Good balance of compression/speed
        )
        logger.info(f"Successfully saved data to {out_path}")
    except Exception as e:
        logger.error(f"Failed to save Parquet file: {e}")
        raise


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Load dataset
        df = load_dataset(args.input, args.columns)

        # Preprocess dataset
        df = preprocess_dataset(df)

        # Save to Parquet
        save_parquet(df, args.output)

        logger.info("Preprocessing completed successfully")

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
