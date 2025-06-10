import os
from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import typer

from rl_ids.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


def load_and_preprocess_data(data_dir: Path = RAW_DATA_DIR):
    """Load and preprocess CSV data files from the specified directory."""
    logger.info(
        f"Starting data processing pipeline from directory: {data_dir}")

    # Validate input directory
    if not data_dir.exists():
        logger.error(f"Data directory does not exist: {data_dir}")
        raise typer.Exit(1)

    # Find CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    if not csv_files:
        logger.warning(f"No CSV files found in directory: {data_dir}")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} CSV files to process")

    # Load CSV files with progress tracking
    data_frames = []
    for file in tqdm(csv_files, desc="Loading CSV files", unit="file"):
        file_path = os.path.join(data_dir, file)
        logger.debug(f"Loading file: {file}")

        try:
            df = pd.read_csv(file_path, low_memory=False)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            data_frames.append(df)
            logger.debug(f"Successfully loaded {file} with {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to load {file}: {str(e)}")
            continue

    if not data_frames:
        logger.error("No data frames were successfully loaded")
        raise typer.Exit(1)

    # Concatenate data frames
    logger.info("Concatenating data frames...")
    data = pd.concat(data_frames, ignore_index=True)
    logger.info(f"Combined dataset shape: {data.shape}")

    # Data preprocessing steps
    logger.info("Starting data preprocessing...")

    # Strip column names again after concatenation (safety measure)
    data.columns = data.columns.str.strip()
    logger.info("Stripped whitespace from column names")

    # Remove columns with all null values
    initial_cols = data.shape[1]
    data.dropna(axis=1, how="all", inplace=True)
    removed_cols = initial_cols - data.shape[1]
    if removed_cols > 0:
        logger.info(f"Removed {removed_cols} columns with all null values")

    # Fill remaining null values
    null_count = data.isnull().sum().sum()
    if null_count > 0:
        logger.info(f"Filling {null_count} null values with 0")
        data.fillna(0, inplace=True)

    # Encode labels
    if "Label" in data.columns:
        logger.info("Encoding labels...")
        # Keep original label column
        data["Label_Original"] = data["Label"].copy()
        # Encode the Label column
        label_encoder = LabelEncoder()
        data["Label"] = label_encoder.fit_transform(data["Label"])
        unique_labels = len(label_encoder.classes_)
        logger.info(f"Encoded {unique_labels} unique labels")
        logger.info("Original labels preserved in 'Label_Original' column")
    else:
        logger.warning("No 'Label' column found in dataset")

    logger.success(
        f"Data processing complete. Final dataset shape: {data.shape}")

    return data


@app.command()
def main(raw_dir: Path = RAW_DATA_DIR, processed_dir: Path = PROCESSED_DATA_DIR):
    """Main command to process raw data and save to processed directory."""
    logger.info("Starting main data processing command")

    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {processed_dir}")

    # Load and preprocess data
    data = load_and_preprocess_data(raw_dir)

    if data.empty:
        logger.error("No data to save - preprocessing returned empty dataset")
        raise typer.Exit(1)

    # Save processed data
    processed_file_path = processed_dir / "cicids2017_processed.csv"
    logger.info(f"Saving processed data to: {processed_file_path}")

    try:
        data.to_csv(processed_file_path, index=False)
        logger.success(
            f"Successfully saved {len(data)} rows to {processed_file_path}")
    except Exception as e:
        logger.error(f"Failed to save processed data: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
