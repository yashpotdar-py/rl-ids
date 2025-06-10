import os
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from rl_ids.config import NORMALISED_DATA_FILE, PROCESSED_DATA_FILE

app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_FILE,
    output_path: Path = NORMALISED_DATA_FILE,
):
    logger.info("Normalizing dataset...")

    # Load the processed data
    logger.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)

    logger.info(f"Initial dataset shape: {df.shape}")

    # Drop label columns for features
    feature_cols = [col for col in df.columns if col not in [
        "Label", "Label_Original"]]
    X = df[feature_cols]
    y = df[["Label", "Label_Original"]]

    logger.info(f"Found {len(feature_cols)} feature columns")

    # Data validation and cleaning
    logger.info("Checking for infinite and extreme values...")

    # Check for infinite values
    inf_mask = np.isinf(X).any(axis=1)
    inf_count = inf_mask.sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} rows with infinite values")

        # Replace infinite values with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        logger.info("Replaced infinite values with NaN")

    # Check for NaN values
    nan_mask = X.isnull().any(axis=1)
    nan_count = nan_mask.sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} rows with NaN values")

        # Drop rows with NaN values
        valid_mask = ~nan_mask
        X = X[valid_mask]
        y = y[valid_mask]
        logger.info(f"Dropped {nan_count} rows with NaN values")
        logger.info(f"Dataset shape after cleaning: {X.shape}")

    # Check for extremely large values that might cause overflow
    max_safe_value = np.finfo(np.float64).max / 1e6  # Conservative threshold
    extreme_mask = (X.abs() > max_safe_value).any(axis=1)
    extreme_count = extreme_mask.sum()
    if extreme_count > 0:
        logger.warning(
            f"Found {extreme_count} rows with extremely large values")

        # Option 1: Cap extreme values
        X = X.clip(-max_safe_value, max_safe_value)
        logger.info(f"Capped extreme values to ±{max_safe_value:.2e}")

        # Option 2: Remove extreme rows (uncomment if preferred)
        # valid_mask = ~extreme_mask
        # X = X[valid_mask]
        # y = y[valid_mask]
        # logger.info(f"Removed {extreme_count} rows with extreme values")

    # Reset indices after potential row removal
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Normalize
    logger.info("Applying StandardScaler normalization...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

    # Add labels back
    df_scaled = pd.concat([X_scaled_df, y], axis=1)

    # Save
    logger.info(f"Saving normalized data to {output_path}")
    os.makedirs(output_path.parent, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)

    # Optional: Update config constants
    # config_path = Path("rl_ids/config.py")
    # logger.info(f"Updating config file at {config_path}")
    # with open(config_path, "a") as f:
    #     f.write(f"\nFEATURE_COLUMNS = {feature_cols}\n")
    #     f.write("LABEL_COLUMN = 'Label'\n")
    #     f.write(f"DATA_PATH = '{output_path}'\n")

    logger.success(
        f"Data normalization complete. Final shape: {df_scaled.shape}")


if __name__ == "__main__":
    app()
