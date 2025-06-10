import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.utils import resample
from loguru import logger
import typer

from rl_ids.config import NORMALISED_DATA_FILE, PROCESSED_DATA_DIR, BALANCED_DATA_FILE


app = typer.Typer()


@app.command()
def main(
        input_path: Path = NORMALISED_DATA_FILE,
        output_path: Path = BALANCED_DATA_FILE,
        strategy='hybrid'):
    """Balance the dataset using various strategies"""

    logger.info("Loading dataset...")
    df = pd.read_csv(input_path)
    logger.info(f"Old dataset shape: {df.shape}")

    # Separate by class
    class_dfs = []
    for label in df['Label'].unique():
        class_df = df[df['Label'] == label]
        class_dfs.append((label, class_df))

    # Sort by size
    class_dfs.sort(key=lambda x: len(x[1]))

    if strategy == 'undersample':
        # Undersample the majority class
        min_size = len(class_dfs[-1][1])  # Second smallest class
        target_size = min(min_size * 10, 50000)  # Cap at 50k

    elif strategy == 'oversample':
        # Oversample minority classes
        max_size = len(class_dfs[-1][1])  # Largest class
        target_size = min(max_size // 5, 100000)  # Reduce largest class import

    elif strategy == 'hybrid':
        # Hybrid approach
        target_size = 50000

    balanced_dfs = []

    for label, class_df in class_dfs:
        current_size = len(class_df)

        if current_size > target_size:
            # Undersample
            resampled = resample(
                class_df, n_samples=target_size, random_state=42)
            logger.info(
                f"Class {label}: {current_size} -> {target_size} (Undersampled)")

        else:
            # Oversample
            resampled = resample(
                class_df, n_samples=target_size, random_state=42, replace=True)
            logger.info(
                f"Class {label}: {current_size} -> {target_size} (oversampled)")

        balanced_dfs.append(resampled)

    # Combine and shuffle
    logger.info(f"Combining and shuffling balance data")
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(
        frac=1, random_state=42).reset_index(drop=True)

    # Save
    balanced_df.to_csv(output_path, index=False)
    logger.success(f"Balanced dataset saved to {output_path}")
    logger.info(f"New dataset shape: {balanced_df.shape}")
    logger.info(
        f"New class distribution:\n{balanced_df['Label'].value_counts().sort_index()}")

    return output_path


if __name__ == "__main__":
    app()
