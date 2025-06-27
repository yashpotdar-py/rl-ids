from pathlib import Path
from typing import List

import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

app = typer.Typer()


class IDSDetectionEnv(gym.Env):
    """Custom Environment for RL-Based IDS Detection"""

    def __init__(self, data_path: Path, feature_cols: List, label_col="Label"):
        super(IDSDetectionEnv, self).__init__()

        logger.info(f"Loading data from {data_path}")
        self.df = pd.read_csv(data_path)
        self.features = feature_cols
        self.label_col = label_col
        self.x = self.df[feature_cols].values.astype(np.float32)
        self.y = self.df[label_col].values

        self.current_step = 0
        self.total_steps = len(self.df)

        self.num_classes = len(np.unique(self.y))
        self.action_space = spaces.Discrete(self.num_classes)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(feature_cols),), dtype=np.float32
        )

        logger.info(
            f"Environment initialized with {self.total_steps} samples, {self.num_classes} classes"
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.x[self.current_step]
        info = {}
        return obs, info

    def step(self, action):
        actual_label = self.y[self.current_step]
        reward = 1 if action == actual_label else -1

        self.current_step += 1
        done = self.current_step >= self.total_steps
        truncated = False

        if not done:
            obs = self.x[self.current_step]
        else:
            obs = np.zeros_like(self.x[0])

        info = {"actual_label": actual_label, "predicted_action": action}
        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        if self.current_step < self.total_steps:
            print(f"Step: {self.current_step}, True: {self.y[self.current_step]}")


@app.command()
def train_env(
    data_path: Path = typer.Argument(..., help="Path to the dataset CSV file"),
    feature_cols: str = typer.Option(..., help="Comma-separated feature column names"),
    label_col: str = typer.Option("Label", help="Label column name"),
    episodes: int = typer.Option(100, help="Number of episodes to run"),
):
    """Train and test the IDS environment"""

    logger.info("Starting IDS environment training")

    # Parse feature columns
    features = [col.strip() for col in feature_cols.split(",")]

    # Create environment
    env = IDSDetectionEnv(data_path, features, label_col)

    # Simple training loop with progress bar
    for episode in tqdm(range(episodes), desc="Training Episodes"):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Random action for demonstration
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if done or truncated:
                break

        if episode % 10 == 0:
            logger.info(f"Episode {episode}, Total Reward: {total_reward}")

    logger.success("Training completed!")


@app.command()
def validate_data(
    data_path: Path = typer.Argument(..., help="Path to the dataset CSV file"),
    feature_cols: str = typer.Option(..., help="Comma-separated feature column names"),
    label_col: str = typer.Option("Label", help="Label column name"),
):
    """Validate the dataset for IDS environment"""

    logger.info(f"Validating dataset: {data_path}")

    try:
        df = pd.read_csv(data_path)
        features = [col.strip() for col in feature_cols.split(",")]

        # Check if columns exist
        missing_cols = [col for col in features + [label_col] if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return

        # Basic statistics
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Number of classes: {df[label_col].nunique()}")
        logger.info(f"Class distribution:\n{df[label_col].value_counts()}")

        # Check for missing values
        missing_values = df[features + [label_col]].isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values found:\n{missing_values[missing_values > 0]}")
        else:
            logger.success("No missing values found")

    except Exception as e:
        logger.error(f"Error validating data: {e}")


if __name__ == "__main__":
    app()
