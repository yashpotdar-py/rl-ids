"""
Gym environment for network intrusion detection using flow-based features.

This module provides a reinforcement learning environment that simulates
a network intrusion detection system making binary decisions on network flows.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import spaces
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class EnvMetrics:
    """Container for environment performance metrics."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    total_reward: float = 0.0

    def _asdict(self):
        """Make it compatible with existing code that expects a NamedTuple."""
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


class IntrusionEnv(gym.Env):
    """
    Gymnasium environment for flow-based network intrusion detection.

    This environment simulates a network intrusion detection system where an agent
    must decide whether to allow (0) or block (1) network flows. The agent receives
    a positive reward for correct decisions and a negative reward for incorrect ones.

    Attributes:
        observation_space: The Box space containing network flow features
        action_space: Discrete space with 2 actions (0=allow, 1=block)
        metrics: Performance metrics for the current episode
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        data_path: Union[str, Path],
        normalize_features: bool = True,
        sample_limit: Optional[int] = None,
        random_sampling: bool = False,
        reward_correct: float = 1.0,
        reward_incorrect: float = -1.0,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the intrusion detection environment.

        Args:
            data_path: Path to the parquet file containing network flow data
            normalize_features: Whether to normalize features to zero mean and unit variance
            sample_limit: Optional limit on number of samples to use (for testing/debug)
            random_sampling: Whether to randomly sample from dataset (vs. sequential)
            reward_correct: Reward value for correct predictions
            reward_incorrect: Reward value for incorrect predictions
            render_mode: The render mode to use (None, "human", or "ansi")

        Raises:
            FileNotFoundError: If the data_path doesn't exist
            ValueError: If the data doesn't have the expected format
        """
        # Validate inputs
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self.render_mode = render_mode
        self._reward_correct = reward_correct
        self._reward_incorrect = reward_incorrect
        self._random_sampling = random_sampling

        # Load and preprocess data
        logger.info(f"Loading network flow data from {data_path}")
        try:
            data = pd.read_parquet(data_path)

            if "Label" not in data.columns:
                raise ValueError("Data must contain a 'Label' column")

            # Extract features and labels
            self._X = data.drop(columns=["Label"]).values.astype(np.float32)
            self._y = data["Label"].values.astype(int)
            self._feature_names = data.drop(columns=["Label"]).columns.tolist()

            # Optional feature normalization
            if normalize_features:
                logger.info("Normalizing features")
                scaler = StandardScaler()
                self._X = scaler.fit_transform(self._X).astype(np.float32)
                self._scaler = scaler

            # Optional data sampling
            if sample_limit and sample_limit < len(self._X):
                logger.info(f"Limiting dataset to {sample_limit} samples")
                if random_sampling:
                    indices = np.random.choice(len(self._X), sample_limit, replace=False)
                    self._X = self._X[indices]
                    self._y = self._y[indices]
                else:
                    self._X = self._X[:sample_limit]
                    self._y = self._y[:sample_limit]

            logger.info(f"Loaded {len(self._X)} samples with {self._X.shape[1]} features")

            # Calculate class distribution for logging
            unique, counts = np.unique(self._y, return_counts=True)
            class_dist = dict(zip(unique, counts))
            logger.info(f"Class distribution: {class_dist}")

        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise

        # Initialize spaces
        self.observation_space = spaces.Box(
            low=-np.inf,  # Use -inf/inf for normalized data
            high=np.inf,
            shape=self._X.shape[1:],
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(2)  # 0 = allow, 1 = block

        # Internal state
        self._idx = 0
        self._episode_length = len(self._X)
        self._episode_step = 0
        self._metrics = self._init_metrics()
        self._done = False
        self._current_trajectory: List[Tuple[np.ndarray, int, float, bool]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Optional seed for reproducibility
            options: Optional configuration dictionary with keys:
                - "max_steps": Override the default episode length
                - "start_idx": Start from a specific index in the dataset

        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)

        # Process options
        if options is not None:
            if "max_steps" in options:
                self._episode_length = min(options["max_steps"], len(self._X))
                logger.debug(f"Setting episode length to {self._episode_length}")

            if "start_idx" in options and not self._random_sampling:
                self._idx = max(0, min(options["start_idx"], len(self._X) - 1))
                logger.debug(f"Starting episode from index {self._idx}")

        # Reset internal state if random sampling
        if self._random_sampling:
            # Ensure we don't exceed array bounds
            max_start_idx = max(0, len(self._X) - self._episode_length)
            self._idx = np.random.randint(0, max_start_idx + 1)
        else:
            # For sequential access, wrap around if at the end
            if self._idx + self._episode_length > len(self._X):
                logger.debug("End of dataset reached, wrapping around to beginning")
                self._idx = 0

        self._episode_step = 0
        self._done = False
        self._metrics = self._init_metrics()
        self._current_trajectory = []

        # Get initial observation
        obs = self._get_obs()
        info = self._get_info()

        logger.debug(f"Environment reset, starting from index {self._idx}")
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute an action in the environment.

        Args:
            action: Integer action to take (0 = allow, 1 = block)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)

        Raises:
            RuntimeError: If step is called on a done environment
        """
        if self._done:
            logger.warning("Step called on a done environment, call reset first")
            return np.zeros_like(self._X[0]), 0.0, True, False, {}

        # Safety check to prevent index out of bounds
        if self._idx >= len(self._X):
            logger.warning(f"Index {self._idx} out of bounds for dataset of size {len(self._X)}")
            self._done = True
            return np.zeros_like(self._X[0]), 0.0, True, False, self._get_info()

        # Validate action
        if not self.action_space.contains(action):
            logger.warning(f"Invalid action {action}, using 0 instead")
            action = 0

        # Ground truth and current state
        label = self._y[self._idx]
        current_obs = self._get_obs()
        print(current_obs)

        # Calculate reward and update metrics
        if action == label:
            reward = self._reward_correct
            if action == 1:  # True positive
                # Create new EnvMetrics with updated true_positives
                self._metrics = EnvMetrics(
                    true_positives=self._metrics.true_positives + 1,
                    false_positives=self._metrics.false_positives,
                    true_negatives=self._metrics.true_negatives,
                    false_negatives=self._metrics.false_negatives,
                    accuracy=self._metrics.accuracy,
                    precision=self._metrics.precision,
                    recall=self._metrics.recall,
                    f1_score=self._metrics.f1_score,
                    total_reward=self._metrics.total_reward + reward
                )
            else:  # True negative
                # Create new EnvMetrics with updated true_negatives
                self._metrics = EnvMetrics(
                    true_positives=self._metrics.true_positives,
                    false_positives=self._metrics.false_positives,
                    true_negatives=self._metrics.true_negatives + 1,
                    false_negatives=self._metrics.false_negatives,
                    accuracy=self._metrics.accuracy,
                    precision=self._metrics.precision,
                    recall=self._metrics.recall,
                    f1_score=self._metrics.f1_score,
                    total_reward=self._metrics.total_reward + reward
                )
        else:
            reward = self._reward_incorrect
            if action == 1:  # False positive
                # Create new EnvMetrics with updated false_positives
                self._metrics = EnvMetrics(
                    true_positives=self._metrics.true_positives,
                    false_positives=self._metrics.false_positives + 1,
                    true_negatives=self._metrics.true_negatives,
                    false_negatives=self._metrics.false_negatives,
                    accuracy=self._metrics.accuracy,
                    precision=self._metrics.precision,
                    recall=self._metrics.recall,
                    f1_score=self._metrics.f1_score,
                    total_reward=self._metrics.total_reward + reward
                )
            else:  # False negative
                # Create new EnvMetrics with updated false_negatives
                self._metrics = EnvMetrics(
                    true_positives=self._metrics.true_positives,
                    false_positives=self._metrics.false_positives,
                    true_negatives=self._metrics.true_negatives,
                    false_negatives=self._metrics.false_negatives + 1,
                    accuracy=self._metrics.accuracy,
                    precision=self._metrics.precision,
                    recall=self._metrics.recall,
                    f1_score=self._metrics.f1_score,
                    total_reward=self._metrics.total_reward + reward
                )

        # Update internal state
        self._idx += 1
        self._episode_step += 1
        self._done = (self._idx >= len(self._X)) or (self._episode_step >= self._episode_length)

        # Update accuracy metrics when done
        if self._done:
            self._update_metrics()

        return self._get_obs(), reward, self._done, False, self._get_info()

    def render(self):
        """
        Render the environment (text-based).

        For "human" mode, prints to console.
        For "ansi" mode, returns a string representation.
        """
        if not hasattr(self, "_current_trajectory") or not self._current_trajectory:
            msg = "No steps taken yet"
            if self.render_mode == "ansi":
                return msg
            print(msg)
            return

        # Get the last step
        step_info = self._current_trajectory[-1]
        obs, action, reward, label = step_info
        action_name = "BLOCK" if action == 1 else "ALLOW"
        true_label = "ATTACK" if label == 1 else "BENIGN"
        correct = "✓" if action == label else "✗"

        msg = (
            f"Step {self._episode_step}: Action={action_name}, True={true_label}, "
            f"Reward={reward:.1f} {correct}"
        )

        if self.render_mode == "ansi":
            return msg
        print(msg)

    def close(self):
        """Release resources."""
        logger.debug("Closing environment")
        # Clear buffers to free memory
        if hasattr(self, "_current_trajectory"):
            self._current_trajectory = []

    def get_episode_metrics(self) -> EnvMetrics:
        """
        Get performance metrics for the current episode.

        Returns:
            EnvMetrics named tuple with various performance metrics
        """
        return EnvMetrics(**self._metrics.__dict__)

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        if self._done or self._idx >= len(self._X):
            return np.zeros_like(self._X[0])
        return self._X[self._idx].copy()  # Return copy to prevent modification

    def _get_info(self) -> Dict[str, Any]:
        """Get the current info dictionary."""
        return {
            "step": self._episode_step,
            "total_steps": self._episode_length,
            "metrics": self._metrics._asdict(),  # Use _asdict() method from NamedTuple
        }

    def _init_metrics(self) -> EnvMetrics:
        """Initialize metrics for a new episode."""
        return EnvMetrics(
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            total_reward=0.0
        )

    def _update_metrics(self) -> None:
        """Update derived metrics at the end of an episode."""
        tp = self._metrics.true_positives
        fp = self._metrics.false_positives
        tn = self._metrics.true_negatives
        fn = self._metrics.false_negatives

        # Calculate accuracy, precision, recall, F1
        total = tp + fp + tn + fn
        accuracy = (tp + tn) / max(1, total)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-8, precision + recall)

        # Update the metrics
        self._metrics = EnvMetrics(
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_reward=self._metrics.total_reward
        )
