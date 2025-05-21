"""
Test suite for the environment module.
"""
import os
from pathlib import Path
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

from rl_ids.environment import EnvMetrics, IntrusionEnv


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = pd.DataFrame({
        'Feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Feature2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'Feature3': [10, 20, 30, 40, 50],
        'Label': [0, 1, 0, 1, 0]
    })
    return data


@pytest.fixture
def temp_parquet_path(sample_data):
    """Create a temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
        filepath = Path(tf.name)
        sample_data.to_parquet(filepath)

    yield filepath

    # Clean up after test
    if filepath.exists():
        os.unlink(filepath)


class TestEnvMetrics:
    """Tests for the EnvMetrics dataclass."""

    def test_init_default(self):
        """Test initialization with default values."""
        metrics = EnvMetrics()
        assert metrics.true_positives == 0
        assert metrics.false_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_negatives == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.total_reward == 0.0

    def test_init_with_values(self):
        """Test initialization with custom values."""
        metrics = EnvMetrics(
            true_positives=10,
            false_positives=5,
            true_negatives=20,
            false_negatives=2,
            accuracy=0.75,
            precision=0.8,
            recall=0.7,
            f1_score=0.6,
            total_reward=15.0
        )
        assert metrics.true_positives == 10
        assert metrics.false_positives == 5
        assert metrics.true_negatives == 20
        assert metrics.false_negatives == 2
        assert metrics.accuracy == 0.75
        assert metrics.precision == 0.8
        assert metrics.recall == 0.7
        assert metrics.f1_score == 0.6
        assert metrics.total_reward == 15.0

    def test_asdict(self):
        """Test the _asdict method."""
        metrics = EnvMetrics(true_positives=5, false_positives=2)
        d = metrics._asdict()
        assert isinstance(d, dict)
        assert d['true_positives'] == 5
        assert d['false_positives'] == 2
        assert 'accuracy' in d


class TestIntrusionEnv:
    """Tests for the IntrusionEnv class."""

    def test_init(self, temp_parquet_path):
        """Test initialization with defaults."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)

        assert isinstance(env, gym.Env)
        assert env.observation_space.shape == (3,)  # 3 features from sample data
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2
        assert env._reward_correct == 1.0
        assert env._reward_incorrect == -1.0
        assert env._random_sampling is False

    def test_init_custom_parameters(self, temp_parquet_path):
        """Test initialization with custom parameters."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(
                data_path=temp_parquet_path,
                normalize_features=False,
                sample_limit=3,
                random_sampling=True,
                reward_correct=2.0,
                reward_incorrect=-2.0,
                render_mode="human"
            )

        assert env._reward_correct == 2.0
        assert env._reward_incorrect == -2.0
        assert env._random_sampling is True
        assert env.render_mode == "human"
        assert len(env._X) == 3  # Should be limited to 3 samples

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with patch('rl_ids.environment.logger'), \
                pytest.raises(FileNotFoundError):
            IntrusionEnv(data_path="/non/existent/path.parquet")

    def test_init_missing_label_column(self, sample_data):
        """Test initialization with data missing the Label column."""
        data_no_label = sample_data.drop(columns=["Label"])
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            filepath = Path(tf.name)
            data_no_label.to_parquet(filepath)

        try:
            with patch('rl_ids.environment.logger'), \
                    pytest.raises(ValueError, match="Data must contain a 'Label' column"):
                IntrusionEnv(data_path=filepath)
        finally:
            if filepath.exists():
                os.unlink(filepath)

    def test_reset(self, temp_parquet_path):
        """Test reset method."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)  # 3 features
        assert env._episode_step == 0
        assert env._done is False
        assert isinstance(info, dict)
        assert 'step' in info and info['step'] == 0

    def test_reset_with_options(self, temp_parquet_path):
        """Test reset with options."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            obs, info = env.reset(options={"max_steps": 2, "start_idx": 1})

        assert env._episode_length == 2
        assert env._idx == 1

    def test_reset_with_random_sampling(self, temp_parquet_path):
        """Test reset with random sampling."""
        with patch('rl_ids.environment.logger'), \
                patch('numpy.random.randint', return_value=2):
            env = IntrusionEnv(data_path=temp_parquet_path, random_sampling=True)
            obs, info = env.reset()

        assert env._idx == 2  # Should be set to the mocked random value

    def test_step_correct_action(self, temp_parquet_path):
        """Test step method with correct action."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            _, info = env.reset()
            first_label = env._y[env._idx]
            obs, reward, terminated, truncated, info = env.step(first_label)

        assert reward == env._reward_correct
        assert env._episode_step == 1
        assert env._idx == 1
        if first_label == 0:
            assert env._metrics.true_negatives == 1
        else:
            assert env._metrics.true_positives == 1

    def test_step_incorrect_action(self, temp_parquet_path):
        """Test step method with incorrect action."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            _, info = env.reset()
            first_label = env._y[env._idx]
            incorrect_action = 1 - first_label  # Invert the correct label
            obs, reward, terminated, truncated, info = env.step(incorrect_action)

        assert reward == env._reward_incorrect
        if first_label == 0:
            assert env._metrics.false_positives == 1
        else:
            assert env._metrics.false_negatives == 1

    def test_step_invalid_action(self, temp_parquet_path):
        """Test step with invalid action."""
        with patch('rl_ids.environment.logger') as mock_logger:
            env = IntrusionEnv(data_path=temp_parquet_path)
            env.reset()
            obs, reward, terminated, truncated, info = env.step(999)  # Invalid action# filepath: /home/yashpotdar/projects/rl-ids/tests/test_environment.py


"""
Test suite for the environment module.
"""


@pytest.fixture
def sample_data():
    """Create a small sample dataset for testing."""
    data = pd.DataFrame({
        'Feature1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Feature2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'Feature3': [10, 20, 30, 40, 50],
        'Label': [0, 1, 0, 1, 0]
    })
    return data


@pytest.fixture
def temp_parquet_path(sample_data):
    """Create a temporary parquet file with sample data."""
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
        filepath = Path(tf.name)
        sample_data.to_parquet(filepath)

    yield filepath

    # Clean up after test
    if filepath.exists():
        os.unlink(filepath)


class TestEnvMetrics:
    """Tests for the EnvMetrics dataclass."""

    def test_init_default(self):
        """Test initialization with default values."""
        metrics = EnvMetrics()
        assert metrics.true_positives == 0
        assert metrics.false_positives == 0
        assert metrics.true_negatives == 0
        assert metrics.false_negatives == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.total_reward == 0.0

    def test_init_with_values(self):
        """Test initialization with custom values."""
        metrics = EnvMetrics(
            true_positives=10,
            false_positives=5,
            true_negatives=20,
            false_negatives=2,
            accuracy=0.75,
            precision=0.8,
            recall=0.7,
            f1_score=0.6,
            total_reward=15.0
        )
        assert metrics.true_positives == 10
        assert metrics.false_positives == 5
        assert metrics.true_negatives == 20
        assert metrics.false_negatives == 2
        assert metrics.accuracy == 0.75
        assert metrics.precision == 0.8
        assert metrics.recall == 0.7
        assert metrics.f1_score == 0.6
        assert metrics.total_reward == 15.0

    def test_asdict(self):
        """Test the _asdict method."""
        metrics = EnvMetrics(true_positives=5, false_positives=2)
        d = metrics._asdict()
        assert isinstance(d, dict)
        assert d['true_positives'] == 5
        assert d['false_positives'] == 2
        assert 'accuracy' in d


class TestIntrusionEnv:
    """Tests for the IntrusionEnv class."""

    def test_init(self, temp_parquet_path):
        """Test initialization with defaults."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)

        assert isinstance(env, gym.Env)
        assert env.observation_space.shape == (3,)  # 3 features from sample data
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2
        assert env._reward_correct == 1.0
        assert env._reward_incorrect == -1.0
        assert env._random_sampling is False

    def test_init_custom_parameters(self, temp_parquet_path):
        """Test initialization with custom parameters."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(
                data_path=temp_parquet_path,
                normalize_features=False,
                sample_limit=3,
                random_sampling=True,
                reward_correct=2.0,
                reward_incorrect=-2.0,
                render_mode="human"
            )

        assert env._reward_correct == 2.0
        assert env._reward_incorrect == -2.0
        assert env._random_sampling is True
        assert env.render_mode == "human"
        assert len(env._X) == 3  # Should be limited to 3 samples

    def test_init_file_not_found(self):
        """Test initialization with non-existent file."""
        with patch('rl_ids.environment.logger'), \
                pytest.raises(FileNotFoundError):
            IntrusionEnv(data_path="/non/existent/path.parquet")

    def test_init_missing_label_column(self, sample_data):
        """Test initialization with data missing the Label column."""
        data_no_label = sample_data.drop(columns=["Label"])
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tf:
            filepath = Path(tf.name)
            data_no_label.to_parquet(filepath)

        try:
            with patch('rl_ids.environment.logger'), \
                    pytest.raises(ValueError, match="Data must contain a 'Label' column"):
                IntrusionEnv(data_path=filepath)
        finally:
            if filepath.exists():
                os.unlink(filepath)

    def test_reset(self, temp_parquet_path):
        """Test reset method."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)  # 3 features
        assert env._episode_step == 0
        assert env._done is False
        assert isinstance(info, dict)
        assert 'step' in info and info['step'] == 0

    def test_reset_with_options(self, temp_parquet_path):
        """Test reset with options."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            obs, info = env.reset(options={"max_steps": 2, "start_idx": 1})

        assert env._episode_length == 2
        assert env._idx == 1

    def test_reset_with_random_sampling(self, temp_parquet_path):
        """Test reset with random sampling."""
        with patch('rl_ids.environment.logger'), \
                patch('numpy.random.randint', return_value=2):
            env = IntrusionEnv(data_path=temp_parquet_path, random_sampling=True)
            obs, info = env.reset()

        assert env._idx == 2  # Should be set to the mocked random value

    def test_step_correct_action(self, temp_parquet_path):
        """Test step method with correct action."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            _, info = env.reset()
            first_label = env._y[env._idx]
            obs, reward, terminated, truncated, info = env.step(first_label)

        assert reward == env._reward_correct
        assert env._episode_step == 1
        assert env._idx == 1
        if first_label == 0:
            assert env._metrics.true_negatives == 1
        else:
            assert env._metrics.true_positives == 1

    def test_step_incorrect_action(self, temp_parquet_path):
        """Test step method with incorrect action."""
        with patch('rl_ids.environment.logger'):
            env = IntrusionEnv(data_path=temp_parquet_path)
            _, info = env.reset()
            first_label = env._y[env._idx]
            incorrect_action = 1 - first_label  # Invert the correct label
            obs, reward, terminated, truncated, info = env.step(incorrect_action)

        assert reward == env._reward_incorrect
        if first_label == 0:
            assert env._metrics.false_positives == 1
        else:
            assert env._metrics.false_negatives == 1

    def test_step_invalid_action(self, temp_parquet_path):
        """Test step with invalid action."""
        with patch('rl_ids.environment.logger') as mock_logger:
            env = IntrusionEnv(data_path=temp_parquet_path)
            env.reset()
            obs, reward, terminated, truncated, info = env.step(999)  # Invalid action
