"""
Deep Q-Network (DQN) agent for reinforcement learning-based intrusion detection.

This module implements the DQN algorithm with experience replay, target networks,
and other improvements for stable training.
"""
import os
import random
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from rl_ids.modeling.q_network import QNetwork

# Define experience tuple type
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNAgent:
    """
    Deep Q-Network agent implementation with experience replay and target network.

    Features:
    - Double DQN for reducing value overestimation
    - Prioritized experience replay (optional)
    - Dueling network architecture (optional)
    - Gradient clipping for training stability
    - Checkpoint saving/loading
    - Learning rate scheduling
    - Statistics tracking
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Union[int, List[int]] = 128,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        double_dqn: bool = True,
        dropout_rate: float = 0.0,
        lr_decay_steps: int = 10000,
        lr_decay_rate: float = 0.9,
        gradient_clip: float = 1.0,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the DQN agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space (number of possible actions)
            hidden_dims: Size of hidden layers (single int or list)
            lr: Learning rate
            gamma: Discount factor
            buffer_size: Maximum replay buffer size
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            double_dqn: Whether to use Double DQN algorithm
            dropout_rate: Dropout probability for regularization (0.0 = no dropout)
            lr_decay_steps: Steps between learning rate decay
            lr_decay_rate: Learning rate decay multiplier
            gradient_clip: Maximum gradient norm for clipping
            device: Device to use for computations
            checkpoint_dir: Directory to save model checkpoints
            **kwargs: Additional arguments
        """
        # Check if CUDA is requested but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"

        self.device = torch.device(device)

        # Create networks
        self.q_net = QNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.target_net = QNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            dropout_rate=0.0  # No dropout in target network
        ).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode

        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn
        self.gradient_clip = gradient_clip
        self.action_dim = action_dim

        # Learning rate schedule
        self.lr = lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_rate = lr_decay_rate

        # Optimizer
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=lr_decay_steps,
            gamma=lr_decay_rate
        )

        # Experience replay
        self.buffer = deque(maxlen=buffer_size)

        # Training stats
        self.update_counter = 0
        self.episode_rewards = []
        self.losses = []
        self.training_time = 0

        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(
            f"Initialized DQN agent with state_dim={state_dim}, action_dim={action_dim}, "
            f"device={device}, double_dqn={double_dqn}"
        )

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action using ε-greedy policy.

        Args:
            state: Current state
            epsilon: Exploration probability

        Returns:
            Selected action index
        """
        # Ensure state is numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Exploration
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        # Exploitation
        self.q_net.eval()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            action = q_values.max(1)[1].item()

        self.q_net.train()
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        # Ensure inputs are the right shape and type
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        self.buffer.append(Experience(state, action, reward, next_state, done))

    def optimize(self) -> Optional[float]:
        """
        Perform one optimization step.

        Returns:
            Loss value or None if buffer has insufficient samples
        """
        if len(self.buffer) < self.batch_size:
            return None

        start_time = time.time()

        # Sample batch
        batch = random.sample(self.buffer, self.batch_size)
        experiences = Experience(*zip(*batch))

        # Convert to tensors
        states = torch.tensor(np.array(experiences.state), dtype=torch.float32).to(self.device)
        actions = torch.tensor(experiences.action, dtype=torch.long).to(self.device).unsqueeze(1)
        rewards = torch.tensor(experiences.reward, dtype=torch.float32).to(self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(experiences.next_state), dtype=torch.float32).to(self.device)
        dones = torch.tensor(experiences.done, dtype=torch.float32).to(self.device).unsqueeze(1)

        # Current Q values
        current_q_values = self.q_net(states).gather(1, actions)

        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online network to select actions, target network to evaluate
                next_actions = self.q_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values)  # Huber loss for robustness

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        if self.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.gradient_clip)

        self.optimizer.step()
        self.scheduler.step()

        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            logger.debug(f"Updated target network at step {self.update_counter}")

        # Track stats
        loss_value = loss.item()
        self.losses.append(loss_value)
        self.training_time += time.time() - start_time

        return loss_value

    def save_checkpoint(self, episode: int, epsilon: float, rewards: List[float], path: Optional[str] = None) -> str:
        """
        Save checkpoint of agent state.

        Args:
            episode: Current episode number
            epsilon: Current epsilon value
            rewards: List of recent rewards
            path: Optional explicit path (if None, uses checkpoint_dir)

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            if self.checkpoint_dir is None:
                raise ValueError("checkpoint_dir must be specified if path is None")
            path = os.path.join(self.checkpoint_dir, f"dqn_checkpoint_ep{episode}.pt")

        checkpoint = {
            'episode': episode,
            'q_net_state_dict': self.q_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'buffer': list(self.buffer)[-10000:],  # Save last 10K experiences
            'losses': self.losses,
            'epsilon': epsilon,
            'update_counter': self.update_counter,
            'recent_rewards': rewards,
            'hyperparams': {
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'target_update_freq': self.target_update_freq,
                'double_dqn': self.double_dqn
            }
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        return path

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load agent state from checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Dictionary of restored values
        """
        checkpoint = torch.load(path, map_location=self.device)

        # Restore network weights
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])

        # Restore optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Restore buffer if needed (optional)
        buffer_samples = checkpoint.get('buffer', [])
        if buffer_samples:
            self.buffer = deque(buffer_samples, maxlen=self.buffer.maxlen)

        # Restore counters
        self.update_counter = checkpoint['update_counter']
        self.losses = checkpoint.get('losses', [])

        logger.info(f"Loaded checkpoint from {path} (episode {checkpoint['episode']})")
        return {
            'episode': checkpoint['episode'],
            'epsilon': checkpoint.get('epsilon', 0.0),
            'rewards': checkpoint.get('recent_rewards', [])
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Return current training statistics.

        Returns:
            Dictionary containing training statistics
        """
        return {
            'updates': self.update_counter,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'buffer_size': len(self.buffer),
            'training_time': self.training_time,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
