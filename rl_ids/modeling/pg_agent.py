"""
Policy Gradient (PG) agent for reinforcement learning-based intrusion detection.

This module implements the REINFORCE algorithm with baseline for policy-based
learning in the intrusion detection environment.
"""
from typing import List, Tuple, Optional, Dict, Any, Union
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from loguru import logger

from rl_ids.modeling.q_network import QNetwork


class PolicyNetwork(nn.Module):
    """
    Neural network for policy representation in policy gradient algorithms.

    Features:
    - Configurable hidden layers with dropout
    - Layer normalization for training stability
    - Weight initialization for faster convergence
    - Softmax output distribution for action probabilities
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, List[int]],
        output_dim: int,
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
    ):
        """
        Initialize the Policy Network.

        Args:
            input_dim: Dimension of the input features
            hidden_dims: Dimensions of hidden layers (single int or list of ints)
            output_dim: Number of possible actions
            dropout_rate: Probability of dropout between hidden layers (0.0 = no dropout)
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        # Standardize hidden_dims to be a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Build network layers
        layers = []
        prev_dim = input_dim

        for idx, dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, dim))

            # Optional layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))

            # Activation
            layers.append(nn.ReLU())

            # Optional dropout
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = dim

        # Output layer with softmax activation for action probabilities
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Softmax(dim=-1))

        # Create sequential model
        self.net = nn.Sequential(*layers)

        # Apply custom weight initialization
        self._initialize_weights()

        logger.debug(f"Created PolicyNetwork with input_dim={input_dim}, "
                     f"hidden_dims={hidden_dims}, output_dim={output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Action probability distribution
        """
        return self.net(x)

    def _initialize_weights(self) -> None:
        """Initialize weights for faster training convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # He initialization for ReLU networks
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ValueNetwork(nn.Module):
    """
    Neural network for state value estimation (baseline).

    Used to reduce variance in policy gradient updates.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[int, List[int]],
        dropout_rate: float = 0.0,
        use_layer_norm: bool = True,
    ):
        """
        Initialize the Value Network.

        Args:
            input_dim: Dimension of the input features
            hidden_dims: Dimensions of hidden layers (single int or list of ints)
            dropout_rate: Probability of dropout between hidden layers
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()

        # Standardize hidden_dims to be a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Build network layers
        layers = []
        prev_dim = input_dim

        for idx, dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, dim))

            # Optional layer normalization
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))

            # Activation
            layers.append(nn.ReLU())

            # Optional dropout
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = dim

        # Output layer (single value)
        layers.append(nn.Linear(prev_dim, 1))

        # Create sequential model
        self.net = nn.Sequential(*layers)

        # Apply custom weight initialization
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Estimated state value
        """
        return self.net(x)

    def _initialize_weights(self) -> None:
        """Initialize weights for faster training convergence."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class PGAgent:
    """
    Policy Gradient agent implementation using REINFORCE with baseline.

    Features:
    - State value baseline for variance reduction
    - Entropy regularization for exploration encouragement
    - Gradient clipping for training stability
    - Experience batching for efficient updates
    - Checkpoint saving/loading
    - Statistics tracking
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Union[int, List[int]] = [128, 128],
        lr_policy: float = 1e-4,
        lr_value: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda",
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Policy Gradient agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: Hidden layer dimensions for both networks
            lr_policy: Learning rate for policy network
            lr_value: Learning rate for value network
            gamma: Discount factor
            entropy_coef: Coefficient for entropy regularization
            value_coef: Coefficient for value loss in the combined loss
            max_grad_norm: Maximum gradient norm for clipping
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
        self.policy_net = PolicyNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=action_dim,
            dropout_rate=0.1  # Small dropout for regularization
        ).to(self.device)

        self.value_net = ValueNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            dropout_rate=0.1
        ).to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr_policy)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr_value)

        # Hyperparameters
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.action_dim = action_dim

        # Episode memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

        # Training stats
        self.update_counter = 0
        self.episode_rewards = []
        self.policy_losses = []
        self.value_losses = []
        self.entropies = []
        self.training_time = 0

        # Checkpoint directory
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        logger.info(
            f"Initialized PG agent with state_dim={state_dim}, action_dim={action_dim}, "
            f"device={device}, gamma={gamma}"
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Union[int, Tuple[int, torch.Tensor]]:
        """
        Select action based on current policy.

        Args:
            state: Current state
            deterministic: If True, select most probable action instead of sampling

        Returns:
            action: Selected action (or tuple of action and log prob during training)
        """
        # Ensure state is a numpy array
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Convert to tensor
        state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)

        # Get action probabilities
        self.policy_net.eval()  # Set to eval mode for inference
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor)

        # Create distribution
        dist = Categorical(action_probs)

        if deterministic:
            # Select most probable action
            action = action_probs.argmax(dim=1)
            self.policy_net.train()  # Back to train mode
            return action.item()
        else:
            # Sample from distribution
            action = dist.sample()
            log_prob = dist.log_prob(action)
            self.policy_net.train()  # Back to train mode
            return action.item(), log_prob

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: torch.Tensor
    ) -> None:
        """
        Store transition in episode memory.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            log_prob: Log probability of taking the action
        """
        # Convert to numpy arrays if needed
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)

        # Store transition
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def update(self) -> Dict[str, float]:
        """
        Update policy and value networks based on collected experience.

        Returns:
            Dictionary with loss statistics
        """
        # Skip update if no experience
        if not self.rewards:
            logger.warning("No experience to update from")
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "total_loss": 0.0
            }

        start_time = time.time()

        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)

        # Calculate discounted returns
        returns = self._compute_returns(self.rewards, self.dones)
        returns = torch.FloatTensor(returns).to(self.device)

        # Get value predictions
        values = self.value_net(states).squeeze()

        # Normalize returns (helps with training stability)
        returns_std = returns.std() + 1e-8
        returns_normalized = (returns - returns.mean()) / returns_std
        values_normalized = (values - values.mean()) / returns_std

        # Compute advantages
        advantages = returns_normalized - values_normalized.detach()

        # Re-compute action probabilities
        action_probs = self.policy_net(states)
        dist = Categorical(action_probs)
        current_log_probs = dist.log_prob(actions)

        # Compute entropy (for exploration)
        entropy = dist.entropy().mean()

        # Policy loss
        policy_loss = -(current_log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Total loss with entropy regularization
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        # Optimize
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

        self.policy_optimizer.step()
        self.value_optimizer.step()

        # Track stats
        policy_loss_val = policy_loss.item()
        value_loss_val = value_loss.item()
        entropy_val = entropy.item()
        total_loss_val = total_loss.item()

        self.policy_losses.append(policy_loss_val)
        self.value_losses.append(value_loss_val)
        self.entropies.append(entropy_val)
        self.update_counter += 1
        self.training_time += time.time() - start_time

        # Clear episode memory
        self._clear_memory()

        return {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss_val,
            "entropy": entropy_val,
            "total_loss": total_loss_val
        }

    def _compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """
        Compute discounted returns for each state.

        Args:
            rewards: List of rewards
            dones: List of done flags

        Returns:
            List of discounted returns
        """
        returns = []
        R = 0

        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        return returns

    def _clear_memory(self) -> None:
        """Clear episode memory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.dones = []

    def save_checkpoint(
        self,
        episode: int,
        rewards: List[float],
        path: Optional[str] = None
    ) -> str:
        """
        Save checkpoint of agent state.

        Args:
            episode: Current episode number
            rewards: List of recent rewards
            path: Optional explicit path (if None, uses checkpoint_dir)

        Returns:
            Path where checkpoint was saved
        """
        if path is None:
            if self.checkpoint_dir is None:
                raise ValueError("checkpoint_dir must be specified if path is None")
            path = os.path.join(self.checkpoint_dir, f"pg_checkpoint_ep{episode}.pt")

        checkpoint = {
            'episode': episode,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
            'entropies': self.entropies,
            'update_counter': self.update_counter,
            'recent_rewards': rewards,
            'hyperparams': {
                'gamma': self.gamma,
                'entropy_coef': self.entropy_coef,
                'value_coef': self.value_coef
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
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])

        # Restore optimizer states
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])

        # Restore counters
        self.update_counter = checkpoint['update_counter']
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        self.entropies = checkpoint.get('entropies', [])

        logger.info(f"Loaded checkpoint from {path} (episode {checkpoint['episode']})")
        return {
            'episode': checkpoint['episode'],
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
            'avg_policy_loss': np.mean(self.policy_losses[-100:]) if self.policy_losses else 0,
            'avg_value_loss': np.mean(self.value_losses[-100:]) if self.value_losses else 0,
            'avg_entropy': np.mean(self.entropies[-100:]) if self.entropies else 0,
            'training_time': self.training_time
        }
