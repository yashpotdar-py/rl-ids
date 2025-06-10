"""DQN Agent implementation for reinforcement learning based intrusion detection."""

from collections import deque
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import numpy as np
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import torch.optim as optim
import typer


class DQNConfig(BaseModel):
    """Configuration for DQN Agent."""

    state_dim: int = Field(..., description="State space dimension")
    action_dim: int = Field(..., description="Action space dimension")
    lr: float = Field(1e-4, description="Learning rate")
    gamma: float = Field(0.99, description="Discount factor")
    epsilon: float = Field(1.0, description="Initial exploration rate")
    eps_decay: float = Field(0.995, description="Epsilon decay rate")
    eps_min: float = Field(0.1, description="Minimum epsilon value")
    memory_size: int = Field(3000000, description="Replay buffer size")
    batch_size: int = Field(64, description="Training batch size")
    hidden_dims: List[int] = Field(
        [256, 128], description="Hidden layer dimensions")


class DQN(nn.Module):
    """Deep Q-Network model."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 128]):
        super(DQN, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.fc = nn.Sequential(*layers)

        logger.debug(
            f"Created DQN with architecture: {input_dim} -> {hidden_dims} -> {output_dim}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.fc(x)


class DQNAgent:
    """Deep Q-Network Agent for reinforcement learning."""

    def __init__(self, config: DQNConfig):
        """Initialize DQN Agent with configuration."""
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.eps_decay = config.eps_decay
        self.eps_min = config.eps_min
        self.batch_size = config.batch_size

        # Add device selection
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize networks
        self.model = DQN(config.state_dim, config.action_dim,
                         config.hidden_dims).to(self.device)
        self.target_model = DQN(config.state_dim, config.action_dim, config.hidden_dims).to(
            self.device
        )
        self.update_target()

        # Initialize training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.memory: deque = deque(maxlen=config.memory_size)

        # Training metrics
        self.training_step = 0
        self.episode_count = 0

        logger.info(
            f"Initialized DQN Agent with state_dim={config.state_dim}, action_dim={config.action_dim}"
        )

    def update_target(self) -> None:
        """Update target network with current network weights."""
        self.target_model.load_state_dict(self.model.state_dict())
        logger.debug("Updated target network")

    def remember(
        self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool
    ) -> None:
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
            logger.debug(f"Random action selected: {action}")
            return action

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)

        action = q_values.argmax().item()
        logger.debug(f"Greedy action selected: {action}")
        return action

    def replay(self) -> Optional[float]:
        """Train the model on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Compute Q-values
        curr_Q = self.model(states).gather(1, actions).squeeze()
        next_Q = self.target_model(next_states).max(1)[0]
        target_Q = rewards + self.gamma * next_Q * (~dones)

        # Compute loss and update
        loss = self.criterion(curr_Q, target_Q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay

        self.training_step += 1

        if self.training_step % 1000 == 0:
            logger.info(
                f"Training step {self.training_step}, loss: {loss.item():.4f}, epsilon: {self.epsilon:.4f}"
            )

        return loss.item()

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Save model state dict to file."""
        filepath = Path(filepath)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
                "config": self.config.dict(),
            },
            filepath,
        )
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path, map_location: Optional[torch.device] = None) -> None:
        """Load a saved model."""
        try:
            logger.info(f"Loading model from {filepath}")

            # Use map_location if provided, otherwise use the agent's device
            device = map_location if map_location is not None else (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )

            checkpoint = torch.load(filepath, map_location=device)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.target_model.load_state_dict(
                checkpoint["target_model_state_dict"])

            # Move models to the correct device
            self.model.to(device)
            self.target_model.to(device)

            logger.success(f"✅ Model loaded successfully from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_metrics(self) -> Dict[str, Any]:
        """Get current training metrics."""
        return {
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "memory_size": len(self.memory),
            "episode_count": self.episode_count,
        }


def main(
    state_dim: int = typer.Option(10, help="State space dimension"),
    action_dim: int = typer.Option(2, help="Action space dimension"),
    lr: float = typer.Option(1e-4, help="Learning rate"),
    gamma: float = typer.Option(0.99, help="Discount factor"),
    epsilon: float = typer.Option(1.0, help="Initial exploration rate"),
    eps_decay: float = typer.Option(0.995, help="Epsilon decay rate"),
    eps_min: float = typer.Option(0.1, help="Minimum epsilon value"),
    memory_size: int = typer.Option(3000000, help="Replay buffer size"),
    batch_size: int = typer.Option(64, help="Training batch size"),
) -> None:
    """Initialize and test DQN Agent."""
    config = DQNConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        eps_decay=eps_decay,
        eps_min=eps_min,
        memory_size=memory_size,
        batch_size=batch_size,
    )

    agent = DQNAgent(config)
    logger.info("DQN Agent initialized successfully")

    # Example usage
    dummy_state = np.random.random(state_dim)
    action = agent.act(dummy_state)
    logger.info(f"Sample action for random state: {action}")


if __name__ == "__main__":
    typer.run(main)
