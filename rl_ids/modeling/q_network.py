"""
Neural network architecture for Q-value approximation in reinforcement learning.

This module defines network architectures used by DQN agents for approximating
state-action values in the intrusion detection environment.
"""
from typing import List, Union

import torch
import torch.nn as nn
from loguru import logger


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values with configurable architecture.

    Features:
    - Configurable hidden layers with dropout
    - Layer normalization for training stability
    - Weight initialization for faster convergence
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
        Initialize the Q-Network.

        Args:
            input_dim: Dimension of the input features
            hidden_dims: Dimensions of hidden layers (single int or list of ints)
            output_dim: Number of output actions
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

            # Optional layer normalization (improves training stability)
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))

            # Activation
            layers.append(nn.ReLU())

            # Optional dropout for regularization
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        # Create sequential model
        self.net = nn.Sequential(*layers)

        # Apply custom weight initialization
        self._initialize_weights()

        logger.debug(f"Created QNetwork with input_dim={input_dim}, "
                     f"hidden_dims={hidden_dims}, output_dim={output_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Q-values for each action
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

    def save(self, path: str) -> None:
        """
        Save model parameters.

        Args:
            path: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'model_config': {
                'input_dim': self.net[0].in_features,
                'output_dim': self.net[-1].out_features,
                'architecture': [
                    module.out_features for module in self.net
                    if isinstance(module, nn.Linear)][:-1]
            }
        }, path)
        logger.info(f"Saved QNetwork to {path}")

    @classmethod
    def load(cls, path: str, device: torch.device) -> 'QNetwork':
        """
        Load model from a saved checkpoint.

        Args:
            path: Path to the saved model
            device: Device to load the model to

        Returns:
            Loaded model
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['model_config']
        model = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['architecture'],
            output_dim=config['output_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        logger.info(f"Loaded QNetwork from {path}")
        return model
