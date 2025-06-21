# DQN Agent Module

## Overview

The `rl_ids.agents.dqn_agent` module implements a state-of-the-art Deep Q-Network (DQN) agent for reinforcement learning-based intrusion detection. It features advanced techniques including Double DQN, Dueling DQN, prioritized experience replay, and sophisticated neural network architectures optimized for network traffic pattern recognition.

The agent is designed to learn adaptive intrusion detection policies from network traffic data, continuously improving its detection capabilities through reinforcement learning principles.

---

## Classes

### `DQNConfig`

Configuration class for DQN Agent parameters using Pydantic for validation.

**Signature:**
```python
class DQNConfig(BaseModel):
    """Configuration for DQN Agent."""
```

**Description:**
A comprehensive configuration class that defines all hyperparameters and settings for the DQN agent. Uses Pydantic for automatic validation and type checking.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dim` | `int` | Required | Dimension of the state space (number of input features) |
| `action_dim` | `int` | Required | Number of possible actions (attack classes) |
| `lr` | `float` | `1e-4` | Learning rate for the optimizer |
| `gamma` | `float` | `0.99` | Discount factor for future rewards |
| `epsilon` | `float` | `1.0` | Initial exploration rate for ε-greedy policy |
| `eps_decay` | `float` | `0.995` | Exponential decay rate for epsilon |
| `eps_min` | `float` | `0.01` | Minimum epsilon value |
| `memory_size` | `int` | `100000` | Size of the experience replay buffer |
| `batch_size` | `int` | `32` | Training batch size |
| `hidden_dims` | `List[int]` | `[512, 256]` | Hidden layer dimensions |
| `device` | `str` | `"auto"` | Computing device ("cpu", "cuda", or "auto") |
| `dropout_rate` | `float` | `0.2` | Dropout rate for regularization |
| `use_layer_norm` | `bool` | `True` | Whether to use layer normalization |
| `weight_decay` | `float` | `1e-5` | L2 regularization strength |
| `double_dqn` | `bool` | `True` | Enable Double DQN algorithm |
| `dueling` | `bool` | `True` | Enable Dueling DQN architecture |
| `prioritized_replay` | `bool` | `False` | Enable prioritized experience replay |

**Example:**
```python
from rl_ids.agents.dqn_agent import DQNConfig

config = DQNConfig(
    state_dim=77,
    action_dim=15,
    lr=1e-4,
    hidden_dims=[1024, 512, 256, 128],
    double_dqn=True,
    dueling=True,
    prioritized_replay=True
)
```

---

### `DQN`

Deep Q-Network neural network model.

**Inherits**: `torch.nn.Module`

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `input_dim` | `int` | Input feature dimension |
| `output_dim` | `int` | Output action dimension |
| `hidden_dims` | `List[int]` | Hidden layer dimensions |

**Methods**

#### `forward(x: torch.Tensor) -> torch.Tensor`

Forward pass through the network.

**Parameters**
- `x`: Input tensor of shape `(batch_size, input_dim)`

**Returns**
- `torch.Tensor`: Q-values for each action, shape `(batch_size, output_dim)`

**Examples**

```python
import torch
from rl_ids.agents.dqn_agent import DQN

# Create network
model = DQN(
    input_dim=78,
    output_dim=15,
    hidden_dims=[512, 256, 128]
)

# Forward pass
state = torch.randn(32, 78)  # batch_size=32
q_values = model(state)      # shape: (32, 15)
```

---

### `DQNAgent`

Main DQN agent class implementing the Deep Q-Network algorithm.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `config` | `DQNConfig` | Agent configuration |

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `model` | `DQN` | Main Q-network |
| `target_model` | `DQN` | Target Q-network |
| `optimizer` | `torch.optim.Adam` | Model optimizer |
| `memory` | `deque` | Experience replay buffer |
| `epsilon` | `float` | Current exploration rate |
| `device` | `torch.device` | Computing device (CPU/CUDA) |

**Methods**

#### `act(state: np.ndarray, training: bool = True) -> int`

Choose action using epsilon-greedy policy.

**Parameters**
- `state`: Current environment state
- `training`: Whether in training mode (enables exploration)

**Returns**
- `int`: Selected action

**Examples**

```python
import numpy as np
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig

# Initialize agent
config = DQNConfig(state_dim=78, action_dim=15)
agent = DQNAgent(config)

# Select action
state = np.random.randn(78)
action = agent.act(state, training=True)  # Exploration enabled
action = agent.act(state, training=False) # Pure greedy
```

#### `remember(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None`

Store experience in replay buffer.

**Parameters**
- `state`: Current state
- `action`: Action taken
- `reward`: Reward received
- `next_state`: Next state
- `done`: Whether episode ended

**Examples**

```python
# Store experience
agent.remember(
    state=current_state,
    action=action,
    reward=1.0,
    next_state=next_state,
    done=False
)
```

#### `replay() -> Optional[float]`

Train the model on a batch of experiences.

**Returns**
- `Optional[float]`: Training loss if training occurred, None otherwise

**Examples**

```python
# Training step
loss = agent.replay()
if loss is not None:
    print(f"Training loss: {loss:.4f}")
```

#### `update_target() -> None`

Update target network with current network weights.

**Examples**

```python
# Update target network (typically done every N episodes)
agent.update_target()
```

#### `save_model(filepath: Union[str, Path]) -> None`

Save model state to file.

**Parameters**
- `filepath`: Path to save the model

**Examples**

```python
from pathlib import Path

# Save model
agent.save_model("models/dqn_model.pt")
agent.save_model(Path("models/episode_100.pt"))
```

#### `load_model(filepath: Path, map_location: Optional[torch.device] = None) -> None`

Load a saved model.

**Parameters**
- `filepath`: Path to model file
- `map_location`: Device to load model on

**Raises**
- `FileNotFoundError`: If model file doesn't exist
- `RuntimeError`: If model loading fails

**Examples**

```python
# Load model
agent.load_model(Path("models/dqn_model_best.pt"))

# Load on specific device
agent.load_model(
    Path("models/dqn_model.pt"),
    map_location=torch.device('cpu')
)
```

## Usage Examples

### Basic Training Loop

```python
import numpy as np
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.environments.ids_env import IDSDetectionEnv

# Initialize environment and agent
env = IDSDetectionEnv(data_path="data/train.csv", feature_cols=features)
config = DQNConfig(
    state_dim=len(features),
    action_dim=env.action_space.n
)
agent = DQNAgent(config)

# Training loop
for episode in range(1000):
    state, _ = env.reset()
    total_reward = 0
    
    while True:
        # Select action
        action = agent.act(state, training=True)
        
        # Environment step
        next_state, reward, done, truncated, _ = env.step(action)
        
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train agent
        loss = agent.replay()
        
        state = next_state
        total_reward += reward
        
        if done or truncated:
            break
    
    # Update target network periodically
    if episode % 10 == 0:
        agent.update_target()
    
    print(f"Episode {episode}: Reward = {total_reward}")
```

### Model Evaluation

```python
# Load trained model
agent.load_model("models/dqn_model_best.pt")
agent.epsilon = 0.0  # Disable exploration

# Evaluation
state, _ = env.reset()
total_reward = 0

while True:
    action = agent.act(state, training=False)
    state, reward, done, truncated, _ = env.step(action)
    total_reward += reward
    
    if done or truncated:
        break

print(f"Evaluation reward: {total_reward}")
```

## See Also

- [Environment Module](environments.md) - Custom Gym environment
- [Training Module](modeling.md) - Training and evaluation pipelines
- [Configuration](config.md) - System configuration
