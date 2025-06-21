# Environment Module

## Overview

The `rl_ids.environments.ids_env` module provides a custom Gymnasium environment specifically designed for intrusion detection system training using reinforcement learning. It creates a standardized interface between network traffic data and RL agents, enabling seamless training and evaluation of DQN models on cybersecurity datasets.

The environment transforms the intrusion detection problem into a sequential decision-making task where agents learn to classify network traffic samples as either benign or one of 14 different attack types.

---

## Classes

### `IDSDetectionEnv`

A custom Gymnasium environment for RL-based intrusion detection training.

**Signature:**
```python
class IDSDetectionEnv(gym.Env):
    """Custom Environment for RL-Based IDS Detection"""
    
    def __init__(self, data_path: Path, feature_cols: List, label_col: str = "Label"):
        """Initialize the IDS Detection Environment."""
```

**Description:**
A Gym-compatible environment that loads network traffic data and provides a sequential interface for training reinforcement learning agents. Each step corresponds to classifying one network traffic sample, with rewards based on classification accuracy.

**Parameters:**
- `data_path` (`Path`): Path to the CSV file containing network traffic data
- `feature_cols` (`List`): List of column names to use as features
- `label_col` (`str`): Name of the column containing true labels (default: "Label")

**Attributes:**
- `df` (`pd.DataFrame`): Loaded network traffic dataset
- `x` (`np.ndarray`): Feature matrix (samples × features)
- `y` (`np.ndarray`): Label vector (true classifications)
- `current_step` (`int`): Current position in the dataset
- `total_steps` (`int`): Total number of samples in dataset
- `num_classes` (`int`): Number of unique attack classes
- `action_space` (`spaces.Discrete`): Discrete action space for classifications
- `observation_space` (`spaces.Box`): Continuous observation space for features

**Spaces:**
- **Action Space**: `Discrete(num_classes)` - Agent selects classification (0 to num_classes-1)
- **Observation Space**: `Box(shape=(num_features,), dtype=float32)` - Network traffic features

**Example:**
```python
from pathlib import Path
from rl_ids.environments.ids_env import IDSDetectionEnv

# Define feature columns (77 network traffic features)
feature_cols = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    # ... (72 more features)
]

# Initialize environment
env = IDSDetectionEnv(
    data_path=Path("data/processed/train.csv"),
    feature_cols=feature_cols,
    label_col="Label"
)

print(f"Environment initialized:")
print(f"  Samples: {env.total_steps:,}")
print(f"  Features: {env.observation_space.shape[0]}")
print(f"  Classes: {env.num_classes}")
print(f"  Action space: {env.action_space}")
```

**Output:**
```
Environment initialized:
  Samples: 1,048,575
  Features: 77
  Classes: 15
  Action space: Discrete(15)
```

---

### Methods

#### `reset(seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]`

Reset the environment to the initial state and return the first observation.

**Parameters:**
- `seed` (`Optional[int]`): Random seed for reproducibility
- `options` (`Optional[dict]`): Additional reset options (currently unused)

**Returns:**
- `tuple`: A tuple containing:
  - `observation` (`np.ndarray`): Initial state (first network traffic sample)
  - `info` (`dict`): Additional information (empty dict)

**Description:**
Resets the environment to the beginning of the dataset. This method is called at the start of each training episode and sets `current_step` to 0.

**Example:**
```python
# Reset environment for new episode
observation, info = env.reset(seed=42)

print(f"Initial observation shape: {observation.shape}")
print(f"Initial observation: {observation[:5]}...")  # First 5 features
print(f"Info: {info}")
```

**Output:**
```
Initial observation shape: (77,)
Initial observation: [0.    0.    0.    80.   80. ]...
Info: {}
```

#### `step(action: int) -> Tuple[np.ndarray, float, bool, bool, dict]`

Execute an action in the environment and return the result.

**Parameters:**
- `action` (`int`): Predicted class/attack type (0 to num_classes-1)

**Returns:**
- `tuple`: A tuple containing:
  - `observation` (`np.ndarray`): Next state (next network traffic sample)
  - `reward` (`float`): Reward for the action (+1 for correct, -1 for incorrect)
  - `terminated` (`bool`): Whether the episode has ended
  - `truncated` (`bool`): Whether the episode was truncated (always False)
  - `info` (`dict`): Additional information about the step

**Description:**
Processes one step in the environment by comparing the agent's predicted classification with the true label. Advances to the next sample in the dataset.

**Reward Structure:**
- **Correct Classification**: `+1.0`
- **Incorrect Classification**: `-1.0`

**Episode Termination:**
- Episodes end when all samples in the dataset have been processed
- No early termination based on performance

**Example:**
```python
observation, info = env.reset()

# Agent makes a prediction
action = 0  # Predict "BENIGN" class

# Execute action
next_obs, reward, terminated, truncated, info = env.step(action)

print(f"Action taken: {action}")
print(f"Reward received: {reward}")
print(f"Actual label: {info['actual_label']}")
print(f"Predicted action: {info['predicted_action']}")
print(f"Episode terminated: {terminated}")
```

**Output:**
```
Action taken: 0
Reward received: 1.0
Actual label: 0
Predicted action: 0
Episode terminated: False
```

#### `render(mode: str = "human") -> None`

Render the current state of the environment.

**Parameters:**
- `mode` (`str`): Rendering mode (currently only supports "human")

**Description:**
Prints the current step and true label for debugging and monitoring purposes. Useful during training to track agent progress.

**Example:**
```python
# During training loop
env.render()
```

**Output:**
```
Step: 1250, True: 5
```

---

## Usage Patterns

### Basic Training Loop

```python
from rl_ids.environments.ids_env import IDSDetectionEnv
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
import numpy as np

# Initialize environment
env = IDSDetectionEnv(
    data_path="data/processed/train.csv",
    feature_cols=feature_columns
)

# Initialize DQN agent
config = DQNConfig(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)
agent = DQNAgent(config)

# Training loop
for episode in range(num_episodes):
    observation, info = env.reset()
    episode_reward = 0
    step_count = 0
    
    while True:
        # Agent selects action
        action = agent.act(observation, training=True)
        
        # Environment processes action
        next_observation, reward, terminated, truncated, info = env.step(action)
        
        # Store experience for training
        agent.remember(observation, action, reward, next_observation, terminated)
        
        # Train agent periodically
        if len(agent.memory) > agent.config.batch_size:
            agent.replay()
        
        episode_reward += reward
        step_count += 1
        observation = next_observation
        
        if terminated:
            break
    
    # Episode statistics
    accuracy = (episode_reward + step_count) / (2 * step_count)  # Convert to accuracy
    print(f"Episode {episode}: Steps={step_count}, Reward={episode_reward}, Accuracy={accuracy:.3f}")
```

### Evaluation Loop

```python
# Evaluation with trained agent
agent.epsilon = 0.0  # Disable exploration
total_correct = 0
total_samples = 0
predictions = []

observation, info = env.reset()

while True:
    action = agent.act(observation, training=False)
    next_observation, reward, terminated, truncated, info = env.step(action)
    
    # Collect prediction statistics
    correct = reward > 0
    total_correct += correct
    total_samples += 1
    
    predictions.append({
        'predicted': action,
        'actual': info['actual_label'],
        'correct': correct,
        'confidence': agent.get_confidence(observation)
    })
    
    observation = next_observation
    if terminated:
        break

accuracy = total_correct / total_samples
print(f"Evaluation Accuracy: {accuracy:.4f}")
```

## See Also

- [`rl_ids.agents`](agents.md) - DQN agent implementation for training
- [`rl_ids.modeling.train`](modeling.md) - Training pipeline using this environment
- [Training Tutorial](../tutorials/training.md) - Step-by-step training guide
- [Getting Started](../getting-started.md) - Environment setup and first usage
- `done`: Whether episode is finished
- `truncated`: Whether episode was truncated
- `info`: Additional information including actual label

**Examples**

```python
# Environment step
action = 0  # Predict class 0
observation, reward, done, truncated, info = env.step(action)

print(f"Action: {action}")
print(f"Actual label: {info['actual_label']}")
print(f"Reward: {reward}")
print(f"Done: {done}")

# Continue until episode ends
while not (done or truncated):
    action = env.action_space.sample()  # Random action
    observation, reward, done, truncated, info = env.step(action)
```

#### `render(mode: str = "human") -> None`

Render the current environment state.

**Parameters**
- `mode`: Rendering mode (only "human" supported)

**Examples**

```python
# Render current state
env.render()
# Output: Step: 150, True: 2
```

## Usage Examples

### Basic Environment Usage

```python
from pathlib import Path
from rl_ids.environments.ids_env import IDSDetectionEnv
import pandas as pd

# Load dataset to get feature columns
df = pd.read_csv("data/processed/train.csv")
feature_cols = [col for col in df.columns if col != "Label"]

# Initialize environment
env = IDSDetectionEnv(
    data_path=Path("data/processed/train.csv"),
    feature_cols=feature_cols,
    label_col="Label"
)

# Single episode
observation, info = env.reset()
total_reward = 0
step_count = 0

while True:
    # Take random action
    action = env.action_space.sample()
    
    # Environment step
    next_observation, reward, done, truncated, info = env.step(action)
    
    total_reward += reward
    step_count += 1
    
    # Optional: render state
    if step_count % 1000 == 0:
        env.render()
    
    observation = next_observation
    
    if done or truncated:
        break

print(f"Episode finished after {step_count} steps")
print(f"Total reward: {total_reward}")
print(f"Accuracy: {(total_reward + step_count) / (2 * step_count):.2%}")
```

### Agent Training Integration

```python
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.environments.ids_env import IDSDetectionEnv

# Initialize environment
env = IDSDetectionEnv(
    data_path=Path("data/processed/train.csv"),
    feature_cols=feature_cols
)

# Initialize agent
config = DQNConfig(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)
agent = DQNAgent(config)

# Training loop
num_episodes = 1000

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    while True:
        # Agent selects action
        action = agent.act(state, training=True)
        
        # Environment step
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store experience for agent learning
        agent.remember(state, action, reward, next_state, done)
        
        # Train agent
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if done or truncated:
            break
    
    # Log progress
    accuracy = (episode_reward + episode_steps) / (2 * episode_steps)
    print(f"Episode {episode}: Steps={episode_steps}, "
          f"Reward={episode_reward}, Accuracy={accuracy:.2%}")
```

### Multi-Episode Evaluation

```python
def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent performance over multiple episodes."""
    total_rewards = []
    total_accuracies = []
    
    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        while True:
            action = agent.act(state, training=False)
            state, reward, done, truncated, _ = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            
            if done or truncated:
                break
        
        accuracy = (episode_reward + episode_steps) / (2 * episode_steps)
        total_rewards.append(episode_reward)
        total_accuracies.append(accuracy)
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_accuracy': np.mean(total_accuracies),
        'std_accuracy': np.std(total_accuracies)
    }

# Evaluate trained agent
results = evaluate_agent(agent, env, num_episodes=50)
print(f"Evaluation Results:")
print(f"  Mean Accuracy: {results['mean_accuracy']:.2%} ± {results['std_accuracy']:.2%}")
print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
```

## Environment Properties

### Observation Space

- **Type**: `gymnasium.spaces.Box`
- **Shape**: `(n_features,)` where `n_features` is the number of feature columns
- **Data Type**: `np.float32`
- **Range**: `[-inf, inf]` (features are typically normalized)

### Action Space

- **Type**: `gymnasium.spaces.Discrete`
- **Range**: `[0, n_classes-1]` where `n_classes` is the number of unique labels
- **Description**: Each action represents predicting a specific class

### Reward Structure

| Condition | Reward | Description |
|-----------|---------|-------------|
| Correct prediction | `+1` | Agent correctly classified the sample |
| Incorrect prediction | `-1` | Agent misclassified the sample |

### Episode Termination

- **Natural termination**: When all samples in the dataset have been processed
- **Truncation**: Not used in this environment
- **Episode length**: Varies based on dataset size

## Command Line Interface

The environment module also provides CLI commands for testing and validation:

### `train_env`

Test the environment with random actions.

```bash
python -m rl_ids.environments.ids_env train_env \
    --data-path data/processed/train.csv \
    --feature-cols "Destination_Port,Flow_Duration,Total_Fwd_Packets" \
    --episodes 10
```

### `validate_data`

Validate dataset format and features.

```bash
python -m rl_ids.environments.ids_env validate_data \
    --data-path data/processed/train.csv \
    --feature-cols "Destination_Port,Flow_Duration,Total_Fwd_Packets"
```

## See Also

- [DQN Agent Module](agents.md) - Reinforcement learning agent
- [Training Module](modeling.md) - Training pipelines  
- [Data Processing](make_dataset.md) - Dataset preparation
