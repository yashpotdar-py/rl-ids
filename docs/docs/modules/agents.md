# RL Agents

This document covers the reinforcement learning agents used in RL-IDS, specifically the Deep Q-Network (DQN) implementation.

## Overview

The RL-IDS system uses Deep Q-Network (DQN) agents for network intrusion detection. The implementation is located in `rl_ids/agents/dqn_agent.py` and provides a configurable, high-performance agent suitable for multi-class classification tasks.

## DQN Agent Architecture

### Core Components

The DQN agent consists of several key components:

1. **Deep Q-Network Model** - Neural network for Q-value estimation
2. **Target Network** - Stabilized target for Q-learning updates
3. **Replay Buffer** - Experience replay for stable learning
4. **Epsilon-Greedy Policy** - Exploration vs exploitation strategy

### Network Architecture

```python
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.fc(x)
```

**Default Architecture:**
- Input Layer: 78 features (CICIDS2017 feature set)
- Hidden Layers: Configurable (default: [1024, 512, 256, 128])
- Output Layer: 15 classes (attack types + benign)
- Activation: ReLU between layers
- Output: Raw Q-values (no activation)

## DQN Configuration

### Configuration Class

```python
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
    hidden_dims: List[int] = Field([256, 128], description="Hidden layer dimensions")
```

### Training Configuration

The training script (`rl_ids/modeling/train.py`) provides extensive configuration options:

```python
# Core training parameters
num_episodes: int = 500              # Number of training episodes
target_update_interval: int = 10     # Target network update frequency
lr: float = 1e-4                     # Learning rate
gamma: float = 0.995                 # Discount factor

# Exploration parameters
epsilon: float = 1.0                 # Initial exploration rate
eps_decay: float = 0.9995           # Epsilon decay rate
eps_min: float = 0.01               # Minimum epsilon value

# Network architecture
hidden_dims: str = "1024,512,256,128"  # Hidden layer dimensions
dropout_rate: float = 0.2            # Dropout rate for regularization
use_layer_norm: bool = True          # Use layer normalization

# Training optimization
memory_size: int = 100000            # Replay buffer size
batch_size: int = 256                # Training batch size
warmup_steps: int = 1000             # Warmup steps before training

# Advanced features
curriculum_learning: bool = True     # Use curriculum learning
double_dqn: bool = True             # Use Double DQN
dueling: bool = True                # Use Dueling DQN architecture
prioritized_replay: bool = False    # Use prioritized experience replay
```

## DQN Agent Implementation

### Agent Class

```python
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

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.model = DQN(config.state_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.target_model = DQN(config.state_dim, config.action_dim, config.hidden_dims).to(self.device)
        self.update_target()

        # Training components
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.memory = deque(maxlen=config.memory_size)

        # Training metrics
        self.training_step = 0
        self.episode_count = 0
```

### Key Methods

#### Action Selection

```python
def act(self, state: np.ndarray, training: bool = True) -> int:
    """Choose action using epsilon-greedy policy."""
    if training and random.random() < self.epsilon:
        action = random.randint(0, self.action_dim - 1)
        return action

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.model(state_tensor)

    action = q_values.argmax().item()
    return action
```

#### Experience Storage

```python
def remember(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
    """Store experience in replay buffer."""
    self.memory.append((state, action, reward, next_state, done))
```

#### Training Step

```python
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
    return loss.item()
```

## Training Process

### Enhanced Training Pipeline

The training process (`rl_ids/modeling/train.py`) includes several advanced features:

#### 1. Curriculum Learning

Training progresses through stages of increasing difficulty:

```python
if curriculum_learning:
    stage_size = num_episodes // curriculum_stages
    for i in range(curriculum_stages):
        start_ep = i * stage_size
        end_ep = min((i + 1) * stage_size, num_episodes)
        curriculum_episodes.append((start_ep, end_ep, i + 1))
```

#### 2. Learning Rate Scheduling

Adaptive learning rate adjustment:

```python
if lr_scheduler == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        agent.optimizer, T_max=num_episodes, eta_min=lr * 0.01
    )
elif lr_scheduler == "step":
    scheduler = torch.optim.lr_scheduler.StepLR(
        agent.optimizer, step_size=num_episodes // 4, gamma=0.5
    )
```

#### 3. Early Stopping

Prevents overfitting with validation-based early stopping:

```python
if val_accuracy > best_val_accuracy:
    best_val_accuracy = val_accuracy
    patience_counter = 0
    agent.save_model(best_model_path)
else:
    patience_counter += 1

if patience_counter >= early_stopping_patience:
    logger.info(f"Early stopping triggered at episode {episode}")
    break
```

#### 4. Gradient Clipping

Stabilizes training with gradient clipping:

```python
if grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), grad_clip)
```

### Training Metrics

The system tracks comprehensive training metrics:

```python
# Core metrics
train_rewards = []           # Episode rewards
train_losses = []           # Training losses
train_accuracies = []       # Episode accuracies
val_accuracies = []         # Validation accuracies
learning_rates = []         # Learning rate progression
episode_lengths = []        # Steps per episode

# Advanced metrics
reward_stability = np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8)
accuracy_stability = np.std(recent_accuracies)
avg_episode_length = np.mean(episode_lengths)
```

## Model Evaluation

### Evaluation Pipeline

The evaluation script (`rl_ids/modeling/evaluate.py`) provides comprehensive model assessment:

#### Performance Metrics

```python
# Classification metrics
accuracy = accuracy_score(all_true_labels, all_predictions)
precision, recall, f1, _ = precision_recall_fscore_support(
    all_true_labels, all_predictions, average='weighted'
)

# Confusion matrix
cm = confusion_matrix(all_true_labels, all_predictions)

# Classification report
report = classification_report(
    all_true_labels, all_predictions, 
    target_names=class_names, 
    output_dict=True
)
```

#### Confidence Analysis

```python
# High confidence predictions
high_conf_mask = np.array(all_confidences) >= confidence_threshold
high_conf_accuracy = accuracy_score(
    np.array(all_true_labels)[high_conf_mask],
    np.array(all_predictions)[high_conf_mask]
)

# Confidence distribution by class
confidence_by_class = {}
for class_idx, class_name in enumerate(class_names):
    class_mask = np.array(all_true_labels) == class_idx
    if np.any(class_mask):
        confidence_by_class[class_name] = np.mean(np.array(all_confidences)[class_mask])
```

#### Prediction Timing

```python
# Prediction performance
avg_prediction_time = np.mean(prediction_times) * 1000  # Convert to ms
predictions_per_second = 1.0 / np.mean(prediction_times)
```

## Usage Examples

### Basic Agent Training

```python
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig

# Create configuration
config = DQNConfig(
    state_dim=78,
    action_dim=15,
    lr=1e-4,
    gamma=0.995,
    hidden_dims=[1024, 512, 256, 128]
)

# Initialize agent
agent = DQNAgent(config)

# Training loop
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    
    while not done:
        action = agent.act(state, training=True)
        next_state, reward, done, _, info = env.step(action)
        
        agent.remember(state, action, reward, next_state, done)
        
        if len(agent.memory) >= warmup_steps:
            loss = agent.replay()
        
        state = next_state
    
    # Update target network periodically
    if episode % target_update_interval == 0:
        agent.update_target()
```

### Model Inference

```python
# Load trained model
agent = DQNAgent(config)
agent.load_model("models/dqn_model_best.pt")
agent.epsilon = 0.0  # Disable exploration

# Make predictions
features = extract_features(network_packet)
action = agent.act(features, training=False)
attack_type = class_names[action]
```
