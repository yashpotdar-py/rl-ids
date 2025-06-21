# Training & Evaluation Module

## Overview

The `rl_ids.modeling` module provides comprehensive training and evaluation pipelines for the DQN-based intrusion detection system. It includes advanced training features such as curriculum learning, early stopping, and detailed performance analysis.

## Modules

### `rl_ids.modeling.train`

Advanced DQN training pipeline with optimization techniques.

### `rl_ids.modeling.evaluate`

Comprehensive model evaluation and performance analysis.

---

## Training Module (`train.py`)

### `main()` Function

Main training function with extensive configuration options.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_data_path` | `Path` | `TRAIN_DATA_FILE` | Path to training data |
| `val_data_path` | `Path` | `VAL_DATA_FILE` | Path to validation data |
| `num_episodes` | `int` | `500` | Number of training episodes |
| `target_update_interval` | `int` | `10` | Target network update frequency |
| `lr` | `float` | `1e-4` | Learning rate |
| `lr_scheduler` | `str` | `"cosine"` | LR scheduler type |
| `gamma` | `float` | `0.995` | Discount factor |
| `epsilon` | `float` | `1.0` | Initial exploration rate |
| `eps_decay` | `float` | `0.9995` | Epsilon decay rate |
| `eps_min` | `float` | `0.01` | Minimum epsilon |
| `memory_size` | `int` | `100000` | Replay buffer size |
| `batch_size` | `int` | `256` | Training batch size |
| `hidden_dims` | `str` | `"1024,512,256,128"` | Network architecture |
| `dropout_rate` | `float` | `0.2` | Dropout rate |
| `use_layer_norm` | `bool` | `True` | Use layer normalization |
| `save_interval` | `int` | `50` | Model save frequency |
| `validation_interval` | `int` | `5` | Validation frequency |
| `early_stopping_patience` | `int` | `30` | Early stopping patience |
| `grad_clip` | `float` | `1.0` | Gradient clipping value |
| `weight_decay` | `float` | `1e-5` | Weight decay |
| `double_dqn` | `bool` | `True` | Use Double DQN |
| `dueling` | `bool` | `True` | Use Dueling DQN |
| `curriculum_learning` | `bool` | `True` | Use curriculum learning |

**Examples**

```bash
# Basic training
python -m rl_ids.modeling.train

# Advanced training with custom parameters
python -m rl_ids.modeling.train \
    --num-episodes 1000 \
    --lr 1e-4 \
    --batch-size 256 \
    --hidden-dims "1024,512,256,128" \
    --gamma 0.995 \
    --epsilon 1.0 \
    --eps-decay 0.9995 \
    --memory-size 200000 \
    --double-dqn \
    --dueling \
    --curriculum-learning

# Training with specific data paths
python -m rl_ids.modeling.train \
    --train-data-path data/processed/train.csv \
    --val-data-path data/processed/val.csv \
    --models-dir models/experiment_1
```

### Advanced Features

#### Curriculum Learning

Progressive training difficulty to improve learning stability.

```python
# Automatically enabled with --curriculum-learning
# Stages:
# 1. Easy samples (high confidence labels)
# 2. Medium difficulty samples
# 3. Full dataset including difficult samples
```

#### Early Stopping

Prevents overfitting by monitoring validation performance.

```python
# Monitors validation accuracy
# Stops training if no improvement for 'patience' episodes
# Automatically saves best model
```

#### Learning Rate Scheduling

Adaptive learning rate adjustment.

| Scheduler | Description |
|-----------|-------------|
| `"cosine"` | Cosine annealing |
| `"step"` | Step decay |
| `"none"` | Fixed learning rate |

#### Advanced DQN Techniques

- **Double DQN**: Reduces overestimation bias
- **Dueling DQN**: Separate value and advantage streams
- **Prioritized Replay**: Focus on important experiences

---

## Evaluation Module (`evaluate.py`)

### `main()` Function

Comprehensive model evaluation with detailed analysis.

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `test_data_path` | `Path` | `TEST_DATA_FILE` | Path to test data |
| `model_path` | `Path` | `MODELS_DIR / "dqn_model_final.pt"` | Model to evaluate |
| `test_episodes` | `int` | `15` | Number of test episodes |
| `max_steps_per_episode` | `int` | `20000` | Max steps per episode |
| `save_predictions` | `bool` | `True` | Save detailed predictions |
| `use_best_model` | `bool` | `True` | Use best model vs final |
| `detailed_analysis` | `bool` | `True` | Perform error analysis |
| `confidence_threshold` | `float` | `0.8` | High-confidence threshold |

**Examples**

```bash
# Basic evaluation
python -m rl_ids.modeling.evaluate

# Evaluate specific model
python -m rl_ids.modeling.evaluate \
    --model-path models/dqn_model_best.pt \
    --test-episodes 25 \
    --detailed-analysis

# Quick evaluation without detailed analysis
python -m rl_ids.modeling.evaluate \
    --test-episodes 5 \
    --detailed-analysis false \
    --save-predictions false
```

### Generated Reports

The evaluation generates comprehensive reports in the `reports/` directory:

#### Summary Reports
- **`evaluation_summary_enhanced.csv`**: Overall performance metrics
- **`evaluation_episode_details_enhanced.csv`**: Per-episode results

#### Detailed Analysis
- **`evaluation_detailed_predictions_enhanced.csv`**: Individual predictions
- **`evaluation_classification_report.csv`**: Per-class metrics
- **`evaluation_confusion_matrix.csv`**: Confusion matrix data

#### Visualizations
- **`evaluation_overview.png`**: Comprehensive performance overview
- **`class_analysis.png`**: Per-class performance analysis
- **`error_analysis.png`**: Error pattern analysis
- **`enhanced_confusion_matrix.png`**: Detailed confusion matrix

---

## Usage Examples

### Complete Training Pipeline

```python
from pathlib import Path
from rl_ids.modeling.train import main as train_main
from rl_ids.modeling.evaluate import main as evaluate_main

# Training with custom parameters
train_main(
    num_episodes=1000,
    lr=1e-4,
    batch_size=256,
    hidden_dims="1024,512,256,128",
    gamma=0.995,
    double_dqn=True,
    dueling=True,
    curriculum_learning=True,
    early_stopping_patience=30
)

# Evaluate the trained model
evaluate_main(
    test_episodes=25,
    detailed_analysis=True,
    save_predictions=True
)
```

### Custom Training Loop

```python
import torch
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.environments.ids_env import IDSDetectionEnv
from rl_ids.config import TRAIN_DATA_FILE

# Load data and setup
feature_cols = [...] # Your feature columns
env = IDSDetectionEnv(
    data_path=TRAIN_DATA_FILE,
    feature_cols=feature_cols
)

# Initialize agent
config = DQNConfig(
    state_dim=len(feature_cols),
    action_dim=env.action_space.n,
    lr=1e-4,
    gamma=0.995,
    hidden_dims=[1024, 512, 256, 128]
)
agent = DQNAgent(config)

# Training metrics
episode_rewards = []
episode_accuracies = []
losses = []

# Training loop
num_episodes = 1000
target_update_interval = 10

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_losses = []
    
    while True:
        # Agent action
        action = agent.act(state, training=True)
        
        # Environment step
        next_state, reward, done, truncated, info = env.step(action)
        
        # Store experience
        agent.remember(state, action, reward, next_state, done)
        
        # Train agent
        if len(agent.memory) > agent.batch_size:
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
        
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if done or truncated:
            break
    
    # Update target network
    if episode % target_update_interval == 0:
        agent.update_target()
    
    # Calculate metrics
    accuracy = (episode_reward + episode_steps) / (2 * episode_steps)
    avg_loss = np.mean(episode_losses) if episode_losses else 0
    
    episode_rewards.append(episode_reward)
    episode_accuracies.append(accuracy)
    losses.append(avg_loss)
    
    # Logging
    if episode % 50 == 0:
        print(f"Episode {episode}:")
        print(f"  Reward: {episode_reward}")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Save model checkpoint
        agent.save_model(f"models/episode_{episode}.pt")

# Save final model
agent.save_model("models/dqn_model_final.pt")
```

### Validation During Training

```python
def validate_agent(agent, val_env, max_steps=5000):
    """Validate agent performance."""
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Disable exploration
    
    state, _ = val_env.reset()
    total_reward = 0
    step_count = 0
    correct_predictions = 0
    
    while step_count < max_steps:
        action = agent.act(state, training=False)
        next_state, reward, done, truncated, info = val_env.step(action)
        
        total_reward += reward
        step_count += 1
        
        if reward > 0:
            correct_predictions += 1
        
        state = next_state
        
        if done or truncated:
            break
    
    agent.epsilon = original_epsilon
    
    accuracy = correct_predictions / step_count if step_count > 0 else 0
    return {
        'accuracy': accuracy,
        'total_reward': total_reward,
        'steps': step_count
    }

# Use in training loop
if episode % validation_interval == 0:
    val_results = validate_agent(agent, val_env)
    print(f"Validation - Accuracy: {val_results['accuracy']:.2%}")
```

## Performance Metrics

### Training Metrics

- **Episode Reward**: Cumulative reward per episode
- **Accuracy**: Percentage of correct classifications
- **Loss**: Training loss value
- **Epsilon**: Current exploration rate
- **Learning Rate**: Current learning rate (if using scheduler)

### Evaluation Metrics

- **Overall Accuracy**: Total correct predictions / total predictions
- **Per-Class Metrics**: Precision, recall, F1-score for each class
- **Confidence Analysis**: Performance at different confidence levels
- **Confusion Matrix**: Detailed classification results
- **Prediction Times**: Average inference time per sample

## Configuration Tips

### Memory Management

```bash
# For limited GPU memory
python -m rl_ids.modeling.train --batch-size 64 --memory-size 50000

# For high-memory systems
python -m rl_ids.modeling.train --batch-size 512 --memory-size 500000
```

### Learning Rate Tuning

```bash
# Conservative learning
python -m rl_ids.modeling.train --lr 5e-5 --lr-scheduler cosine

# Aggressive learning
python -m rl_ids.modeling.train --lr 1e-3 --lr-scheduler step
```

### Network Architecture

```bash
# Lightweight model
python -m rl_ids.modeling.train --hidden-dims "256,128"

# Large model
python -m rl_ids.modeling.train --hidden-dims "2048,1024,512,256,128"
```

## See Also

- [DQN Agent Module](agents.md) - Reinforcement learning implementation
- [Environment Module](environments.md) - Training environment
- [Visualization Module](plots.md) - Analysis and plotting tools
