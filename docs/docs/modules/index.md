# Module Reference

## Overview

The RL-IDS system is organized into several key modules, each handling specific aspects of the reinforcement learning-based intrusion detection system. This reference provides comprehensive documentation for all major components, classes, and functions.

## Core Modules

| Module | Description | Key Components |
|--------|-------------|----------------|
| [`rl_ids.agents`](agents.md) | Enhanced DQN agent implementation | `DQNAgent`, `DQNConfig`, `DQN` |
| [`rl_ids.environments`](environments.md) | Custom Gymnasium environment | `IDSDetectionEnv` |
| [`rl_ids.modeling`](modeling.md) | Training and evaluation pipelines | `train`, `evaluate` |
| [`api`](../api/index.md) | FastAPI service for predictions | `IDSPredictionService`, `main` |

## Utility Modules

| Module | Description | Key Components |
|--------|-------------|----------------|
| [`rl_ids.config`](config.md) | Configuration and path management | Paths, logging setup, environment variables |
| [`rl_ids.make_dataset`](make_dataset.md) | Data preprocessing pipeline | `DataGenerator`, `DataPreprocessor` |
| [`rl_ids.plots`](plots.md) | Advanced visualization tools | `IDSPlotter` |

## Architecture Overview

### Data Flow
```
Raw CICIDS2017 Data → Data Preprocessing → Feature Engineering → Train/Val/Test Split
                                            ↓
Training Environment ← Custom Gym Environment ← Processed Data
        ↓
DQN Agent Training → Model Checkpoints → Best Model Selection
        ↓
Model Evaluation → Performance Reports → Visualization
        ↓
Production API → Real-time Predictions → Monitoring
```

### Component Interactions

**Training Pipeline:**
1. **Data Module** (`make_dataset`) processes raw CICIDS2017 data
2. **Environment Module** (`environments`) provides Gym interface for RL training
3. **Agent Module** (`agents`) implements DQN algorithm with advanced features
4. **Modeling Module** (`modeling`) orchestrates training and evaluation
5. **Plotting Module** (`plots`) generates comprehensive visualizations

**Inference Pipeline:**
1. **API Module** (`api`) provides REST interface for predictions
2. **Service Layer** (`api.services`) handles model loading and predictions
3. **Models** (`api.models`) define request/response schemas
4. **Client** (`api.client`) provides programmatic API access

## Module Dependencies

```
rl_ids/
├── config.py              # Base configuration (imported by all modules)
├── make_dataset.py         # Data preprocessing (depends on: config)
├── agents/
│   └── dqn_agent.py       # DQN implementation (depends on: config)
├── environments/
│   └── ids_env.py         # Gym environment (depends on: config)
├── modeling/
│   ├── train.py           # Training pipeline (depends on: agents, environments, config)
│   └── evaluate.py        # Evaluation pipeline (depends on: agents, environments, plots)
└── plots.py               # Visualization (depends on: config)

api/
├── main.py                # FastAPI app (depends on: services, models)
├── services.py            # Prediction service (depends on: rl_ids.agents, config)
├── models.py              # Pydantic schemas (standalone)
└── client.py              # API client (depends on: models)
```

## Quick Reference

### Essential Classes

| Class | Module | Purpose |
|-------|--------|---------|
| `DQNAgent` | `rl_ids.agents` | Main reinforcement learning agent |
| `DQNConfig` | `rl_ids.agents` | Agent configuration and hyperparameters |
| `IDSDetectionEnv` | `rl_ids.environments` | Custom Gym environment for training |
| `IDSPredictionService` | `api.services` | Production prediction service |
| `IDSPlotter` | `rl_ids.plots` | Comprehensive visualization tools |

### Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `main()` | `rl_ids.modeling.train` | Training pipeline entry point |
| `main()` | `rl_ids.modeling.evaluate` | Evaluation pipeline entry point |
| `load_and_process_data()` | `rl_ids.make_dataset` | Data preprocessing function |
| `predict()` | `api.services` | Single prediction function |
| `predict_batch()` | `api.services` | Batch prediction function |

### Configuration Classes

| Config Class | Module | Purpose |
|--------------|--------|---------|
| `DQNConfig` | `rl_ids.agents` | DQN agent hyperparameters |
| `IDSPredictionRequest` | `api.models` | API request schema |
| `IDSPredictionResponse` | `api.models` | API response schema |

## Usage Patterns

### Training Workflow
```python
from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.environments.ids_env import IDSDetectionEnv
from rl_ids.config import TRAIN_DATA_FILE

# Configure agent
config = DQNConfig(state_dim=77, action_dim=15)
agent = DQNAgent(config)

# Setup environment
env = IDSDetectionEnv(data_path=TRAIN_DATA_FILE, 
                      feature_cols=feature_columns)

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    while True:
        action = agent.act(state)
        next_state, reward, done, _, info = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        
        if len(agent.memory) > batch_size:
            agent.replay()
        
        if done:
            break
        state = next_state
```

### API Usage
```python
from api.client import IDSAPIClient
import asyncio

async def predict_sample():
    client = IDSAPIClient("http://localhost:8000")
    
    # Single prediction
    features = [0.1] * 77  # Network traffic features
    result = await client.predict(features)
    
    print(f"Prediction: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Is Attack: {result['is_attack']}")

asyncio.run(predict_sample())
```

### Visualization
```python
from rl_ids.plots import IDSPlotter
from rl_ids.config import REPORTS_DIR, FIGURES_DIR

plotter = IDSPlotter(figures_dir=FIGURES_DIR, dpi=300)

# Generate comprehensive evaluation plots
plotter.plot_evaluation_overview(REPORTS_DIR)
plotter.plot_class_analysis(REPORTS_DIR)
plotter.plot_error_analysis(REPORTS_DIR)
```

## See Also

- [Getting Started Guide](../getting-started.md) - Setup and first steps
- [API Reference](../api/index.md) - REST API documentation
- [Tutorials](../tutorials/index.md) - Hands-on guides and examples
- [FAQ & Troubleshooting](../faq.md) - Common issues and solutions

rl_ids/
├── agents/
│   └── dqn_agent.py → torch, gymnasium
├── environments/
│   └── ids_env.py   → gymnasium, pandas
├── modeling/
│   ├── train.py     → agents, environments
│   └── evaluate.py  → agents, environments
└── plots.py         → matplotlib, seaborn
```

## Quick Navigation

- [DQN Agent Documentation](agents.md)
- [Environment Documentation](environments.md) 
- [Training & Evaluation](modeling.md)
- [API Service](../api/index.md)
- [Configuration](config.md)
- [Data Processing](make_dataset.md)
- [Visualization](plots.md)
