# RL-IDS Modules

This section provides detailed documentation for all RL-IDS system modules, covering their architecture, functionality, and usage.

## Overview

The RL-IDS system is built with a modular architecture that separates concerns and enables easy extension and maintenance. Each module handles specific aspects of the intrusion detection pipeline.

## Core Modules

### [Reinforcement Learning Agents](agents.md)
The heart of the RL-IDS system, implementing Deep Q-Network (DQN) algorithms for adaptive threat detection.

**Key Components:**
- **DQN Agent**: Main reinforcement learning agent
- **Neural Networks**: Q-network and target network architectures
- **Experience Replay**: Memory management for training
- **Training Pipeline**: Complete training and evaluation framework

**Features:**
- Adaptive learning from network traffic patterns
- Real-time decision making for threat classification
- Continuous improvement through reward-based learning
- Support for various network environments

### [Feature Extraction](features.md)
Comprehensive feature engineering module that transforms raw network data into meaningful representations for machine learning.

**Key Components:**
- **CICIDS2017 Features**: 78 standardized network flow features
- **Flow Tracking**: Bidirectional flow analysis and aggregation
- **Statistical Analysis**: Time-series and distribution analysis
- **Data Preprocessing**: Normalization and validation

**Features:**
- Real-time feature extraction from network packets
- Statistical traffic characterization
- Protocol-specific feature analysis
- Scalable processing pipeline

## Supporting Modules

### Configuration Module (`rl_ids/config.py`)
Central configuration management for all system components.

**Features:**
- Environment-based configuration
- Model and training parameters
- Network monitoring settings
- API and service configuration

**Key Settings:**
```python
# Model Configuration
MODEL_CONFIG = {
    'input_size': 78,
    'hidden_layers': [256, 128, 64],
    'output_size': 2,  # Binary classification
    'learning_rate': 0.001,
    'gamma': 0.99,
    'epsilon': 0.1
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'episodes': 500,
    'memory_size': 10000,
    'target_update': 100
}

# Environment Configuration
ENV_CONFIG = {
    'max_steps': 1000,
    'reward_function': 'balanced',
    'state_normalization': True
}
```

### Environment Module (`rl_ids/environments/ids_env.py`)
Gymnasium-compatible environment for training RL agents on intrusion detection tasks.

**Features:**
- CICIDS2017 dataset integration
- Configurable reward functions
- Episode management
- State space normalization

**Environment Specifications:**
- **State Space**: 78-dimensional continuous space (network features)
- **Action Space**: Discrete space for classification decisions
- **Reward Structure**: Based on detection accuracy and false positive rates
- **Episode Length**: Configurable based on dataset size

### Plotting and Visualization (`rl_ids/plots.py`)
Comprehensive visualization tools for training analysis and model evaluation.

**Features:**
- Training progress visualization
- Performance metrics plotting
- Confusion matrix generation
- Feature importance analysis

**Available Plots:**
- Training loss and reward curves
- Episode performance trends
- Classification performance metrics
- Feature distribution analysis

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    Module Dependency Graph                  │
├─────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Config    │    │   Features  │    │   Agents    │    │
│  │             │◄───┤             │◄───┤             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│        ▲                   ▲                   ▲          │
│        │                   │                   │          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Plots     │    │Environment  │    │  Training   │    │
│  │             │    │             │    │             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│                                                           │
└─────────────────────────────────────────────────────────────┘
```

## Integration Patterns

### Data Flow Integration
```python
# Example: Complete detection pipeline
from rl_ids.make_dataset import extract_features
from rl_ids.agents.dqn_agent import DQNAgent
from rl_ids.config import MODEL_CONFIG

# Initialize components
agent = DQNAgent(MODEL_CONFIG)
raw_data = capture_network_traffic()

# Process data through pipeline
features = extract_features(raw_data)
prediction = agent.predict(features)
```

### Training Integration
```python
# Example: Training pipeline integration
from rl_ids.environments.ids_env import IDSEnvironment
from rl_ids.modeling.train import train_dqn_agent
from rl_ids.plots import plot_training_metrics

# Set up training environment
env = IDSEnvironment()
agent = train_dqn_agent(env, episodes=500)

# Visualize results
plot_training_metrics(agent.training_history)
```

### API Integration
```python
# Example: API service integration
from api.main import app
from rl_ids.agents.dqn_agent import DQNAgent

# Load trained model for API
agent = DQNAgent.load_model('models/dqn_model_best.pt')

# API automatically uses loaded model for predictions
```

## Module Configuration

### Environment Variables
```bash
# Model Configuration
MODEL_PATH=models/dqn_model_best.pt
FEATURE_SCALER_PATH=models/feature_scaler.pkl

# Training Configuration
EPISODES=500
BATCH_SIZE=32
LEARNING_RATE=0.001

# Environment Configuration
MAX_STEPS=1000
REWARD_TYPE=balanced
```

### Configuration Files
```python
# config.py structure
class Config:
    # Model settings
    MODEL_CONFIG = {...}
    
    # Training settings
    TRAINING_CONFIG = {...}
    
    # Environment settings
    ENV_CONFIG = {...}
    
    # API settings
    API_CONFIG = {...}
```

## Performance Considerations

### Memory Usage
- **Feature Extraction**: O(n) where n is the number of flows
- **Agent Memory**: Configurable replay buffer size
- **Training**: Batch processing for memory efficiency
- **Inference**: Single-pass processing for real-time performance

### Computational Complexity
- **Feature Calculation**: Linear with packet count
- **Model Inference**: Constant time per prediction
- **Training**: Depends on dataset size and architecture
- **Environment**: Minimal overhead for state transitions

### Scalability
- **Horizontal**: Multiple agent instances for load distribution
- **Vertical**: GPU acceleration for training and inference
- **Memory**: Configurable buffer sizes and batch processing
- **Storage**: Efficient model checkpointing and data handling

## Best Practices

### Module Usage
1. **Initialize Configuration First**: Load configuration before other modules
2. **Use Consistent Interfaces**: Follow established patterns for integration
3. **Handle Errors Gracefully**: Implement proper error handling and logging
4. **Monitor Performance**: Use built-in metrics and monitoring tools

### Extension Guidelines
1. **Follow Module Structure**: Maintain consistent organization
2. **Document Interfaces**: Provide clear API documentation
3. **Write Tests**: Include unit tests for new functionality
4. **Maintain Compatibility**: Ensure backward compatibility when possible

### Development Workflow
1. **Local Testing**: Use provided test datasets and configurations
2. **Integration Testing**: Test module interactions thoroughly
3. **Performance Testing**: Benchmark against expected performance
4. **Documentation**: Update documentation for any changes

## Getting Started

### Quick Start
1. **Read Module Documentation**: Start with [Agents](agents.md) and [Features](features.md)
2. **Review Configuration**: Check `rl_ids/config.py` for available settings
3. **Run Examples**: Execute provided example scripts
4. **Explore API**: Use the API documentation for integration examples

### Development Setup
1. **Install Dependencies**: Follow installation guide
2. **Configure Environment**: Set up configuration files
3. **Run Tests**: Verify module functionality
4. **Start Development**: Begin with existing examples and extend

For detailed information about each module, please refer to their individual documentation pages.