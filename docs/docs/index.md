# RL-IDS Adaptive System Documentation

![Architecture Diagram](../assets/architecture.png)

## Overview

The **RL-IDS Adaptive System** is a cutting-edge Reinforcement Learning-driven Adaptive Intrusion Detection System designed to detect network intrusions using Deep Q-Network (DQN) agents. This system combines advanced machine learning techniques with real-time network traffic analysis to provide adaptive and accurate threat detection.

Built on the CICIDS2017 dataset, the system employs sophisticated DQN algorithms including Double DQN, Dueling DQN, and prioritized experience replay to achieve state-of-the-art performance in network intrusion detection. The system provides both training capabilities and production-ready API services for real-time threat detection.

## Key Features

- 🤖 **Advanced DQN Agent**: Deep Q-Network with Double DQN, Dueling architecture, and prioritized replay
- 🌐 **Real-time API**: FastAPI-based REST service for live predictions with async processing
- 📊 **Comprehensive Analytics**: Detailed visualization and reporting tools with confusion matrices and performance metrics
- 🔄 **Adaptive Learning**: Continuous improvement through reinforcement learning with curriculum learning
- 🚀 **Production Ready**: Docker containerization, health monitoring, and scalable deployment
- 📈 **Advanced Monitoring**: MLflow integration for experiment tracking and model versioning
- 🎯 **Multi-class Detection**: Supports 15 different attack types from CICIDS2017 dataset
- ⚡ **High Performance**: Optimized for low-latency predictions with batch processing support

## Architecture Components

### Core Modules
- **[`rl_ids.agents`](modules/agents.md)** - Enhanced DQN agent with advanced features
- **[`rl_ids.environments`](modules/environments.md)** - Custom Gymnasium environment for IDS training
- **[`rl_ids.modeling`](modules/modeling.md)** - Training and evaluation pipelines with advanced optimizations
- **[`api`](api/index.md)** - FastAPI service for real-time predictions and monitoring

### Data Pipeline
- **Raw Data Processing** - CICIDS2017 dataset preprocessing and cleaning
- **Feature Engineering** - Network traffic feature extraction and normalization
- **Data Balancing** - SMOTE and advanced sampling techniques for class imbalance
- **Train/Validation/Test Split** - Stratified splitting for robust evaluation

### Machine Learning Pipeline
- **Reinforcement Learning** - DQN-based adaptive learning from network traffic patterns
- **Experience Replay** - Prioritized replay buffer for efficient learning
- **Curriculum Learning** - Progressive difficulty adjustment during training
- **Model Evaluation** - Comprehensive performance analysis with detailed metrics

## Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd rl_ids

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage
```bash
# Process CICIDS2017 data
python -m rl_ids.make_dataset

# Train DQN model with advanced features
python -m rl_ids.modeling.train --double_dqn --dueling --prioritized_replay

# Evaluate model performance
python -m rl_ids.modeling.evaluate

# Start FastAPI service
python -m api.main
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t rl-ids-api .
docker run -p 8000:8000 rl-ids-api

# Or use Docker Compose
docker-compose up -d
```

## Project Structure

```
rl_ids/
├── api/                    # FastAPI service for real-time predictions
│   ├── main.py            # FastAPI application and endpoints
│   ├── models.py          # Pydantic models for API schemas
│   ├── services.py        # Prediction service implementation
│   └── client.py          # API client for testing and integration
├── rl_ids/
│   ├── agents/            # DQN agent implementation
│   │   └── dqn_agent.py   # Enhanced DQN with advanced features
│   ├── environments/      # Custom Gymnasium environment
│   │   └── ids_env.py     # IDS detection environment
│   ├── modeling/          # Training and evaluation pipelines
│   │   ├── train.py       # Advanced training with curriculum learning
│   │   └── evaluate.py    # Comprehensive model evaluation
│   ├── config.py          # Configuration and path management
│   ├── make_dataset.py    # Data preprocessing pipeline
│   └── plots.py           # Advanced visualization tools
├── data/                  # Dataset storage and processing
│   ├── raw/              # Original CICIDS2017 dataset files
│   ├── processed/        # Preprocessed and split datasets
│   └── external/         # External data sources
├── models/               # Trained model storage
│   ├── dqn_model_best.pt # Best performing model
│   └── episodes/         # Episode-wise model checkpoints
├── reports/              # Analysis reports and visualizations
│   ├── figures/          # Generated plots and charts
│   └── *.csv            # Performance metrics and detailed results
└── docs/                 # Comprehensive documentation
    ├── docs/             # Documentation source files
    └── mkdocs.yml        # Documentation configuration
```

## Dataset Information

The system is designed to work with the **CICIDS2017** dataset, which contains:
- **2.8M+ network traffic samples** from realistic network environments
- **15 different attack types** including DDoS, PortScan, Brute Force, XSS, SQL Injection
- **79 network traffic features** extracted using CICFlowMeter
- **Realistic attack scenarios** generated in a controlled environment

## Performance Highlights

- **High Accuracy**: Achieves >95% accuracy on CICIDS2017 test set
- **Low Latency**: <10ms average prediction time for real-time detection
- **Scalable**: Handles batch predictions efficiently with async processing
- **Robust**: Comprehensive error handling and confidence-based predictions
- **Adaptive**: Continuous learning capabilities through reinforcement learning

## Next Steps

- [Getting Started Guide](getting-started.md) - Complete setup and first training
- [API Reference](api/index.md) - REST API documentation and usage
- [Module Documentation](modules/index.md) - Detailed component reference
- [Tutorials](tutorials/index.md) - Step-by-step guides and examples
- [FAQ & Troubleshooting](faq.md) - Common issues and solutions

