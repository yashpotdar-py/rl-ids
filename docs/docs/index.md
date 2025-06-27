# RL-IDS Adaptive System

A Reinforcement Learning-driven Adaptive Intrusion Detection System with real-time threat detection capabilities using Deep Q-Network (DQN) agents trained on CICIDS2017 dataset.

## Overview

RL-IDS is an advanced network security system that combines reinforcement learning with intrusion detection to provide adaptive, real-time threat detection. The system uses Deep Q-Network (DQN) agents trained on the CICIDS2017 dataset to classify network traffic and detect various types of attacks including DoS, DDoS, PortScan, Brute Force, XSS, SQL Injection, and Infiltration attacks.

## Key Features

- **Real-time Network Monitoring** - Monitor live network traffic with packet-level analysis using Scapy
- **Deep Q-Network (DQN) Models** - Trained RL agents with configurable architectures and optimization techniques
- **REST API** - FastAPI-based service for real-time predictions and model information
- **CICIDS2017 Feature Extraction** - Extract 78 standardized network flow features from live traffic
- **Multiple Monitoring Modes** - Network interface monitoring and website-specific traffic analysis
- **Advanced Training Pipeline** - Curriculum learning, early stopping, learning rate scheduling
- **Comprehensive Evaluation** - Detailed performance metrics, confusion matrices, and prediction analysis

## Architecture

The system consists of several key components:

- **RL Agents** (`rl_ids/agents/`): DQN implementation with configurable network architectures
- **Training Environment** (`rl_ids/environments/`): Gymnasium-compatible environment for IDS training
- **Data Processing** (`rl_ids/make_dataset.py`): CICIDS2017 dataset preprocessing and normalization
- **FastAPI Service** (`api/`): Production-ready API for real-time threat detection
- **Real-time Monitors** (`network_monitor.py`, `website_monitor.py`): Live traffic analysis tools

## Quick Start

### Prerequisites

- Python 3.9+
- Network capture permissions (sudo for packet capture)
- CICIDS2017 dataset (place CSV files in `data/raw/`)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Process CICIDS2017 dataset
python -m rl_ids.make_dataset

# Train DQN model
python -m rl_ids.modeling.train

# Evaluate trained model
python -m rl_ids.modeling.evaluate
```

### Basic Usage

1. **Start the API server:**
```bash
python run_api.py
```

2. **Monitor network interface:**
```bash
sudo python network_monitor.py wlan0
```

3. **Monitor specific website:**
```bash
python website_monitor.py example.com
```

## Supported Attack Types

The system can detect the following attack types from the CICIDS2017 dataset:

- **Benign Traffic** - Normal network activity
- **DoS Attacks** - DoS Hulk, DoS Slowloris, DoS Slowhttptest, DoS GoldenEye
- **DDoS Attacks** - Distributed Denial of Service
- **Port Scan** - Network reconnaissance attacks
- **Brute Force** - FTP-Patator, SSH-Patator
- **Web Attacks** - SQL Injection, XSS, Brute Force
- **Bot** - Botnet traffic
- **Infiltration** - Network infiltration attacks
- **Heartbleed** - SSL/TLS vulnerability exploitation

## Project Structure

```
rl_ids/
├── rl_ids/                    # Core RL-IDS modules
│   ├── agents/                # DQN agent implementation
│   │   └── dqn_agent.py      # DQN agent with training and inference
│   ├── environments/          # Training environments
│   │   └── ids_env.py        # IDS detection Gymnasium environment
│   ├── modeling/             # Training and evaluation pipeline
│   │   ├── train.py          # Enhanced DQN training with curriculum learning
│   │   └── evaluate.py       # Comprehensive model evaluation
│   ├── config.py             # Project configuration and paths
│   ├── make_dataset.py       # CICIDS2017 data preprocessing
│   └── plots.py              # Visualization utilities
├── api/                      # FastAPI application
│   ├── main.py               # FastAPI application with endpoints
│   ├── models.py             # Pydantic request/response models
│   ├── services.py           # Prediction service implementation
│   ├── client.py             # API client for testing
│   └── config.py             # API configuration
├── models/                   # Trained model files
│   ├── dqn_model_best.pt     # Best performing model
│   ├── dqn_model_final.pt    # Final training epoch model
│   └── episodes/             # Episodic model checkpoints
├── data/                     # Dataset storage
│   ├── raw/                  # Original CICIDS2017 CSV files
│   └── processed/            # Preprocessed train/val/test splits
├── reports/                  # Training and evaluation reports
│   ├── training_metrics.csv  # Training progress metrics
│   ├── evaluation_*.csv      # Evaluation results
│   └── figures/              # Generated plots and visualizations
├── logs/                     # Runtime logs
├── network_monitor.py        # Real-time network interface monitoring
├── website_monitor.py        # Website-specific traffic monitoring
└── run_api.py               # API server startup script
```

## Getting Started

Visit the [User Guide](user-guide/index.md) for detailed installation and usage instructions, [API Reference](api/index.md) for integration details, or [Modules](modules/index.md) for technical implementation details.

## License

This project is licensed under the terms specified in [LICENSE](license.md).