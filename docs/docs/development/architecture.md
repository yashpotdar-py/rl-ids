# System Architecture

This document provides a comprehensive overview of the RL-IDS system architecture, including its components, data flow, and design principles.

## Overview

RL-IDS is a reinforcement learning-driven adaptive intrusion detection system that combines real-time network monitoring with intelligent threat detection. The system is designed with modularity, scalability, and extensibility in mind.

## High-Level Architecture

```
┌───────────────────────────────────────────────┐
│                    RL-IDS System              │
├───────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐   │
│  │  Network Data   │    │   Web Traffic   │   │
│  │  Collection     │    │   Generation    │   │
│  └─────────────────┘    └─────────────────┘   │
│           │                       │           │
│           ▼                       ▼           │
│  ┌───────────────────────────────────────────┐│
│  │  Feature Extraction Layer                 ││
│  │  - CICIDS2017 Feature Engineering         ││
│  │  - Flow-based Analysis                    ││
│  │  - Real-time Processing                   ││
│  └───────────────────────────────────────────┘│
│           │                                   │
│           ▼                                   │
│  ┌───────────────────────────────────────────┐│
│  │  Reinforcement Learning Core              ││
│  │  - DQN Agent (Deep Q-Network)             ││
│  │  - Adaptive Decision Making               ││
│  │  - Continuous Learning                    ││
│  └───────────────────────────────────────────┘│
│           │                                   │
│           ▼                                   │
│  ┌───────────────────────────────────────────┐│
│  │    Detection & Response                   ││
│  │  - Real-time Threat Classification        ││
│  │  - Alert Generation                       ││
│  │  - API Interface                          ││
│  └───────────────────────────────────────────┘│
└───────────────────────────────────────────────┘
```

## Core Components

### 1. Data Collection Layer

#### Network Monitor (`network_monitor.py`)
- **Purpose**: Real-time network packet capture and analysis
- **Key Features**:
  - Live packet capture using raw sockets
  - Protocol analysis (TCP, UDP, HTTP, HTTPS)
  - Flow-based traffic aggregation
  - Statistical feature computation

#### Website Monitor (`website_monitor.py`)
- **Purpose**: Website-specific traffic generation and monitoring
- **Key Features**:
  - Automated web requests
  - Traffic pattern simulation
  - Packet capture for generated traffic
  - Integration with network monitor

### 2. Feature Engineering Layer (`rl_ids/make_dataset.py`)

#### CICIDS2017 Feature Extraction
- **78 Network Flow Features**:
  - Flow duration and packet timing
  - Packet size statistics (min, max, mean, std)
  - Flow flags and protocol information
  - Inter-arrival time analysis
  - Forward/backward flow characteristics

#### Data Processing Pipeline
```python
Raw Packets → Flow Aggregation → Feature Extraction → Normalization → Model Input
```

### 3. Reinforcement Learning Core

#### DQN Agent (`rl_ids/agents/dqn_agent.py`)
- **Architecture**: Deep Q-Network with experience replay
- **Components**:
  - Q-Network: Neural network for action-value estimation
  - Target Network: Stable target for Q-learning updates
  - Experience Replay Buffer: Storage for training experiences
  - Exploration Strategy: Epsilon-greedy with decay

#### Network Architecture
```
Input Layer (78 features) → Hidden Layers (256, 128, 64) → Output Layer (Action Space)
```

#### Training Process (`rl_ids/modeling/train.py`)
1. **Environment Interaction**: Agent observes network states
2. **Action Selection**: Choose detection strategy based on Q-values
3. **Experience Collection**: Store (state, action, reward, next_state) tuples
4. **Batch Learning**: Update Q-network using sampled experiences
5. **Target Network Update**: Periodic synchronization for stability

### 4. Environment Layer (`rl_ids/environments/ids_env.py`)

#### IDS Gym Environment
- **State Space**: 78-dimensional feature vectors
- **Action Space**: Detection decisions (normal/attack classification)
- **Reward Function**: Based on detection accuracy and false positive rates
- **Episode Structure**: Configurable episode length for training

### 5. API Layer (`api/`)

#### FastAPI Service (`api/main.py`)
- **Endpoints**:
  - `/`: Service information
  - `/health`: Health check
  - `/model/info`: Model metadata
  - `/predict`: Single prediction
  - `/predict/batch`: Batch predictions

#### Models and Validation (`api/models.py`)
- Pydantic models for request/response validation
- Type safety and automatic documentation
- Error handling and status codes

#### Client Library (`api/client.py`)
- Python client for API interaction
- Asynchronous support
- Built-in error handling and retries

## Data Flow Architecture

### Training Phase
```
Historical Data (CICIDS2017) → Feature Extraction → Environment → DQN Agent → Model Training → Saved Model
```

### Inference Phase
```
Live Network Traffic → Feature Extraction → Trained Model → Prediction → Alert/Response
```

### API Integration
```
External Request → API Validation → Model Inference → Response → Client Application
```

## Design Principles

### 1. Modularity
- **Separation of Concerns**: Each component has a single responsibility
- **Loose Coupling**: Components interact through well-defined interfaces
- **Plugin Architecture**: Easy to extend with new features

### 2. Scalability
- **Horizontal Scaling**: API can be deployed across multiple instances
- **Asynchronous Processing**: Non-blocking operations for better throughput
- **Batch Processing**: Efficient handling of multiple requests

### 3. Adaptability
- **Continuous Learning**: Model can adapt to new threat patterns
- **Configuration-Driven**: Behavior controlled through configuration files
- **Environment Flexibility**: Works across different network environments

### 4. Reliability
- **Error Handling**: Comprehensive error handling and logging
- **Graceful Degradation**: System continues operating during partial failures
- **Health Monitoring**: Built-in health checks and status reporting

## Configuration Management

### Configuration Files
- `rl_ids/config.py`: Core system configuration
- `api/config.py`: API-specific settings
- `.env`: Environment variables for deployment

### Key Configuration Areas
- **Model Parameters**: Network architecture, training hyperparameters
- **Environment Settings**: Reward functions, episode configuration
- **API Configuration**: Server settings, security parameters
- **Monitoring Settings**: Logging levels, capture interfaces

## Security Considerations

### Network Access
- **Privilege Management**: Minimal required permissions
- **Interface Isolation**: Secure packet capture
- **Data Privacy**: No sensitive data logging

### API Security
- **Input Validation**: Strict type checking and sanitization
- **Rate Limiting**: Protection against abuse
- **Error Handling**: No sensitive information leakage

## Performance Characteristics

### Training Performance
- **GPU Acceleration**: CUDA support for faster training
- **Memory Efficiency**: Optimized data structures and batch processing
- **Convergence Speed**: Typically 200-500 episodes for convergence

### Inference Performance
- **Real-time Processing**: Sub-second response times
- **Throughput**: Handles thousands of predictions per second
- **Resource Usage**: Optimized for production deployment

## Deployment Architecture

### Development Deployment
```
Local Machine → Python Virtual Environment → Direct Execution
```

### Production Deployment
```
Load Balancer → API Instances → Model Inference → Database/Logging
```

### Monitoring Deployment
```
Network Interface → Packet Capture → Feature Extraction → Real-time Detection
```

## Extension Points

### Adding New Features
1. Extend feature extraction in `make_dataset.py`
2. Update model input dimensions
3. Retrain with enhanced feature set

### New Detection Algorithms
1. Implement new agent in `rl_ids/agents/`
2. Create corresponding environment
3. Update training pipeline

### API Extensions
1. Add new endpoints in `api/main.py`
2. Define request/response models
3. Update client library

## Dependencies and Libraries

### Core Dependencies
- **PyTorch**: Deep learning framework
- **Gymnasium**: RL environment interface
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **FastAPI**: Web API framework

### Monitoring Dependencies
- **Scapy**: Packet capture and analysis
- **Psutil**: System and network utilities
- **Loguru**: Advanced logging

### Development Dependencies
- **Pytest**: Testing framework
- **Ruff**: Code formatting and linting
- **MkDocs**: Documentation generation

## Future Architecture Considerations

### Planned Enhancements
- **Distributed Training**: Multi-node training support
- **Stream Processing**: Kafka/Redis integration
- **Model Versioning**: MLflow integration
- **Container Orchestration**: Kubernetes deployment

### Scalability Improvements
- **Microservices**: Service decomposition
- **Event-Driven Architecture**: Asynchronous event processing
- **Caching Layer**: Redis for improved performance
- **Database Integration**: Persistent storage for alerts and metrics