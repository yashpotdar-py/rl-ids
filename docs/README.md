# RL-IDS Documentation

Welcome to the comprehensive documentation for the **RL-IDS Adaptive System** - a state-of-the-art reinforcement learning-based intrusion detection system that adapts to evolving cybersecurity threats.

## 📚 Documentation Overview

This documentation provides complete coverage of the RL-IDS system, from quick start guides to advanced deployment strategies. Whether you're a researcher, security analyst, or DevOps engineer, you'll find the information you need to effectively use and deploy the system.

## 🚀 Quick Navigation

### For New Users
- **[Getting Started Guide](docs/getting-started.md)** - Complete setup and first training
- **[Project Overview](docs/index.md)** - Architecture and key features
- **[FAQ & Troubleshooting](docs/faq.md)** - Common issues and solutions

### For Developers
- **[API Reference](docs/api/index.md)** - REST API documentation
- **[Module Reference](docs/modules/index.md)** - Detailed code documentation
- **[Configuration Guide](docs/modules/config.md)** - System configuration

### For Advanced Users
- **[Advanced Training](docs/tutorials/advanced_training.md)** - Hyperparameter optimization and curriculum learning
- **[API Usage Patterns](docs/tutorials/api_usage.md)** - Integration and monitoring
- **[Deployment Guide](docs/tutorials/deployment.md)** - Production deployment strategies

## 📋 Table of Contents

### 1. Getting Started
| Document | Description | Audience |
|----------|-------------|----------|
| [Project Overview](docs/index.md) | Architecture, features, and quick start | All users |
| [Getting Started Guide](docs/getting-started.md) | Step-by-step setup and usage | Beginners |
| [FAQ & Troubleshooting](docs/faq.md) | Common issues and solutions | All users |

### 2. API Documentation
| Document | Description | Audience |
|----------|-------------|----------|
| [API Reference](docs/api/index.md) | Complete REST API documentation | Developers |
| [API Usage Tutorial](docs/tutorials/api_usage.md) | Advanced integration patterns | Developers |

### 3. Module Reference
| Document | Description | Audience |
|----------|-------------|----------|
| [Module Overview](docs/modules/index.md) | Architecture and dependencies | Developers |
| [DQN Agent](docs/modules/agents.md) | Reinforcement learning agent | ML Engineers |
| [Environment](docs/modules/environments.md) | Custom Gym environment | ML Engineers |
| [Training & Evaluation](docs/modules/modeling.md) | Training and evaluation pipelines | ML Engineers |
| [Data Processing](docs/modules/make_dataset.md) | Data preprocessing and feature engineering | Data Scientists |
| [Visualization](docs/modules/plots.md) | Plotting and analysis tools | Data Scientists |
| [Configuration](docs/modules/config.md) | System configuration management | DevOps |

### 4. Tutorials & Guides
| Document | Description | Audience |
|----------|-------------|----------|
| [Tutorial Overview](docs/tutorials/index.md) | Available tutorials and learning paths | All users |
| [Advanced Training](docs/tutorials/advanced_training.md) | Hyperparameter optimization, curriculum learning | ML Engineers |
| [API Usage Patterns](docs/tutorials/api_usage.md) | Integration, monitoring, scaling | Developers |

## 🎯 Learning Paths

### Path 1: Research & Experimentation
1. Start with [Project Overview](docs/index.md) to understand the system
2. Follow [Getting Started Guide](docs/getting-started.md) for basic setup
3. Dive into [DQN Agent](docs/modules/agents.md) and [Environment](docs/modules/environments.md) modules
4. Explore [Advanced Training](docs/tutorials/advanced_training.md) techniques
5. Use [Visualization](docs/modules/plots.md) tools for analysis

### Path 2: Production Deployment
1. Begin with [Getting Started Guide](docs/getting-started.md) for system setup
2. Study [API Reference](docs/api/index.md) for service capabilities
3. Follow [API Usage Tutorial](docs/tutorials/api_usage.md) for integration patterns
4. Implement production deployment strategies
5. Set up comprehensive monitoring and alerting

### Path 3: Data Science & Analysis
1. Review [Project Overview](docs/index.md) for context
2. Understand [Data Processing](docs/modules/make_dataset.md) pipeline
3. Learn [Training & Evaluation](docs/modules/modeling.md) workflows
4. Master [Visualization](docs/modules/plots.md) capabilities
5. Apply [Advanced Training](docs/tutorials/advanced_training.md) techniques

## 🛠️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RL-IDS System                           │
├─────────────────────┬─────────────────────┬─────────────────────┤
│   Data Pipeline     │   ML Pipeline       │   API Service       │
│                     │                     │                     │
│ • Data Ingestion    │ • DQN Agent         │ • FastAPI Service   │
│ • Preprocessing     │ • Custom Gym Env    │ • Real-time Predict │
│ • Feature Eng.      │ • Training Pipeline │ • Batch Processing  │
│ • Data Balancing    │ • Evaluation Tools  │ • Monitoring        │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### Core Components
- **[DQN Agent](docs/modules/agents.md)** - Deep Q-Network with advanced features
- **[IDS Environment](docs/modules/environments.md)** - Custom Gymnasium environment
- **[API Service](docs/api/index.md)** - Production-ready FastAPI service
- **[Data Pipeline](docs/modules/make_dataset.md)** - Comprehensive preprocessing
- **[Visualization](docs/modules/plots.md)** - Advanced plotting and analysis

## 📊 Key Features

### Machine Learning
- **Advanced DQN Algorithm** with Double DQN, Dueling DQN, and Prioritized Experience Replay
- **Curriculum Learning** for improved training convergence
- **Multi-objective Optimization** balancing accuracy, precision, and recall
- **Automated Hyperparameter Tuning** with systematic search strategies

### Data Processing
- **CICIDS2017 Dataset Support** with automated preprocessing
- **Advanced Feature Engineering** with normalization and balancing
- **SMOTE and SMOTETomek** for handling class imbalance
- **Stratified Train/Validation/Test Splitting** for robust evaluation

### Production Capabilities
- **High-Performance API** with async processing and connection pooling
- **Comprehensive Monitoring** with Prometheus, ELK, and custom metrics
- **Scalable Deployment** with Docker, Kubernetes, and load balancing
- **Security Features** including authentication, rate limiting, and input validation

### Analysis & Visualization
- **Publication-Quality Plots** for training metrics and evaluation results
- **Interactive Dashboards** for real-time monitoring
- **Comprehensive Reports** with automated generation
- **Error Analysis** tools for model debugging and improvement

## 🔧 Installation & Setup

### Quick Start
```bash
# Clone repository
git clone https://github.com/yashpotdar-py/rl-ids.git
cd rl-ids

# Install dependencies
pip install -r requirements.txt

# Prepare data
python -m rl_ids.make_dataset

# Train model
python -m rl_ids.modeling.train

# Start API service
python run_api.py
```

### Docker Deployment
```bash
# Build and run with Docker
docker build -t rl-ids .
docker run -p 8000:8000 rl-ids

# Or use Docker Compose
docker-compose up -d
```

For detailed setup instructions, see the [Getting Started Guide](docs/getting-started.md).

## 📈 Performance Metrics

### Model Performance
- **Accuracy**: 94.5% on CICIDS2017 test set
- **Precision**: 93.2% for attack detection
- **Recall**: 94.8% for attack detection
- **F1-Score**: 94.0% weighted average

### API Performance
- **Latency**: <50ms for single predictions
- **Throughput**: 1000+ predictions/second
- **Scalability**: Horizontal scaling with load balancing
- **Availability**: 99.9% uptime with health monitoring

## 🔧 Building Documentation

Use [mkdocs](http://www.mkdocs.org/) to build and serve the documentation locally:

```bash
# Build documentation
mkdocs build

# Serve locally for development
mkdocs serve

# Build for production
mkdocs build --clean
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Code Contributions** - Submit pull requests with new features or bug fixes
2. **Documentation** - Help improve and expand documentation
3. **Testing** - Add test cases and improve coverage
4. **Feedback** - Report issues and suggest improvements

## 📞 Support & Community

### Getting Help
- **Documentation** - Start with this comprehensive documentation
- **FAQ** - Check [FAQ & Troubleshooting](docs/faq.md) for common issues
- **Issues** - Report bugs and request features on GitHub
- **Discussions** - Join community discussions for questions and ideas

## 🔗 Quick Links

### Essential Pages
- [🚀 Getting Started](docs/getting-started.md)
- [📖 API Reference](docs/api/index.md)
- [🧠 Model Training](docs/modules/modeling.md)
- [📊 Visualization](docs/modules/plots.md)

### Advanced Topics
- [🔍 Advanced Training](docs/tutorials/advanced_training.md)
- [🔧 API Integration](docs/tutorials/api_usage.md)
- [📚 Module Index](docs/modules/index.md)
- [❓ FAQ](docs/faq.md)

---

*Built with ❤️ for cybersecurity research and production deployments*
