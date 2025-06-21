# Tutorials

## Overview

This section provides step-by-step tutorials for common tasks with the RL-IDS system, from basic usage to advanced deployment scenarios.

## Available Tutorials

### Getting Started
- [Basic Training Tutorial](training.md) - Train your first DQN model
- [Data Preparation Guide](data_preparation.md) - Process CICIDS2017 dataset
- [Quick API Usage](api_usage.md) - Make predictions via REST API

### Advanced Topics
- [Custom Training Pipeline](advanced_training.md) - Advanced training techniques
- [Model Optimization](optimization.md) - Hyperparameter tuning and optimization
- [Production Deployment](deployment.md) - Deploy API service to production

### Integration
- [Real-time Monitoring](monitoring.md) - Integrate with monitoring systems
- [Custom Environments](custom_environments.md) - Create custom training environments
- [Batch Processing](batch_processing.md) - Process large datasets efficiently

## Prerequisites

Before starting the tutorials, ensure you have:

1. **Python 3.13+** installed
2. **CUDA-compatible GPU** (recommended for training)
3. **RL-IDS repository** cloned and set up
4. **Dependencies installed** via `pip install -r requirements.txt`
5. **Raw dataset** placed in `data/raw/` directory

## Quick Setup

```bash
# Clone and setup
git clone <repository-url>
cd rl_ids
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Prepare data
python -m rl_ids.make_dataset

# Verify setup
python -c "import rl_ids; print('Setup successful!')"
```

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Objective** - What you'll learn
2. **Prerequisites** - Required knowledge/setup
3. **Step-by-step Instructions** - Detailed walkthrough
4. **Code Examples** - Working code snippets
5. **Expected Output** - What to expect
6. **Troubleshooting** - Common issues and solutions
7. **Next Steps** - Related tutorials

## Support

If you encounter issues while following the tutorials:

1. Check the [FAQ](../faq.md) for common solutions
2. Review the [API Reference](../api/index.md) for detailed documentation
3. Examine the [Module Reference](../modules/index.md) for implementation details

## Contributing

Help improve these tutorials by:

- Reporting unclear instructions
- Suggesting additional examples
- Contributing new tutorial topics
- Fixing errors or typos
