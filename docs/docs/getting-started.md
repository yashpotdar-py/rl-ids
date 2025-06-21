# Getting Started Guide

## Overview

This guide will walk you through setting up the RL-IDS Adaptive System, from installation to running your first intrusion detection model. By the end of this tutorial, you'll have a fully functional system capable of detecting network intrusions using reinforcement learning.

## Prerequisites

### System Requirements
- **Python 3.13+** (required for optimal performance)
- **8GB+ RAM** (16GB recommended for large datasets)
- **2GB+ disk space** for datasets and models
- **CUDA-compatible GPU** (optional, but recommended for faster training)

### Knowledge Prerequisites
- Basic understanding of Python programming
- Familiarity with command-line interfaces
- Basic concepts of machine learning (helpful but not required)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rl_ids
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv rl_ids_env
source rl_ids_env/bin/activate  # On Windows: rl_ids_env\Scripts\activate

# Using conda (alternative)
conda create -n rl_ids python=3.13
conda activate rl_ids
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
# Test the installation
python -c "import rl_ids; print('RL-IDS installed successfully!')"

# Check available commands
python -m rl_ids.modeling.train --help
```

## Data Preparation

### 1. Obtain CICIDS2017 Dataset

The system is designed to work with the CICIDS2017 dataset. You can download it from:
- [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html)

### 2. Dataset Structure

Place the downloaded CSV files in the `data/raw/` directory:

```
data/raw/
├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Monday-WorkingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
└── Wednesday-workingHours.pcap_ISCX.csv
```

### 3. Process the Data

```bash
# Run the data preprocessing pipeline
python -m rl_ids.make_dataset

# This will create processed files in data/processed/
# - train.csv (training set)
# - val.csv (validation set)  
# - test.csv (test set)
```

**Expected Output:**
```
🗂️  Starting data loading from directory: data/raw
📄 Found 8 CSV files to process
🔗 Concatenating data frames...
🧹 Starting data preprocessing...
✅ Data loading and preprocessing complete! Final shape: (2830743, 79)
```

## First Training Run

### 1. Basic Training

Start with a simple training run to verify everything works:

```bash
python -m rl_ids.modeling.train \
    --episodes 50 \
    --lr 0.001 \
    --batch_size 32
```

### 2. Advanced Training

For better performance, use advanced features:

```bash
python -m rl_ids.modeling.train \
    --episodes 250 \
    --lr 0.0001 \
    --batch_size 64 \
    --double_dqn \
    --dueling \
    --prioritized_replay \
    --curriculum_learning
```

**Training Parameters Explained:**
- `--episodes`: Number of training episodes (more = better performance)
- `--lr`: Learning rate (lower = more stable learning)
- `--batch_size`: Training batch size (higher = more stable gradients)
- `--double_dqn`: Reduces overestimation bias
- `--dueling`: Separates value and advantage estimation
- `--prioritized_replay`: Focuses on important experiences
- `--curriculum_learning`: Progressive difficulty increase

### 3. Monitor Training Progress

The training process will display real-time metrics:

```
🚀 Starting Enhanced DQN training for IDS Detection
===============================================
Episode 10/250: Reward=45.2, Accuracy=0.834, Confidence=0.892, Epsilon=0.85
Episode 20/250: Reward=52.1, Accuracy=0.867, Confidence=0.901, Epsilon=0.72
...
```

## Model Evaluation

### 1. Evaluate Trained Model

```bash
python -m rl_ids.modeling.evaluate \
    --model_path models/dqn_model_best.pt \
    --test_episodes 10
```

### 2. Generate Detailed Reports

The evaluation will create comprehensive reports in `reports/`:

- `evaluation_summary_enhanced.csv` - Overall performance metrics
- `evaluation_detailed_predictions_enhanced.csv` - Per-prediction results
- `evaluation_classification_report.csv` - Class-wise performance
- `figures/` - Visualization plots

### 3. View Results

Check the generated visualizations:

```bash
# View the results in the reports/figures/ directory
ls reports/figures/
# evaluation_overview.png
# class_analysis.png
# error_analysis.png
# confusion_matrix.png
```

## API Service Setup

### 1. Start the API Server

```bash
python -m api.main
```

The API will start on `http://localhost:8000` with:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **Health Check**: `http://localhost:8000/health`

### 2. Test the API

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}'

# Using the provided client
python -m api.client
```

### 3. API Response Format

```json
{
  "prediction": 1,
  "confidence": 0.95,
  "class_probabilities": [0.05, 0.95],
  "predicted_class": "Attack",
  "is_attack": true,
  "processing_time_ms": 2.3,
  "timestamp": "2025-06-21T10:30:00"
}
```

## Docker Deployment

### 1. Build Docker Image

```bash
docker build -t rl-ids-api .
```

### 2. Run Container

```bash
docker run -p 8000:8000 rl-ids-api
```

### 3. Use Docker Compose

```bash
# Basic deployment
docker-compose up -d

# With monitoring
docker-compose --profile monitoring up -d
```

## Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# Server Configuration
RLIDS_HOST=0.0.0.0
RLIDS_PORT=8000
RLIDS_DEBUG=false

# Model Configuration
RLIDS_MODEL_PATH=models/dqn_model_best.pt
RLIDS_MAX_BATCH_SIZE=100

# Logging
RLIDS_LOG_LEVEL=INFO
```

### Advanced Configuration

Modify configuration in `rl_ids/config.py` for:
- Data paths
- Model directories
- Logging settings
- Performance tuning

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python -m rl_ids.modeling.train --batch_size 16

# Or force CPU usage
CUDA_VISIBLE_DEVICES="" python -m rl_ids.modeling.train
```

**2. Data Loading Errors**
```bash
# Check data directory structure
ls data/raw/
# Ensure all 8 CICIDS2017 CSV files are present
```

**3. Model Loading Errors**
```bash
# Check if model file exists
ls models/
# Retrain if necessary
python -m rl_ids.modeling.train
```

### Performance Optimization

**For Training:**
- Use GPU if available (`CUDA_VISIBLE_DEVICES=0`)
- Increase batch size (32, 64, 128)
- Use multiple workers for data loading
- Enable mixed precision training

**For API:**
- Use multiple workers (`uvicorn --workers 4`)
- Enable async processing
- Configure proper batch sizes
- Use Redis for caching (in production)

## Next Steps

Now that you have a working system:

1. **Explore Advanced Features**: Check out [advanced training techniques](tutorials/advanced_training.md)
2. **API Integration**: Learn about [API usage patterns](tutorials/api_usage.md)
3. **Production Deployment**: Follow the [deployment guide](tutorials/deployment.md)
4. **Custom Models**: Create [custom environments](tutorials/custom_environments.md)
5. **Monitoring**: Set up [comprehensive monitoring](tutorials/monitoring.md)

## Getting Help

- **Documentation**: Check the [API reference](api/index.md) and [module docs](modules/index.md)
- **Examples**: Browse the [tutorials](tutorials/index.md) section
- **Issues**: Create an issue on the project repository
- **Community**: Join discussions and share experiences

## Quick Reference

### Essential Commands

```bash
# Data processing
python -m rl_ids.make_dataset

# Training
python -m rl_ids.modeling.train --episodes 100

# Evaluation  
python -m rl_ids.modeling.evaluate

# API server
python -m api.main

# Health check
curl http://localhost:8000/health
```

### Key File Locations

- **Models**: `models/dqn_model_best.pt`
- **Data**: `data/processed/{train,val,test}.csv`
- **Reports**: `reports/evaluation_*.csv`
- **Figures**: `reports/figures/*.png`
- **Logs**: Console output with loguru formatting
