# Basic Training Tutorial

## Objective

Learn how to train a DQN model for intrusion detection from scratch, understand the training process, and evaluate model performance.

## Prerequisites

- RL-IDS system installed and configured
- CICIDS2017 dataset prepared (see [Data Preparation Guide](data_preparation.md))
- Basic understanding of reinforcement learning concepts

## Step 1: Verify Data Preparation

First, ensure your data is properly prepared:

```bash
# Check if processed data exists
ls data/processed/
# Should show: train.csv, val.csv, test.csv

# If not, process the raw data
python -m rl_ids.make_dataset
```

**Expected Output:**
```
🗂️  Starting data loading from directory: data/raw
📄 Found 8 CSV files to process
🔗 Concatenating data frames...
🧹 Starting data preprocessing...
✅ Data loading and preprocessing complete! Final shape: (2830743, 79)
```

## Step 2: Basic Training

Start with a basic training session using default parameters:

```bash
# Basic training (recommended for first run)
python -m rl_ids.modeling.train \
    --num-episodes 100 \
    --batch-size 128 \
    --save-interval 25
```

**Expected Output:**
```
🚀 Starting Enhanced DQN training for IDS Detection
📂 Loading training data from data/processed/train.csv
🌍 Initializing training environment...
🤖 Initializing Enhanced DQN Agent...
🎯 Starting training loop...

Episode 1/100: Reward=1234, Accuracy=67.8%, Loss=0.0234, ε=0.995
Episode 25/100: Reward=2341, Accuracy=78.9%, Loss=0.0156, ε=0.876
Episode 50/100: Reward=3456, Accuracy=85.2%, Loss=0.0098, ε=0.678
Episode 75/100: Reward=4123, Accuracy=89.1%, Loss=0.0067, ε=0.543
Episode 100/100: Reward=4567, Accuracy=91.3%, Loss=0.0045, ε=0.432

✅ Training completed successfully!
📊 Best model saved to: models/dqn_model_best.pt
📊 Final model saved to: models/dqn_model_final.pt
```

## Step 3: Monitor Training Progress

During training, several files are created to track progress:

### Model Checkpoints
```bash
ls models/
# Output:
# dqn_model_best.pt      - Best performing model
# dqn_model_final.pt     - Final model after all episodes
# episodes/              - Directory with episode checkpoints
#   dqn_model_episode_25.pt
#   dqn_model_episode_50.pt
#   dqn_model_episode_75.pt
#   dqn_model_episode_100.pt
```

### Training Reports
```bash
ls reports/
# Output:
# training_metrics.csv   - Detailed training metrics
```

## Step 4: Evaluate Your Model

After training, evaluate the model performance:

```bash
# Evaluate the best model
python -m rl_ids.modeling.evaluate \
    --test-episodes 10 \
    --use-best-model
```

**Expected Output:**
```
🧪 Starting Enhanced DQN Agent Evaluation
🤖 Loading trained model from models/dqn_model_best.pt
🌍 Initializing test environment...

Episode 1/10: Accuracy=91.2%, Reward=4532, Steps=2234
Episode 5/10: Accuracy=92.1%, Reward=4651, Steps=2189
Episode 10/10: Accuracy=91.8%, Reward=4598, Steps=2201

📊 Evaluation Results:
  Overall Accuracy: 91.73% ± 0.34%
  Average Reward: 4596.3 ± 45.2
  Average Steps: 2208.1 ± 18.7
  High-Confidence Accuracy: 95.21%

📈 Reports generated in: reports/
📊 Visualizations saved in: reports/figures/
```

## Step 5: View Training Results

Examine the generated reports and visualizations:

### Key Files Generated
```bash
ls reports/
evaluation_summary_enhanced.csv
evaluation_episode_details_enhanced.csv
evaluation_detailed_predictions_enhanced.csv
evaluation_classification_report.csv

ls reports/figures/
evaluation_overview.png
class_analysis.png
error_analysis.png
enhanced_confusion_matrix.png
```

### Interpreting Results

1. **`evaluation_overview.png`**: Comprehensive performance overview
   - Episode performance trends
   - Confidence score distribution
   - Confusion matrix visualization
   - Class-wise accuracy metrics

2. **`class_analysis.png`**: Per-class performance analysis
   - Precision, recall, F1-score by class
   - Class distribution analysis
   - Confidence analysis by class

3. **`error_analysis.png`**: Error pattern analysis
   - Common misclassification patterns
   - Error distribution by confidence level
   - Most problematic class pairs

## Step 6: Advanced Training Configuration

Once you're comfortable with basic training, try advanced configurations:

```bash
# Advanced training with optimizations
python -m rl_ids.modeling.train \
    --num-episodes 500 \
    --lr 1e-4 \
    --batch-size 256 \
    --hidden-dims "1024,512,256,128" \
    --gamma 0.995 \
    --eps-decay 0.9995 \
    --memory-size 200000 \
    --double-dqn \
    --dueling \
    --curriculum-learning \
    --early-stopping-patience 50 \
    --validation-interval 10
```

### Parameter Explanation

| Parameter | Purpose | Recommended Values |
|-----------|---------|-------------------|
| `--num-episodes` | Training duration | 500-1000 for good models |
| `--lr` | Learning rate | 1e-4 to 1e-5 |
| `--batch-size` | Training batch size | 128-512 (based on GPU memory) |
| `--hidden-dims` | Network architecture | "512,256,128" to "2048,1024,512,256" |
| `--gamma` | Discount factor | 0.99-0.999 |
| `--eps-decay` | Exploration decay | 0.995-0.9999 |
| `--memory-size` | Replay buffer size | 50000-500000 |
| `--double-dqn` | Use Double DQN | Recommended |
| `--dueling` | Use Dueling DQN | Recommended |
| `--curriculum-learning` | Progressive difficulty | Recommended |

## Step 7: Compare Different Configurations

Train multiple models with different configurations:

```bash
# Experiment 1: Conservative learning
python -m rl_ids.modeling.train \
    --num-episodes 300 \
    --lr 5e-5 \
    --batch-size 128 \
    --models-dir models/conservative \
    --reports-dir reports/conservative

# Experiment 2: Aggressive learning  
python -m rl_ids.modeling.train \
    --num-episodes 300 \
    --lr 1e-3 \
    --batch-size 512 \
    --models-dir models/aggressive \
    --reports-dir reports/aggressive

# Evaluate both models
python -m rl_ids.modeling.evaluate \
    --model-path models/conservative/dqn_model_best.pt \
    --reports-dir reports/conservative_eval

python -m rl_ids.modeling.evaluate \
    --model-path models/aggressive/dqn_model_best.pt \
    --reports-dir reports/aggressive_eval
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size
python -m rl_ids.modeling.train --batch-size 64

# Or use CPU only
export CUDA_VISIBLE_DEVICES=""
python -m rl_ids.modeling.train
```

#### 2. Poor Training Performance
```bash
# Check data distribution
python -c "
import pandas as pd
df = pd.read_csv('data/processed/train.csv')
print(df['Label'].value_counts())
"

# If imbalanced, regenerate with balancing
python -m rl_ids.make_dataset --balance-classes
```

#### 3. Training Stuck/Not Improving
```bash
# Try curriculum learning
python -m rl_ids.modeling.train \
    --curriculum-learning \
    --curriculum-stages 3

# Or adjust learning rate
python -m rl_ids.modeling.train \
    --lr 1e-5 \
    --lr-scheduler cosine
```

#### 4. Model Not Saving
```bash
# Check permissions
ls -la models/
mkdir -p models episodes

# Verify disk space
df -h
```

### Expected Training Times

| Configuration | Episodes | Estimated Time |
|---------------|----------|----------------|
| Basic (CPU) | 100 | 30-60 minutes |
| Basic (GPU) | 100 | 10-20 minutes |
| Advanced (GPU) | 500 | 1-3 hours |
| Full Training (GPU) | 1000 | 3-6 hours |

## Next Steps

After completing basic training:

1. **[API Usage Tutorial](api_usage.md)** - Deploy your model as a REST API
2. **[Advanced Training](advanced_training.md)** - Learn hyperparameter optimization
3. **[Custom Environments](custom_environments.md)** - Create custom training scenarios
4. **[Production Deployment](deployment.md)** - Deploy to production systems

## Key Takeaways

- Start with basic training to understand the process
- Monitor training metrics to detect issues early
- Use evaluation reports to understand model performance
- Experiment with different configurations
- Save and compare multiple model versions
- Use GPU acceleration for faster training

The trained model will be ready for deployment via the API service or further optimization based on your specific requirements.
