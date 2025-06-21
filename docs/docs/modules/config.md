# Configuration Module

## Overview

The configuration module (`rl_ids.config`) provides centralized configuration management for the RL-IDS system. It defines project paths, data file locations, logging configuration, and environment variable handling.

## Key Features

- **Centralized Path Management** - All project paths defined in one place
- **Environment Variable Support** - Configuration via `.env` files
- **Advanced Logging Setup** - Integrated with loguru and tqdm
- **Debug Mode Support** - Enhanced logging for development
- **Cross-Platform Compatibility** - Path handling works on all operating systems

## Configuration Variables

### Project Structure

| Variable | Type | Description |
|----------|------|-------------|
| `PROJ_ROOT` | `Path` | Project root directory |
| `DATA_DIR` | `Path` | Main data directory |
| `RAW_DATA_DIR` | `Path` | Raw CICIDS2017 data location |
| `INTERIM_DATA_DIR` | `Path` | Intermediate processing results |
| `PROCESSED_DATA_DIR` | `Path` | Final processed data |
| `EXTERNAL_DATA_DIR` | `Path` | External datasets |

### Data Files

| Variable | Type | Description |
|----------|------|-------------|
| `PROCESSED_DATA_FILE` | `Path` | Main processed dataset |
| `NORMALISED_DATA_FILE` | `Path` | Normalized feature dataset |
| `BALANCED_DATA_FILE` | `Path` | Class-balanced dataset |
| `TRAIN_DATA_FILE` | `Path` | Training split |
| `VAL_DATA_FILE` | `Path` | Validation split |
| `TEST_DATA_FILE` | `Path` | Test split |

### Model Storage

| Variable | Type | Description |
|----------|------|-------------|
| `MODELS_DIR` | `Path` | Model storage directory |
| `EPISODES_DIR` | `Path` | Episode checkpoint storage |

### Reports and Figures

| Variable | Type | Description |
|----------|------|-------------|
| `REPORTS_DIR` | `Path` | Evaluation reports directory |
| `FIGURES_DIR` | `Path` | Generated plots and visualizations |

## Environment Variables

### Core Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `RLIDS_LOG_LEVEL` | `"INFO"` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `RLIDS_DEBUG` | `"false"` | Enable debug mode for detailed logging |

### Usage

Create a `.env` file in the project root:

```bash
# .env file
RLIDS_LOG_LEVEL=DEBUG
RLIDS_DEBUG=true

# Optional: Override default paths
RLIDS_DATA_PATH=/custom/data/path
RLIDS_MODELS_PATH=/custom/models/path
```

## Logging Configuration

The module automatically configures the loguru logger with:

- **Color-coded output** for different log levels
- **Timestamp formatting** with millisecond precision
- **Module/function/line information** for debugging
- **tqdm integration** for progress bar compatibility
- **Fallback configuration** when tqdm is not available

### Log Format

```
2024-01-15 14:30:25 | INFO     | rl_ids.agents:train:123 - Training completed successfully
2024-01-15 14:30:26 | DEBUG    | rl_ids.environments:step:45 - Action 2 resulted in reward 0.85
2024-01-15 14:30:27 | WARNING  | api.services:predict:78 - Model confidence below threshold: 0.65
```

## Usage Examples

### Basic Configuration Access

```python
from rl_ids.config import (
    PROJ_ROOT, MODELS_DIR, TRAIN_DATA_FILE,
    LOG_LEVEL, DEBUG_MODE
)
from loguru import logger

# Check if data file exists
if TRAIN_DATA_FILE.exists():
    logger.info(f"Training data found at: {TRAIN_DATA_FILE}")
else:
    logger.warning(f"Training data missing: {TRAIN_DATA_FILE}")

# Load model from standard location
model_path = MODELS_DIR / "dqn_model_best.pt"
if model_path.exists():
    logger.info(f"Loading model from: {model_path}")
```

### Custom Path Configuration

```python
from rl_ids.config import PROJ_ROOT
from pathlib import Path

# Create custom data paths
custom_data_dir = PROJ_ROOT / "experiments" / "run_001"
custom_data_dir.mkdir(parents=True, exist_ok=True)

# Save experiment results
results_file = custom_data_dir / "training_results.json"
logger.info(f"Saving results to: {results_file}")
```

### Environment-Specific Configuration

```python
import os
from rl_ids.config import DEBUG_MODE, LOG_LEVEL

# Conditional behavior based on debug mode
if DEBUG_MODE:
    logger.debug("Debug mode enabled - verbose logging active")
    # Enable additional debug features
    torch.autograd.set_detect_anomaly(True)
else:
    logger.info(f"Production mode - log level: {LOG_LEVEL}")
```

## Directory Structure Validation

The configuration module automatically validates the project structure:

```python
from rl_ids.config import DATA_DIR, MODELS_DIR, REPORTS_DIR

# Ensure required directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Directory validated: {directory}")
```

## Integration with Other Modules

### With Data Processing

```python
from rl_ids.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from rl_ids.make_dataset import DataPreprocessor

preprocessor = DataPreprocessor(
    input_dir=RAW_DATA_DIR,
    output_dir=PROCESSED_DATA_DIR
)
```

### With Model Training

```python
from rl_ids.config import TRAIN_DATA_FILE, MODELS_DIR
from rl_ids.modeling.train import main as train_model

# Training uses configured paths automatically
train_model()

# Models are saved to MODELS_DIR automatically
```

### With API Service

```python
from rl_ids.config import MODELS_DIR
from api.config import APISettings

# API configuration inherits from base config
api_settings = APISettings(
    model_path=MODELS_DIR / "dqn_model_final.pt"
)
```

## Best Practices

### 1. Path Handling

```python
# ✅ Correct - Use configured paths
from rl_ids.config import PROCESSED_DATA_DIR
data_file = PROCESSED_DATA_DIR / "my_data.csv"

# ❌ Avoid - Hardcoded paths
data_file = Path("data/processed/my_data.csv")
```

### 2. Environment Variables

```python
# ✅ Correct - Use environment variables for deployment
import os
from rl_ids.config import MODELS_DIR

# Allow override in production
model_path = os.getenv("RLIDS_MODEL_PATH", MODELS_DIR / "default_model.pt")

# ❌ Avoid - Hardcoded production values
model_path = "/production/models/model.pt"
```

### 3. Logging Integration

```python
# ✅ Correct - Use configured logger
from loguru import logger
from rl_ids.config import DEBUG_MODE

logger.info("Starting process...")
if DEBUG_MODE:
    logger.debug("Debug information...")

# ❌ Avoid - Creating new loggers
import logging
custom_logger = logging.getLogger(__name__)
```

## Troubleshooting

### Common Issues

1. **Path Not Found Errors**
   ```python
   # Check if required directories exist
   from rl_ids.config import DATA_DIR
   if not DATA_DIR.exists():
       DATA_DIR.mkdir(parents=True, exist_ok=True)
   ```

2. **Environment Variable Not Loaded**
   ```python
   # Ensure .env file is in project root
   from rl_ids.config import PROJ_ROOT
   env_file = PROJ_ROOT / ".env"
   print(f"Looking for .env at: {env_file}")
   print(f"Exists: {env_file.exists()}")
   ```

3. **Logging Issues**
   ```python
   # Check logging configuration
   from loguru import logger
   logger.info("Test log message")
   
   # Verify debug mode
   from rl_ids.config import DEBUG_MODE, LOG_LEVEL
   print(f"Debug: {DEBUG_MODE}, Level: {LOG_LEVEL}")
   ```

## See Also

- [Getting Started Guide](../getting-started.md) - Initial setup and configuration
- [Module Reference](index.md) - Overview of all modules
- [API Configuration](../api/index.md) - API-specific configuration
- [Training Configuration](modeling.md) - Training-specific settings
