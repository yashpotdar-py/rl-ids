# Installation

This guide covers the installation and setup of the RL-IDS (Reinforcement Learning Intrusion Detection System).

## System Requirements

### Minimum Requirements
- **Operating System**: Linux (Ubuntu 18.04+), macOS (10.15+), or Windows 10/11
- **Python**: 3.13.0+ (recommended: 3.13.0)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 10GB free space for datasets and models
- **Network**: Administrative privileges for packet capture

### Hardware Recommendations
- **CPU**: Multi-core processor (8+ cores recommended for training)
- **GPU**: NVIDIA GPU with CUDA support (optional, for faster training)
- **Network Interface**: Ethernet adapter for network monitoring

## Installation Methods

### Method 1: pip Installation (Recommended)

```bash
# Create virtual environment
python -m venv rl_ids_env
source rl_ids_env/bin/activate  # Linux/macOS
# or
rl_ids_env\Scripts\activate     # Windows

# Install RL-IDS
pip install rl_ids

# Verify installation
python -c "import rl_ids; print('RL-IDS installed successfully')"
```

### Method 2: Development Installation

```bash
# Clone the repository
git clone https://github.com/yashpotdar-py/rl-ids.git
cd rl-ids

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install in development mode
pip install -e .

# Install additional dependencies
pip install -r requirements.txt
```

### Method 3: Docker Installation (Coming Soon)

```bash
# Pull the Docker image
docker pull yashpotdar/rl-ids:latest

# Run the container
docker run -p 8000:8000 yashpotdar/rl-ids:latest
```

## Post-Installation Setup

### 1. Verify Network Permissions

For network monitoring capabilities, ensure you have the necessary permissions:

```bash
# Linux: Check if user can capture packets
sudo setcap cap_net_raw,cap_net_admin=eip $(which python)

# Or run with sudo (not recommended for production)
sudo python network_monitor.py
```

### 2. Download Pre-trained Models

```bash
# Download the latest DQN model (if available)
python -c "
from rl_ids.modeling.train import download_pretrained_model
download_pretrained_model()
"
```

### 3. Configure Environment

Create a `.env` file in your project directory:

```bash
# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=true

# Model Configuration
MODEL_PATH=models/dqn_model_best.pt
FEATURE_SCALER_PATH=models/feature_scaler.pkl

# Monitoring Configuration
CAPTURE_INTERFACE=eth0
LOG_LEVEL=INFO
LOG_FILE=logs/rl_ids.log
```

### 4. Test Installation

Run the test suite to verify everything is working:

```bash
# Basic functionality test
pytest tests/test_api.py -v

# Complete test suite (if available)
python -m pytest tests/ -v
```

## Configuration

### Network Interface Configuration

Identify available network interfaces:

```bash
# List network interfaces
python -c "
import psutil
interfaces = psutil.net_if_addrs()
for interface, addresses in interfaces.items():
    print(f'Interface: {interface}')
    for addr in addresses:
        if addr.family.name == 'AF_INET':
            print(f'  IP: {addr.address}')
"
```

### GPU Configuration (Optional)

If you have an NVIDIA GPU and want to use CUDA acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
"
```

## Troubleshooting

### Common Issues

#### Permission Denied Error
```bash
# Error: [Errno 1] Operation not permitted
# Solution: Run with appropriate permissions or set capabilities
sudo setcap cap_net_raw,cap_net_admin=eip $(which python)
```

#### Import Error
```bash
# Error: ModuleNotFoundError: No module named 'rl_ids'
# Solution: Ensure virtual environment is activated and package is installed
source venv/bin/activate
pip install -e .
```

#### CUDA Out of Memory
```bash
# Error: RuntimeError: CUDA out of memory
# Solution: Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### Network Interface Not Found
```bash
# Error: Interface 'eth0' not found
# Solution: List available interfaces and update configuration
python -c "import psutil; print(list(psutil.net_if_addrs().keys()))"
```

### Logging Configuration

Enable detailed logging for troubleshooting:

```python
# Add to your Python script
import logging
from loguru import logger

# Configure logging level
logger.add("logs/debug.log", level="DEBUG", rotation="10 MB")
```

### Performance Optimization

#### For Training
- Use GPU acceleration when available
- Increase batch size if memory permits
- Use multiple CPU cores for data preprocessing

#### For Monitoring
- Adjust capture buffer size based on network traffic
- Use appropriate logging levels to reduce overhead
- Consider using background processes for real-time monitoring

## Next Steps

After successful installation:

1. **Read the [Network Monitoring Guide](network-monitoring.md)** to start monitoring network traffic
2. **Explore the [API Documentation](../api/index.md)** to integrate with existing systems
3. **Check the [Agent Configuration](../modules/agents.md)** to customize the RL model
4. **Review [Development Setup](../development/contributing.md)** if you plan to contribute

## Getting Help

- **Documentation**: Browse the complete documentation in this site
- **Issues**: Report bugs on the [GitHub Issues](https://github.com/yashpotdar-py/rl-ids/issues) page
- **Discussions**: Join community discussions on the repository

## Version Information

- **Current Version**: 1.2.0
- **Python Compatibility**: 3.10.0+
- **Last Updated**: 2025