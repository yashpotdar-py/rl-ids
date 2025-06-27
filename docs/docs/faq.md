# Frequently Asked Questions

This FAQ addresses common questions about the RL-IDS (Reinforcement Learning Intrusion Detection System).

## General Questions

### What is RL-IDS?

RL-IDS is a reinforcement learning-driven adaptive intrusion detection system that uses Deep Q-Network (DQN) algorithms to detect network intrusions in real-time. It combines traditional network monitoring with modern machine learning techniques to provide intelligent threat detection.

### What makes RL-IDS different from traditional IDS?

- **Adaptive Learning**: Continuously learns from new data and adapts to evolving threats
- **Reinforcement Learning**: Uses reward-based learning to improve detection accuracy
- **Real-time Processing**: Provides immediate threat detection and response
- **Feature-rich Analysis**: Analyzes 78 different network flow features
- **API Integration**: Easy integration with existing security infrastructure

### What types of attacks can RL-IDS detect?

Based on the CICIDS2017 dataset, RL-IDS can detect:
- **DDoS Attacks**: Distributed Denial of Service
- **Port Scan**: Network reconnaissance activities
- **Web Attacks**: SQL injection, XSS, and other web-based attacks
- **Infiltration**: Advanced persistent threats
- **Brute Force**: Password and authentication attacks
- **Botnet**: Command and control communications

## Installation and Setup

### What are the system requirements?

**Minimum Requirements:**
- Python 3.13.0+
- 8GB RAM (16GB recommended)
- 10GB free disk space
- Network interface with administrative privileges

**Recommended:**
- Multi-core CPU (8+ cores for training)
- NVIDIA GPU with CUDA support
- Fast SSD storage

### Do I need root/administrator privileges?

For network monitoring features, you need:
- **Linux**: Either root privileges or set capabilities: `sudo setcap cap_net_raw,cap_net_admin=eip $(which python)`
- **Windows**: Run as administrator
- **macOS**: Use `sudo` or configure appropriate permissions

For API-only usage, no special privileges are required.

### Can I run RL-IDS without a pre-trained model?

Yes, but with limitations:
- You can train a new model using the provided training scripts
- Training requires the CICIDS2017 dataset (or similar labeled data)
- Initial training may take several hours depending on hardware
- Pre-trained models provide immediate detection capabilities

## Usage and Configuration

### How do I configure network monitoring?

1. **Identify Network Interface**:
   ```python
   import psutil
   print(list(psutil.net_if_addrs().keys()))
   ```

2. **Update Configuration**:
   ```python
   # In network_monitor.py or configuration file
   INTERFACE = "eth0"  # Replace with your interface
   ```

3. **Set Permissions** (Linux):
   ```bash
   sudo setcap cap_net_raw,cap_net_admin=eip $(which python)
   ```

### How do I integrate RL-IDS with my existing security tools?

**API Integration:**
```python
from api.client import RLIDSClient

client = RLIDSClient("http://localhost:8000")
result = client.predict(network_features)
```

**Webhook Integration:**
- Configure webhooks in your security tools
- Send network data to RL-IDS API endpoints
- Process responses for alerts and actions

**Log Integration:**
- Configure RL-IDS to output to standard security log formats
- Integrate with SIEM systems through log forwarding

### Can I customize the detection model?

Yes, several customization options are available:

**Model Parameters:**
- Adjust neural network architecture in `rl_ids/agents/dqn_agent.py`
- Modify training hyperparameters in `rl_ids/config.py`
- Change reward functions in the environment

**Feature Engineering:**
- Add new features to `rl_ids/make_dataset.py`
- Modify existing feature calculations
- Implement custom preprocessing pipelines

**Training Data:**
- Use your own labeled dataset
- Combine multiple datasets
- Implement active learning strategies

## Performance and Scalability

### What is the expected performance?

**Detection Speed:**
- Single prediction: < 100ms
- Batch predictions: > 1000 predictions/second
- Real-time monitoring: Handles typical enterprise network loads

**Accuracy:**
- Trained on CICIDS2017: > 95% accuracy
- False positive rate: < 5%
- Performance may vary with different network environments

### How can I improve performance?

**Hardware Optimizations:**
- Use GPU acceleration for training and large-scale inference
- Increase RAM for larger batch processing
- Use SSD storage for faster data access

**Software Optimizations:**
- Adjust batch sizes based on available memory
- Use multiple worker processes for API deployment
- Implement caching for frequently accessed data

**Network Optimizations:**
- Optimize packet capture buffer sizes
- Use appropriate network interface configurations
- Consider distributed deployment for high-traffic environments

### Can RL-IDS scale horizontally?

Yes, through several approaches:

**API Scaling:**
- Deploy multiple API instances behind a load balancer
- Use container orchestration (Docker, Kubernetes)
- Implement distributed caching (Redis)

**Data Processing:**
- Distribute packet capture across multiple interfaces
- Use message queues for processing pipelines
- Implement microservices architecture

## Troubleshooting

### "Permission denied" errors during network monitoring

**Solution:**
```bash
# Linux - Set capabilities
sudo setcap cap_net_raw,cap_net_admin=eip $(which python)

# Or run with sudo (not recommended for production)
sudo python network_monitor.py

# Windows - Run as administrator
# Right-click Command Prompt → "Run as administrator"
```

### "ModuleNotFoundError" when importing rl_ids

**Common Causes:**
1. Virtual environment not activated
2. Package not installed properly
3. Python path issues

**Solutions:**
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Reinstall package
pip install -e .

# Check installation
python -c "import rl_ids; print('Success')"
```

### High memory usage during training

**Solutions:**
1. **Reduce Batch Size:**
   ```python
   # In training configuration
   BATCH_SIZE = 32  # Reduce from default
   ```

2. **Enable Gradient Checkpointing:**
   ```python
   # In model configuration
   USE_CHECKPOINT = True
   ```

3. **Use CPU Training:**
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

### Model not loading or giving errors

**Check Model File:**
```python
import torch
try:
    model = torch.load("models/dqn_model_best.pt")
    print("Model loaded successfully")
except Exception as e:
    print(f"Model loading error: {e}")
```

**Regenerate Model:**
```bash
# Retrain model
python rl_ids/modeling/train.py

# Or download pre-trained model (if available)
python -c "from rl_ids.modeling.train import download_pretrained_model; download_pretrained_model()"
```

### API connection issues

**Check API Status:**
```bash
# Test API connectivity
curl http://localhost:8000/health

# Check if API is running
ps aux | grep "uvicorn\|python.*api"
```

**Common Solutions:**
1. Ensure API server is running
2. Check firewall settings
3. Verify correct host/port configuration
4. Test with different client (curl, browser)

## Development and Contribution

### How can I contribute to RL-IDS?

1. **Fork the Repository**: Create your own fork on GitHub
2. **Set Up Development Environment**: Follow the development setup guide
3. **Make Changes**: Implement features or fix bugs
4. **Write Tests**: Ensure your changes are tested
5. **Submit Pull Request**: Follow the contribution guidelines

### How do I add new attack detection capabilities?

1. **Data Collection**: Gather labeled examples of the new attack type
2. **Feature Analysis**: Identify distinguishing features
3. **Model Training**: Retrain with expanded dataset
4. **Validation**: Test detection accuracy
5. **Integration**: Update classification labels and API responses

### How do I extend the API?

1. **Add Endpoints**: Define new routes in `api/main.py`
2. **Create Models**: Add Pydantic models in `api/models.py`
3. **Implement Logic**: Add business logic in `api/services.py`
4. **Update Client**: Extend the Python client library
5. **Document**: Update API documentation

## Advanced Usage

### Can I use RL-IDS with custom datasets?

Yes, follow these steps:

1. **Format Data**: Ensure data matches CICIDS2017 feature format (78 features)
2. **Update Labels**: Map your attack types to the expected categories
3. **Preprocessing**: Apply the same preprocessing pipeline
4. **Training**: Retrain the model with your dataset
5. **Validation**: Test performance on representative data

### How do I implement custom reward functions?

Modify the reward function in the IDS environment:

```python
# In rl_ids/environments/ids_env.py
def _calculate_reward(self, predicted_class, actual_class):
    # Custom reward logic
    if predicted_class == actual_class:
        if actual_class == 'BENIGN':
            return 1.0  # Correct benign classification
        else:
            return 2.0  # Correct attack detection (higher reward)
    else:
        if actual_class == 'BENIGN':
            return -2.0  # False positive (high penalty)
        else:
            return -1.0  # False negative
```

### Can I deploy RL-IDS in a distributed environment?

Yes, consider these approaches:

**Microservices Deployment:**
- Separate services for data collection, processing, and detection
- Use message queues (RabbitMQ, Kafka) for communication
- Implement service discovery and load balancing

**Container Orchestration:**
```yaml
# docker-compose.yml example
version: '3.8'
services:
  rl-ids-api:
    image: rl-ids:latest
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/dqn_model_best.pt
  
  rl-ids-monitor:
    image: rl-ids:latest
    command: python network_monitor.py
    network_mode: host
    privileged: true
```

## Getting Help

### Where can I find more information?

- **Documentation**: Complete documentation is available in this site
- **GitHub Repository**: Source code and issue tracking
- **API Reference**: Interactive API documentation at `/docs` endpoint
- **Code Examples**: Sample implementations in the repository

### How do I report bugs or request features?

1. **Check Existing Issues**: Search GitHub issues for similar problems
2. **Create Detailed Report**: Include error messages, environment details, and steps to reproduce
3. **Provide Context**: Explain your use case and expected behavior
4. **Follow Up**: Respond to questions and provide additional information

### How do I get support for production deployment?

For production deployments:
1. **Review Best Practices**: Follow the deployment and security guidelines
2. **Performance Testing**: Conduct thorough testing in your environment
3. **Monitoring Setup**: Implement comprehensive logging and monitoring
4. **Backup Strategy**: Ensure model and configuration backup procedures
5. **Update Plan**: Establish procedures for updates and maintenance

Remember to never include sensitive network data or security configurations in support requests.