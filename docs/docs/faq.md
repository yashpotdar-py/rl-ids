# FAQ & Troubleshooting

## Frequently Asked Questions

### General Questions

#### Q: What is RL-IDS and how does it work?

**A:** RL-IDS is a Reinforcement Learning-driven Adaptive Intrusion Detection System that uses Deep Q-Network (DQN) agents to detect network intrusions. Unlike traditional signature-based systems, RL-IDS learns to identify attack patterns through trial and error, making it adaptive to new and evolving threats.

The system works by:
1. **Training Phase**: The DQN agent learns from the CICIDS2017 dataset, receiving rewards for correct classifications
2. **Adaptation**: The agent continuously improves its detection strategy through reinforcement learning
3. **Inference**: Trained models classify real-time network traffic with confidence scores

#### Q: What makes RL-IDS different from traditional IDS systems?

**A:** Key differences include:

| Traditional IDS | RL-IDS |
|----------------|--------|
| Rule/signature-based detection | Learning-based adaptive detection |
| Manual rule updates | Automatic learning from new data |
| Binary decisions | Confidence-based predictions |
| Static thresholds | Dynamic decision boundaries |
| Limited to known attacks | Can detect novel attack patterns |

#### Q: What types of attacks can RL-IDS detect?

**A:** The system can detect 15 different attack types from the CICIDS2017 dataset:

1. **BENIGN** - Normal network traffic
2. **Web Attack – Brute Force** - Password brute force attacks
3. **Web Attack – XSS** - Cross-site scripting attacks
4. **Web Attack – SQL Injection** - Database injection attacks
5. **FTP-Patator** - FTP brute force attacks
6. **SSH-Patator** - SSH brute force attacks
7. **DoS slowloris** - Slow HTTP denial of service
8. **DoS Slowhttptest** - Slow HTTP test attacks
9. **DoS Hulk** - HTTP Unbearable Load King attacks
10. **DoS GoldenEye** - HTTP DoS attacks
11. **Heartbleed** - OpenSSL vulnerability exploitation
12. **Infiltration** - Network infiltration attempts
13. **PortScan** - Network port scanning
14. **DDoS** - Distributed denial of service
15. **Bot** - Botnet traffic

#### Q: How accurate is the RL-IDS system?

**A:** The system typically achieves:
- **Overall Accuracy**: >95% on CICIDS2017 test set
- **Precision**: >94% across all attack types
- **Recall**: >93% for most attack categories
- **F1-Score**: >94% weighted average
- **False Positive Rate**: <2% for normal traffic

Performance varies by attack type, with some rare attacks having lower recall due to limited training data.

---

### Installation & Setup

#### Q: What are the system requirements?

**A:** Minimum requirements:
- **Python**: 3.13+ (required)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 2GB for datasets and models
- **OS**: Linux, macOS, or Windows

Recommended for training:
- **GPU**: CUDA-compatible with 4GB+ VRAM
- **CPU**: Multi-core processor (8+ cores)
- **RAM**: 16GB+

#### Q: I'm getting CUDA out of memory errors during training. What should I do?

**A:** Try these solutions in order:

1. **Reduce batch size**:
   ```bash
   python -m rl_ids.modeling.train --batch_size 16
   ```

2. **Force CPU training**:
   ```bash
   CUDA_VISIBLE_DEVICES="" python -m rl_ids.modeling.train
   ```

3. **Reduce model size**:
   ```bash
   python -m rl_ids.modeling.train --hidden_dims "256,128"
   ```

4. **Use gradient checkpointing** (if available in future versions)

#### Q: The data preprocessing is failing. What's wrong?

**A:** Common causes and solutions:

1. **Missing raw data files**:
   ```bash
   ls data/raw/
   # Should show 8 CICIDS2017 CSV files
   ```

2. **Incorrect file names**: Ensure files match exactly:
   - `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
   - `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
   - etc.

3. **Corrupted files**: Re-download from the official CICIDS2017 source

4. **Insufficient disk space**: Ensure 5GB+ free space

5. **Memory issues**: Close other applications or increase swap space

#### Q: How do I verify my installation is correct?

**A:** Run these verification steps:

```bash
# 1. Check Python version
python --version  # Should be 3.13+

# 2. Test RL-IDS import
python -c "import rl_ids; print('✅ RL-IDS installed')"

# 3. Check dependencies
python -c "import torch; print('✅ PyTorch installed')"
python -c "import sklearn; print('✅ Scikit-learn installed')"

# 4. Test CLI commands
python -m rl_ids.modeling.train --help

# 5. Check GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

### Training & Model Issues

#### Q: Training is very slow. How can I speed it up?

**A:** Optimization strategies:

1. **Use GPU acceleration**:
   ```bash
   # Check GPU availability
   nvidia-smi
   # Train with CUDA
   python -m rl_ids.modeling.train --device cuda
   ```

2. **Increase batch size** (if GPU memory allows):
   ```bash
   python -m rl_ids.modeling.train --batch_size 128
   ```

3. **Reduce training episodes for testing**:
   ```bash
   python -m rl_ids.modeling.train --episodes 50
   ```

4. **Use curriculum learning**:
   ```bash
   python -m rl_ids.modeling.train --curriculum_learning
   ```

5. **Optimize data loading** (ensure data is on SSD)

#### Q: My model isn't converging. What can I do?

**A:** Training troubleshooting steps:

1. **Check learning rate**:
   ```bash
   # Try lower learning rate
   python -m rl_ids.modeling.train --lr 1e-5
   ```

2. **Enable advanced features**:
   ```bash
   python -m rl_ids.modeling.train --double_dqn --dueling --prioritized_replay
   ```

3. **Increase model capacity**:
   ```bash
   python -m rl_ids.modeling.train --hidden_dims "1024,512,256,128"
   ```

4. **Use curriculum learning**:
   ```bash
   python -m rl_ids.modeling.train --curriculum_learning --curriculum_stages 5
   ```

5. **Check data quality**: Ensure preprocessing completed successfully

#### Q: How do I know if my training is working properly?

**A:** Monitor these training indicators:

1. **Reward trends**: Should generally increase over episodes
2. **Accuracy**: Should improve and stabilize above 80%
3. **Epsilon decay**: Should decrease from 1.0 to minimum value
4. **Loss convergence**: Should decrease and stabilize

**Good training output example**:
```
Episode 50/250: Reward=65.2, Accuracy=0.834, Confidence=0.892, Epsilon=0.61
Episode 100/250: Reward=78.5, Accuracy=0.901, Confidence=0.923, Epsilon=0.37
Episode 150/250: Reward=89.3, Accuracy=0.934, Confidence=0.945, Epsilon=0.22
```

#### Q: What's the difference between the various model files?

**A:** Model file explanations:

- `dqn_model_best.pt`: Best performing model (highest validation accuracy)
- `dqn_model_final.pt`: Final model from training completion
- `episodes/dqn_model_episode_X.pt`: Checkpoint from episode X
- Models contain: weights, optimizer state, configuration, metadata

---

### API & Deployment

#### Q: The API server won't start. What's the issue?

**A:** Common API startup issues:

1. **Port already in use**:
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   # Use different port
   python -m api.main --port 8001
   ```

2. **Model file missing**:
   ```bash
   ls models/dqn_model_*.pt
   # Train a model if none exist
   python -m rl_ids.modeling.train
   ```

3. **Dependencies missing**:
   ```bash
   pip install fastapi uvicorn
   ```

4. **Permission issues**: Check file permissions and user access

#### Q: API predictions are returning errors. How do I fix this?

**A:** API troubleshooting:

1. **Check input format**:
   ```bash
   # Correct format: exactly 77 features
   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{"features": [/* 77 numeric values */]}'
   ```

2. **Validate feature count**:
   ```python
   import requests
   features = [0.1] * 77  # Exactly 77 features
   response = requests.post("http://localhost:8000/predict", 
                           json={"features": features})
   ```

3. **Check model loading**:
   ```bash
   curl http://localhost:8000/health
   # Should show "model_loaded": true
   ```

#### Q: How do I deploy RL-IDS in production?

**A:** Production deployment checklist:

1. **Docker deployment**:
   ```bash
   docker build -t rl-ids-api .
   docker run -d -p 8000:8000 --name rl-ids rl-ids-api
   ```

2. **Environment configuration**:
   ```bash
   # Create .env file
   RLIDS_HOST=0.0.0.0
   RLIDS_PORT=8000
   RLIDS_LOG_LEVEL=INFO
   RLIDS_MODEL_PATH=models/dqn_model_best.pt
   ```

3. **Security measures**:
   - Add authentication (API keys)
   - Enable HTTPS/TLS
   - Configure firewall rules
   - Set up monitoring

4. **Performance optimization**:
   - Use multiple workers: `uvicorn --workers 4`
   - Enable batch processing
   - Configure load balancing
   - Set up caching

#### Q: How do I monitor the API performance?

**A:** Monitoring strategies:

1. **Health endpoint**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Metrics collection**:
   - Response times
   - Request rates
   - Error rates
   - System resources

3. **Logging analysis**:
   ```bash
   # Check application logs
   tail -f logs/api.log
   ```

4. **Performance testing**:
   ```python
   from api.client import benchmark_api_performance
   import asyncio
   
   asyncio.run(benchmark_api_performance(100))
   ```

---

### Data & Features

#### Q: Can I use my own dataset instead of CICIDS2017?

**A:** Yes, but you'll need to:

1. **Format compatibility**: Ensure your data has:
   - Numeric features (77+ columns)
   - Label column with attack types
   - CSV format

2. **Data preprocessing**: Modify `rl_ids/make_dataset.py` for your schema

3. **Class mapping**: Update class names in the configuration

4. **Feature engineering**: Ensure features are network traffic related

#### Q: What do the 77 features represent?

**A:** The features are network traffic characteristics extracted using CICFlowMeter:

**Flow-based features**:
- Flow duration, packet counts, byte counts
- Forward/backward packet statistics
- Inter-arrival time statistics

**Packet-level features**:
- Packet length statistics (mean, max, min, std)
- Header information and flags
- Window size and urgent pointer

**Time-based features**:
- Flow inter-arrival times
- Active and idle time statistics
- Subflow characteristics

**Statistical features**:
- Flow rate and packet rate
- Bulk transfer characteristics
- Protocol-specific features

#### Q: How do I add new attack types?

**A:** To extend the system for new attacks:

1. **Data collection**: Gather labeled samples of new attack type

2. **Data integration**: Add to training dataset with proper labels

3. **Model retraining**: Retrain with updated class count:
   ```bash
   python -m rl_ids.modeling.train --action_dim 16  # If adding 1 new class
   ```

4. **Configuration update**: Update class names in API service

5. **Validation**: Test detection performance on new attack type

---

### Performance & Optimization

#### Q: Why is inference slow on my deployment?

**A:** Performance optimization checklist:

1. **Model size**: Use smaller architectures for production:
   ```python
   config = DQNConfig(hidden_dims=[256, 128])  # Smaller model
   ```

2. **Batch processing**: Process multiple samples together:
   ```python
   # Use batch prediction endpoint
   POST /predict/batch
   ```

3. **GPU inference**: Use GPU if available:
   ```python
   # Load model on GPU
   agent.load_model("model.pt", map_location="cuda")
   ```

4. **Caching**: Implement result caching for repeated requests

5. **Async processing**: Use FastAPI's async capabilities

#### Q: How can I improve detection accuracy?

**A:** Accuracy improvement strategies:

1. **Advanced training features**:
   ```bash
   python -m rl_ids.modeling.train \
       --double_dqn --dueling --prioritized_replay \
       --curriculum_learning
   ```

2. **Hyperparameter tuning**:
   - Lower learning rate: `--lr 5e-5`
   - Larger model: `--hidden_dims "2048,1024,512,256"`
   - More training: `--episodes 500`

3. **Data augmentation**:
   - Balance dataset with SMOTE
   - Add noise for robustness
   - Feature scaling/normalization

4. **Ensemble methods**: Train multiple models and combine predictions

#### Q: What's the expected memory usage?

**A:** Typical memory requirements:

**Training**:
- Model: ~8-15 MB
- Replay buffer: ~100-500 MB
- Data loading: ~2-4 GB
- GPU memory: ~2-4 GB
- Total system: ~8-16 GB

**Inference**:
- Model: ~8-15 MB  
- API service: ~100-200 MB
- Total system: ~500 MB - 1 GB

**Optimization tips**:
- Use smaller replay buffers
- Enable memory mapping for large datasets
- Use gradient checkpointing
- Process data in chunks

---

### Error Messages & Solutions

#### Error: `ModuleNotFoundError: No module named 'rl_ids'`

**Solution**:
```bash
# Install in development mode
pip install -e .

# Or add to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/rl_ids"
```

#### Error: `CUDA out of memory`

**Solution**:
```bash
# Option 1: Reduce batch size
python -m rl_ids.modeling.train --batch_size 16

# Option 2: Use CPU
CUDA_VISIBLE_DEVICES="" python -m rl_ids.modeling.train

# Option 3: Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

#### Error: `FileNotFoundError: No such file or directory: 'data/processed/train.csv'`

**Solution**:
```bash
# Run data preprocessing first
python -m rl_ids.make_dataset

# Check if raw data exists
ls data/raw/
```

#### Error: `ValidationError: Features list cannot be empty`

**Solution**:
```bash
# Ensure exactly 77 features in API request
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [/* exactly 77 numeric values */]}'
```

#### Error: `RuntimeError: Expected tensor for argument`

**Solution**:
```bash
# Check PyTorch compatibility
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

### Best Practices

#### Training Best Practices

1. **Start small**: Begin with 50-100 episodes to test setup
2. **Monitor progress**: Watch accuracy and reward trends
3. **Use validation**: Enable validation dataset for early stopping
4. **Save checkpoints**: Regular model saving during training
5. **Experiment tracking**: Use MLflow for experiment management

#### Production Best Practices

1. **Model validation**: Thoroughly test before deployment
2. **Monitoring**: Set up comprehensive health monitoring
3. **Backup strategies**: Keep model backpoints and rollback plans
4. **Security**: Implement authentication and input validation
5. **Performance testing**: Load test API endpoints
6. **Documentation**: Maintain deployment and operational docs

#### Development Best Practices

1. **Version control**: Use Git for code and model versioning
2. **Testing**: Write unit tests for critical components
3. **Code quality**: Use linting and formatting tools
4. **Documentation**: Keep documentation updated
5. **Reproducibility**: Use seeds and configuration files

---

### Getting Additional Help

#### Community Resources
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Join community discussions
- **Documentation**: Comprehensive guides and tutorials

#### Professional Support
- **Consulting**: Custom implementation and optimization
- **Training**: Team training and workshops
- **Integration**: Help with enterprise integration

#### Contributing
- **Bug Reports**: Submit detailed bug reports
- **Feature Requests**: Suggest new features
- **Code Contributions**: Submit pull requests
- **Documentation**: Improve documentation and examples

For specific issues not covered here, please create a detailed issue report including:
- Error messages and stack traces
- System information and environment
- Steps to reproduce the problem
- Expected vs. actual behavior
