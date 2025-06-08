# RL-IDS FastAPI Documentation

## Overview

This FastAPI service provides a RESTful API for the RL-based Intrusion Detection System. It allows real-time network traffic classification using a trained Deep Q-Network (DQN) model.

## Features

- **Real-time Predictions**: Single and batch network traffic classification
- **Model Management**: Load and manage trained DQN models
- **Health Monitoring**: Comprehensive health checks and system monitoring
- **Async Processing**: High-performance async request handling
- **Docker Support**: Containerized deployment ready
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model is Available

Make sure you have a trained model at `models/dqn_model_final.pt`:

```bash
# Train a model if you haven't already
python -m rl_ids.modeling.train
```

### 3. Start the API Server

```bash
python run_api.py
```

The API will be available at:
- Main API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

- `GET /` - Service information
- `GET /health` - Health check and system status
- `GET /model/info` - Model information and metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

### Usage Examples

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
  }'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {"features": [0.1, 0.2, 0.3, 0.4, 0.5]},
    {"features": [0.6, 0.7, 0.8, 0.9, 1.0]}
  ]'
```

## Python Client

Use the provided client for programmatic access:

```python
from api.client import IDSAPIClient

client = IDSAPIClient("http://localhost:8000")

# Health check
health = await client.health_check()

# Single prediction
result = await client.predict([0.1, 0.2, 0.3, 0.4, 0.5])

# Batch prediction
results = await client.predict_batch([
    [0.1, 0.2, 0.3, 0.4, 0.5],
    [0.6, 0.7, 0.8, 0.9, 1.0]
])

await client.close()
```

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t rl-ids-api .

# Run the container
docker run -p 8000:8000 rl-ids-api
```

### Docker Compose

```bash
# Start the service
docker-compose up -d

# With monitoring (optional)
docker-compose --profile monitoring up -d

# With production setup (optional)
docker-compose --profile production up -d
```

## Configuration

Environment variables (prefix with `RLIDS_`):

- `RLIDS_HOST`: Server host (default: 0.0.0.0)
- `RLIDS_PORT`: Server port (default: 8000)
- `RLIDS_DEBUG`: Debug mode (default: False)
- `RLIDS_LOG_LEVEL`: Logging level (default: INFO)
- `RLIDS_MODEL_PATH`: Path to model file
- `RLIDS_MAX_BATCH_SIZE`: Maximum batch size (default: 100)

## Performance

The API is optimized for high-performance with:

- Async request handling
- Model caching and reuse
- Batch processing support
- GPU acceleration (when available)
- Connection pooling
- Request timeout protection

Typical performance metrics:
- Single prediction: ~10-50ms
- Batch prediction (10 items): ~30-100ms
- Throughput: 100-500 requests/second (depends on hardware)

## Monitoring and Health Checks

### Health Endpoint

The `/health` endpoint provides:
- Service status
- Model loading status
- Memory usage
- GPU usage (if available)
- Uptime statistics
- Prediction metrics

### Logging

Structured logging with:
- Request/response logging
- Performance metrics
- Error tracking
- Debug information

## Security Considerations

For production deployment:

1. **Authentication**: Add API key or JWT authentication
2. **Rate Limiting**: Implement request rate limiting
3. **Input Validation**: Strict input validation and sanitization
4. **CORS**: Configure CORS appropriately
5. **HTTPS**: Use TLS/SSL encryption
6. **Firewall**: Restrict network access
7. **Monitoring**: Set up security monitoring

## Testing

Run the test suite:

```bash
# Basic functionality tests
python -m api.client

# Performance benchmarks
python -c "
import asyncio
from api.client import benchmark_api_performance
asyncio.run(benchmark_api_performance(100))
"
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure model file exists at correct path
2. **Memory issues**: Reduce batch size or use CPU-only mode
3. **Port conflicts**: Change port in configuration
4. **CUDA errors**: Check GPU compatibility and drivers

### Debug Mode

Enable debug mode for detailed logging:

```bash
python run_api.py --log-level debug --reload
```

## Development

### Code Structure

```
api/
├── __init__.py          # Package initialization
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── services.py          # Business logic
├── config.py            # Configuration
├── utils.py             # Utility functions
└── client.py            # Test client
```

### Contributing

1. Follow existing code style
2. Add tests for new features
3. Update documentation
4. Test with different model configurations

## License

See LICENSE file for details.
