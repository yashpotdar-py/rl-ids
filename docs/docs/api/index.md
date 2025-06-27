# API Reference

RL-IDS provides a FastAPI-based REST API for integration with other systems and programmatic access to threat detection capabilities.

## Overview

The RL-IDS API (`api/main.py`) provides a production-ready FastAPI service for real-time network intrusion detection using trained DQN models. The API is designed for high-performance, scalable threat detection with comprehensive error handling and monitoring capabilities.

## Quick Start

### Start the API Server

```bash
# Start with default settings
python run_api.py

# Start with custom host/port
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Base Configuration

The API is configured through `api/config.py` and can be customized via environment variables:

```python
# Default settings
HOST = "0.0.0.0"
PORT = 8000
MODEL_PATH = "models/dqn_model_final.pt"
```

## API Endpoints

### Root Endpoint

Get basic service information:

```http
GET /
```

**Response:**
```json
{
  "service": "RL-based Intrusion Detection System API",
  "version": "1.2.0",
  "status": "running",
  "docs": "/docs"
}
```

### Health Check

Check API status and model information:

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-27T10:30:00.123456",
  "details": {
    "model_loaded": true,
    "model_path": "models/dqn_model_final.pt",
    "predictions_served": 1234,
    "uptime_seconds": 3600.5,
    "memory_usage_mb": 256.7
  }
}
```

**Error Response (503):**
```json
{
  "detail": "Prediction service not initialized"
}
```

### Model Information

Get detailed information about the loaded model:

```http
GET /model/info
```

**Response:**
```json
{
  "model_name": "DQN_IDS_Model",
  "model_version": "1.0.0",
  "model_type": "Deep Q-Network",
  "input_features": 78,
  "output_classes": 15,
  "training_episodes": null,
  "model_size_mb": 2.5,
  "class_names": [
    "BENIGN",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Web Attack – Sql Injection",
    "FTP-Patator",
    "SSH-Patator",
    "PortScan",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
    "Bot",
    "DDoS",
    "Infiltration"
  ],
  "classification_type": "multi-class"
}
```

### Single Prediction

Analyze network features for threats:

```http
POST /predict
Content-Type: application/json

{
  "features": [
    0.123, 0.456, 0.789, 0.321, 0.654,
    0.987, 0.147, 0.258, 0.369, 0.741,
    ...  // 78 total features
  ]
}
```

**Success Response (200):**
```json
{
  "prediction": 9,
  "confidence": 0.87,
  "class_probabilities": [
    0.01, 0.02, 0.01, 0.02, 0.01,
    0.02, 0.01, 0.01, 0.02, 0.87,
    0.01, 0.01, 0.01, 0.01, 0.01
  ],
  "predicted_class": "DoS Hulk",
  "is_attack": true,
  "processing_time_ms": 12.5,
  "timestamp": "2025-06-27T10:30:45.123456"
}
```

**Validation Error (422):**
```json
{
  "detail": "Invalid input data: Expected 78 features, got 77"
}
```

### Batch Prediction

Analyze multiple feature sets:

```http
POST /predict/batch
Content-Type: application/json

[
  {
    "features": [0.1, 0.2, 0.3, ...]
  },
  {
    "features": [0.4, 0.5, 0.6, ...]
  }
]
```

**Response:**
```json
[
  {
    "prediction": 0,
    "confidence": 0.92,
    "class_probabilities": [0.92, 0.01, 0.01, ...],
    "predicted_class": "BENIGN",
    "is_attack": false,
    "processing_time_ms": 8.3,
    "timestamp": "2025-06-27T10:30:45.123456"
  },
  {
    "prediction": 6,
    "confidence": 0.78,
    "class_probabilities": [0.05, 0.02, 0.01, 0.01, 0.02, 0.01, 0.78, ...],
    "predicted_class": "PortScan",
    "is_attack": true,
    "processing_time_ms": 9.1,
    "timestamp": "2025-06-27T10:30:45.134567"
  }
]
```

**Batch Size Limit (413):**
```json
{
  "detail": "Batch size too large. Maximum 100 requests allowed."
}
```

## Request/Response Models

### IDSPredictionRequest

```python
class IDSPredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Network traffic features for prediction",
        min_items=1
    )
```

**Validation Rules:**
- Must contain exactly 78 features (CICIDS2017 standard)
- All features must be numeric (int or float)
- Features list cannot be empty

### IDSPredictionResponse

```python
class IDSPredictionResponse(BaseModel):
    prediction: int            # Predicted class (0-14)
    confidence: float          # Confidence score (0.0-1.0)
    class_probabilities: List[float]  # Probability for each class
    predicted_class: str       # Human-readable class name
    is_attack: bool           # True if attack detected (non-benign)
    processing_time_ms: float # Processing time in milliseconds
    timestamp: str            # Prediction timestamp (ISO format)
```

## Error Handling

The API returns standard HTTP status codes with detailed error information:

### Status Codes

- **200** - Success
- **422** - Validation Error (invalid input)
- **500** - Internal Server Error
- **503** - Service Unavailable (model not loaded)
- **413** - Request Entity Too Large (batch size exceeded)

### Error Response Format

```json
{
  "detail": "Error description",
  "timestamp": "2025-06-27T10:30:45.123456"
}
```

### Common Errors

**Feature Count Mismatch:**
```json
{
  "detail": "Invalid input data: Expected 78 features, got 75"
}
```

**Invalid Feature Values:**
```json
{
  "detail": "Invalid input data: All features must be numeric"
}
```

**Model Not Loaded:**
```json
{
  "detail": "Prediction service not initialized"
}
```

## API Configuration

### Environment Variables

Configure the API through environment variables:

```bash
# API server settings
RLIDS_API_HOST=0.0.0.0
RLIDS_API_PORT=8000

# Model settings
RLIDS_MODEL_PATH=models/dqn_model_best.pt

# Logging
RLIDS_LOG_LEVEL=INFO
RLIDS_DEBUG=false
```

### CORS Configuration

The API includes CORS middleware for cross-origin requests:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Service Architecture

### Application Lifecycle

The API uses FastAPI's lifespan events for proper service management:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global prediction_service
    model_path = MODELS_DIR / "dqn_model_final.pt"
    prediction_service = IDSPredictionService(model_path=model_path)
    await prediction_service.initialize()
    
    yield
    
    # Shutdown
    if prediction_service:
        await prediction_service.cleanup()
```

### Prediction Service

The `IDSPredictionService` class handles:

- **Model Loading**: Loads trained DQN models with configuration
- **Feature Validation**: Ensures input features match expected format
- **Prediction Processing**: Runs inference with timing metrics
- **Resource Management**: Handles GPU/CPU memory and cleanup

### Performance Characteristics

- **Model Loading Time**: ~2-5 seconds on startup
- **Prediction Latency**: ~8-15ms per prediction
- **Memory Usage**: ~200-500MB depending on model size
- **Throughput**: ~100-500 predictions/second (depends on hardware)

## OpenAPI Documentation

The API automatically generates comprehensive documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

### Interactive Testing

The Swagger UI provides interactive testing capabilities:
1. Navigate to `http://localhost:8000/docs`
2. Expand endpoint sections
3. Click "Try it out"
4. Input test data
5. Execute requests and view responses

## Security Considerations

### Production Deployment

For production environments, consider:

**Authentication**: Add API key or JWT token authentication
```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@app.post("/predict")
async def predict(request: IDSPredictionRequest, token: str = Depends(security)):
    # Verify token
    pass
```

**Rate Limiting**: Implement request rate limiting
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@limiter.limit("100/minute")
@app.post("/predict")
async def predict(request: Request, ...):
    pass
```

**HTTPS**: Use TLS/SSL in production
```bash
uvicorn api.main:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

**CORS**: Restrict origins for production
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Input Validation

The API includes comprehensive input validation:
- Feature count validation (must be exactly 78)
- Numeric type checking for all features
- Range validation to prevent extreme values
- Batch size limits to prevent resource exhaustion

## Monitoring and Logging

### Health Monitoring

The `/health` endpoint provides detailed service status:
- Model loading status
- Memory usage metrics
- Prediction statistics
- Service uptime

### Logging Configuration

Configure logging levels and outputs:

```python
# In api/config.py
LOG_LEVEL = os.getenv("RLIDS_LOG_LEVEL", "INFO")
```

**Log Levels:**
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages  
- `WARNING`: Warning conditions
- `ERROR`: Error conditions
- `CRITICAL`: Critical error conditions

### Performance Metrics

The API tracks performance metrics:
- Prediction processing time
- Total predictions served
- Error rates and types
- Memory usage patterns

RL-IDS provides a FastAPI-based REST API for integration with other systems and programmatic access to threat detection capabilities.

## Quick Start

Start the API server:

```bash
python run_api.py
```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

## Base Configuration

The API is configured through `api/config.py` and can be customized via environment variables.

Default settings:
- **Host**: `0.0.0.0`
- **Port**: `8000`
- **Model Path**: `models/dqn_model_best.pt`

## Core Endpoints

### Health Check

Check API status and model information:

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-27T10:30:00Z",
  "model_loaded": true,
  "model_path": "models/dqn_model_best.pt",
  "api_version": "1.2.0"
}
```

### Threat Prediction

Analyze network features for threats:

```http
POST /predict
Content-Type: application/json

{
  "features": [0.1, 0.2, 0.3, ...],  // 78 CICIDS2017 features
  "metadata": {
    "source_ip": "192.168.1.100",
    "timestamp": "2025-01-27T10:30:00Z"
  }
}
```

**Response:**
```json
{
  "is_attack": true,
  "predicted_class": "DoS Hulk",
  "confidence": 0.87,
  "model_version": "dqn_v1",
  "timestamp": "2025-01-27T10:30:00Z",
  "features_count": 78
}
```

### Batch Prediction

Analyze multiple feature sets:

```http
POST /predict_batch
Content-Type: application/json

{
  "batch": [
    {"features": [0.1, 0.2, ...], "id": "sample_1"},
    {"features": [0.3, 0.4, ...], "id": "sample_2"}
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "id": "sample_1",
      "is_attack": false,
      "predicted_class": "benign",
      "confidence": 0.92
    },
    {
      "id": "sample_2", 
      "is_attack": true,
      "predicted_class": "Port Scan",
      "confidence": 0.78
    }
  ],
  "processed_count": 2
}
```

## Python Client

RL-IDS includes a Python client for easy integration:

```python
from api.client import IDSAPIClient

# Initialize client
client = IDSAPIClient("http://localhost:8000")

# Health check
health = await client.health_check()
print(f"API Status: {health['status']}")

# Single prediction
features = [0.1, 0.2, 0.3, ...]  # 78 features
prediction = await client.predict(features)

if prediction['is_attack']:
    print(f"Attack detected: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.1%}")

# Close client
await client.close()
```

### Client Methods

The `IDSAPIClient` provides:

- `health_check()` - Check API health
- `predict(features)` - Single prediction
- `predict_batch(batch)` - Batch predictions
- `close()` - Close HTTP connections

### Performance Testing

Test API performance:

```python
import asyncio
from api.client import benchmark_api_performance

# Benchmark with 100 requests
results = await benchmark_api_performance(100)
print(f"Average response time: {results['avg_time']:.3f}s")
print(f"Requests per second: {results['rps']:.1f}")
```

## Error Handling

The API returns standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (invalid features)
- `422` - Validation Error
- `500` - Internal Server Error

**Error Response Format:**
```json
{
  "error": "Invalid feature count",
  "details": "Expected 78 features, received 77",
  "timestamp": "2025-01-27T10:30:00Z"
}
```

## Rate Limiting

For production deployment, implement rate limiting:

```python
# Example with slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@limiter.limit("100/minute")
async def predict_endpoint(request: Request, ...):
    # ... endpoint logic
```

## Authentication

For production, add authentication:

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    # Implement token verification
    if not verify_jwt_token(token.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
```

## OpenAPI Documentation

The API automatically generates OpenAPI documentation available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Configuration

Configure the API through environment variables or `api/config.py`:

```python
# api/config.py
class Settings:
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    model_path: str = "models/dqn_model_best.pt"
    log_level: str = "INFO"
    cors_origins: List[str] = ["*"]
```

## Docker Deployment

Deploy using Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_api.py"]
```

```bash
# Build and run
docker build -t rl-ids-api .
docker run -p 8000:8000 rl-ids-api
```