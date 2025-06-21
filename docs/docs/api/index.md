# API Reference

## Overview

The RL-IDS FastAPI service provides a RESTful API for real-time network intrusion detection using trained DQN models. The API is designed for high-performance, scalable deployment with comprehensive monitoring and health checks.

The service offers both single prediction and batch processing capabilities, with detailed response schemas including confidence scores, class probabilities, and performance metrics.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API key authentication
- OAuth 2.0
- JWT tokens

---

## Endpoints

### Health Check

#### `GET /health`

Check the health status of the API service and model availability.

**Response Schema:**
```json
{
  "status": "healthy",
  "timestamp": "2025-06-21T10:30:00.123456",
  "model_loaded": true,
  "system_info": {
    "cpu_percent": 25.5,
    "memory_percent": 67.2,
    "disk_usage_percent": 45.1
  },
  "service_info": {
    "uptime_seconds": 3600.5,
    "predictions_served": 1250,
    "version": "1.0.0"
  }
}
```

**Response Fields:**
- `status`: Service health status (`healthy`, `degraded`, `unhealthy`)
- `timestamp`: Current server timestamp
- `model_loaded`: Whether the DQN model is successfully loaded
- `system_info`: System resource utilization metrics
- `service_info`: Service-specific metrics and information

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is unhealthy or model not loaded

**Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

---

### Model Information

#### `GET /model/info`

Retrieve detailed information about the loaded DQN model.

**Response Schema:**
```json
{
  "model_type": "Enhanced DQN Agent",
  "architecture": "Dueling Double DQN",
  "state_dim": 78,
  "action_dim": 15,
  "hidden_layers": [1024, 512, 256, 128],
  "model_size_mb": 8.5,
  "device": "cuda:0",
  "training_info": {
    "episodes_trained": 250,
    "learning_rate": 0.0001,
    "gamma": 0.99,
    "epsilon": 0.01
  },
  "performance_metrics": {
    "accuracy": 0.9534,
    "precision": 0.9421,
    "recall": 0.9398,
    "f1_score": 0.9409
  },
  "class_names": [
    "BENIGN",
    "Web Attack – Brute Force",
    "Web Attack – XSS",
    "Web Attack – Sql Injection",
    "FTP-Patator",
    "SSH-Patator",
    "DoS slowloris",
    "DoS Slowhttptest",
    "DoS Hulk",
    "DoS GoldenEye",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "DDoS",
    "Bot"
  ]
}
```

**Response Fields:**
- `model_type`: Type of the loaded model
- `architecture`: Specific DQN architecture used
- `state_dim`: Input feature dimension
- `action_dim`: Number of output classes
- `hidden_layers`: Neural network architecture
- `model_size_mb`: Model size in memory
- `device`: Computation device (CPU/GPU)
- `training_info`: Training configuration and parameters
- `performance_metrics`: Model performance on validation set
- `class_names`: List of all attack types the model can detect

**Status Codes:**
- `200 OK`: Model information retrieved successfully
- `503 Service Unavailable`: Model not loaded

**Example:**
```bash
curl -X GET "http://localhost:8000/model/info"
```

---

### Single Prediction

#### `POST /predict`

Predict network intrusion for given network traffic features.

**Request Schema:**
```json
{
  "features": [
    0.123, 0.456, 0.789, 0.321, 0.654,
    0.987, 0.147, 0.258, 0.369, 0.741,
    // ... 78 total features
  ]
}
```

**Request Fields:**
- `features`: Array of 78 numeric features representing network traffic characteristics

**Response Schema:**
```json
{
  "prediction": 1,
  "confidence": 0.9534,
  "class_probabilities": [
    0.0123, 0.9534, 0.0098, 0.0087, 0.0076,
    0.0034, 0.0023, 0.0021, 0.0002, 0.0001,
    0.0001, 0.0000, 0.0000, 0.0000, 0.0000
  ],
  "predicted_class": "Web Attack – Brute Force",
  "is_attack": true,
  "processing_time_ms": 2.3,
  "timestamp": "2025-06-21T10:30:00.123456"
}
```

**Response Fields:**
- `prediction`: Predicted class index (0-14)
- `confidence`: Confidence score of the prediction (0.0-1.0)
- `class_probabilities`: Probability scores for all 15 classes
- `predicted_class`: Human-readable class name
- `is_attack`: Boolean indicating if traffic is malicious
- `processing_time_ms`: Prediction processing time in milliseconds
- `timestamp`: Prediction timestamp

**Status Codes:**
- `200 OK`: Prediction successful
- `422 Unprocessable Entity`: Invalid input data
- `503 Service Unavailable`: Prediction service not available

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [
         0.0, 0.0, 0.0, 80.0, 80.0, 6.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
       ]
     }'
```

---

### Batch Prediction

#### `POST /predict/batch`

Process multiple network traffic samples in a single request for improved efficiency.

**Request Schema:**
```json
{
  "features_batch": [
    [0.123, 0.456, /* ... 78 features */],
    [0.789, 0.321, /* ... 78 features */],
    [0.654, 0.987, /* ... 78 features */]
  ],
  "batch_id": "optional_batch_identifier"
}
```

**Request Fields:**
- `features_batch`: Array of feature arrays, each containing 78 features
- `batch_id`: Optional identifier for tracking batch requests

**Response Schema:**
```json
{
  "predictions": [
    {
      "prediction": 0,
      "confidence": 0.9821,
      "predicted_class": "BENIGN",
      "is_attack": false
    },
    {
      "prediction": 5,
      "confidence": 0.8967,
      "predicted_class": "SSH-Patator", 
      "is_attack": true
    }
  ],
  "batch_summary": {
    "total_samples": 2,
    "attack_count": 1,
    "benign_count": 1,
    "avg_confidence": 0.9394,
    "processing_time_ms": 5.7
  },
  "batch_id": "optional_batch_identifier",
  "timestamp": "2025-06-21T10:30:00.123456"
}
```

**Response Fields:**
- `predictions`: Array of individual prediction results
- `batch_summary`: Aggregated statistics for the batch
- `batch_id`: Echo of the request batch identifier
- `timestamp`: Batch processing timestamp

**Status Codes:**
- `200 OK`: Batch prediction successful
- `413 Payload Too Large`: Batch size exceeds maximum limit
- `422 Unprocessable Entity`: Invalid input data
- `503 Service Unavailable`: Prediction service not available

**Example:**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "features_batch": [
         [/* 78 features for sample 1 */],
         [/* 78 features for sample 2 */]
       ],
       "batch_id": "batch_001"
     }'
```

---

## Error Handling

The API uses standard HTTP status codes and provides detailed error information.

### Error Response Schema

```json
{
  "detail": "Error description",
  "error_type": "ValidationError",
  "timestamp": "2025-06-21T10:30:00.123456",
  "request_id": "req_123456789"
}
```

### Common Error Codes

| Status Code | Error Type | Description |
|-------------|------------|-------------|
| `400 Bad Request` | `ValidationError` | Invalid request format or parameters |
| `422 Unprocessable Entity` | `ValidationError` | Invalid feature data or schema violations |
| `413 Payload Too Large` | `PayloadError` | Batch size exceeds maximum limit |
| `429 Too Many Requests` | `RateLimitError` | Rate limit exceeded |
| `500 Internal Server Error` | `PredictionError` | Model prediction failed |
| `503 Service Unavailable` | `ServiceError` | Model not loaded or service unavailable |

### Validation Rules

**Feature Validation:**
- Must be numeric values (int or float)
- Array length must be exactly 78 features
- No null or missing values allowed
- Infinite values are rejected

**Batch Validation:**
- Maximum batch size: 100 samples (configurable)
- All samples must have consistent feature count
- Empty batches are rejected

---

## Rate Limiting

The API implements rate limiting to ensure fair usage and system stability.

**Default Limits:**
- **Per IP**: 100 requests per minute
- **Batch Requests**: 10 requests per minute
- **Global**: 1000 requests per minute

**Headers:**
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when the current window resets

---

## Performance Characteristics

### Latency
- **Single Prediction**: < 10ms average
- **Batch Prediction (10 samples)**: < 25ms average
- **Model Loading**: < 5 seconds

### Throughput
- **Single Predictions**: ~200 requests/second
- **Batch Processing**: ~500 samples/second
- **Concurrent Requests**: Supports async processing

### Resource Usage
- **Memory**: ~1GB base + 100MB per model
- **CPU**: Optimized for multi-core processing
- **GPU**: Optional CUDA acceleration

---

## Client Libraries

### Python Client

```python
from api.client import IDSAPIClient
import asyncio

async def main():
    client = IDSAPIClient("http://localhost:8000")
    
    # Health check
    health = await client.health_check()
    print(f"Service status: {health['status']}")
    
    # Single prediction
    features = [0.1] * 78  # Example features
    result = await client.predict(features)
    print(f"Prediction: {result['predicted_class']}")
    
    # Batch prediction
    batch_features = [[0.1] * 78, [0.2] * 78]
    batch_result = await client.predict_batch(batch_features)
    print(f"Batch processed: {len(batch_result['predictions'])} samples")

asyncio.run(main())
```

### JavaScript Client

```javascript
class IDSAPIClient {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async predict(features) {
    const response = await fetch(`${this.baseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ features })
    });
    return response.json();
  }

  async healthCheck() {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Usage
const client = new IDSAPIClient();
const result = await client.predict([0.1, 0.2, /* ... 78 features */]);
console.log('Prediction:', result.predicted_class);
```

---

## OpenAPI Documentation

The API provides interactive documentation through:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

These interfaces allow you to:
- Explore all available endpoints
- Test API calls directly in the browser
- View detailed request/response schemas
- Download the OpenAPI specification

---

## Monitoring and Observability

### Health Metrics

The `/health` endpoint provides comprehensive system metrics:

```json
{
  "system_info": {
    "cpu_percent": 25.5,
    "memory_percent": 67.2,
    "memory_available_gb": 8.2,
    "disk_usage_percent": 45.1,
    "gpu_utilization": 12.3
  },
  "service_info": {
    "uptime_seconds": 3600.5,
    "predictions_served": 1250,
    "requests_per_minute": 45.2,
    "avg_response_time_ms": 8.5,
    "error_rate_percent": 0.1
  }
}
```

### Logging

The API provides structured logging for monitoring and debugging:

- **Access Logs**: All API requests with timing and status
- **Error Logs**: Detailed error information and stack traces
- **Performance Logs**: Request processing times and resource usage
- **Model Logs**: Model loading, prediction metrics, and health status

### Metrics Integration

The service exposes metrics compatible with:
- **Prometheus**: `/metrics` endpoint (when enabled)
- **Grafana**: Pre-built dashboards available
- **ELK Stack**: Structured JSON logging
- **Custom Monitoring**: Health check endpoint for status monitoring

---

## Security Considerations

### Production Deployment

For production environments, implement these security measures:

1. **Authentication**: Add API key or JWT authentication
2. **Rate Limiting**: Configure appropriate request limits
3. **Input Validation**: Strict validation and sanitization
4. **HTTPS**: Use TLS/SSL encryption
5. **CORS**: Configure Cross-Origin Resource Sharing
6. **Firewall**: Restrict network access to authorized sources
7. **Monitoring**: Implement security monitoring and alerting

### Example Security Configuration

```python
# Add to API configuration
SECURITY_CONFIG = {
    "enable_auth": True,
    "api_key_header": "X-API-Key", 
    "rate_limit": "100/minute",
    "cors_origins": ["https://yourdomain.com"],
    "https_only": True
}
```

---

## Next Steps

- [Getting Started Guide](../getting-started.md) - Set up your first API service
- [Tutorials](../tutorials/api_usage.md) - Learn advanced API usage patterns
- [Deployment Guide](../tutorials/deployment.md) - Production deployment strategies
- [Module Reference](../modules/index.md) - Detailed code documentation

## Endpoints

### Health Check

#### `GET /health`

Check API service health and status.

**Response Model**: `HealthResponse`

**Response**

```json
{
  "status": "healthy",
  "timestamp": "2024-06-21T10:30:45.123456",
  "version": "1.0.0",
  "uptime_seconds": 3661.5,
  "model_loaded": true,
  "predictions_served": 1245,
  "system_info": {
    "cpu_percent": 15.2,
    "memory_percent": 32.8,
    "disk_percent": 45.1
  }
}
```

**Status Codes**
- `200`: Service is healthy
- `503`: Service unavailable

---

### Model Information

#### `GET /model/info`

Get detailed information about the loaded model.

**Response Model**: `ModelInfoResponse`

**Response**

```json
{
  "model_name": "DQN_IDS_Model",
  "model_version": "1.0.0",
  "input_features": 78,
  "output_classes": 15,
  "model_size_mb": 12.34,
  "training_info": {
    "episodes": 1000,
    "best_accuracy": 0.9876,
    "training_time_hours": 4.5
  },
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
  "feature_names": [...],
  "loaded_at": "2024-06-21T10:25:30.123456"
}
```

**Status Codes**
- `200`: Model information retrieved successfully
- `503`: Model not loaded

---

### Single Prediction

#### `POST /predict`

Predict network intrusion for given features.

**Request Model**: `IDSPredictionRequest`

**Request Body**

```json
{
  "features": [
    0.123, 0.456, 0.789, 0.321, 0.654,
    0.987, 0.147, 0.258, 0.369, 0.741,
    0.852, 0.963, 0.159, 0.357, 0.468,
    // ... 78 total features
  ]
}
```

**Response Model**: `IDSPredictionResponse`

**Response**

```json
{
  "prediction": 0,
  "confidence": 0.9234,
  "class_probabilities": [0.9234, 0.0543, 0.0123, ...],
  "predicted_class": "BENIGN",
  "is_attack": false,
  "processing_time_ms": 2.34,
  "timestamp": "2024-06-21T10:30:45.123456"
}
```

**Status Codes**
- `200`: Prediction successful
- `422`: Invalid input data
- `503`: Service unavailable

---

### Batch Prediction

#### `POST /predict/batch`

Predict multiple network samples at once.

**Request Model**: `List[IDSPredictionRequest]`

**Request Body**

```json
[
  {
    "features": [0.123, 0.456, ...]
  },
  {
    "features": [0.789, 0.321, ...]
  }
]
```

**Response Model**: `List[IDSPredictionResponse]`

**Response**

```json
[
  {
    "prediction": 0,
    "confidence": 0.9234,
    "predicted_class": "BENIGN",
    "is_attack": false,
    "processing_time_ms": 1.23,
    "timestamp": "2024-06-21T10:30:45.123456"
  },
  {
    "prediction": 6,
    "confidence": 0.8765,
    "predicted_class": "PortScan",
    "is_attack": true,
    "processing_time_ms": 1.45,
    "timestamp": "2024-06-21T10:30:45.134567"
  }
]
```

**Status Codes**
- `200`: Batch prediction successful
- `422`: Invalid input data
- `413`: Request too large (batch size limit exceeded)
- `503`: Service unavailable

---

## Data Models

### Request Models

#### `IDSPredictionRequest`

Single prediction request model.

**Fields**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `features` | `List[float]` | Yes | Network traffic features (78 values) |

**Validation**
- Features list cannot be empty
- All features must be numeric (int or float)
- Typically 78 features expected

**Example**

```json
{
  "features": [
    0.123, 0.456, 0.789, 0.321, 0.654,
    0.987, 0.147, 0.258, 0.369, 0.741,
    // ... remaining 68 features
  ]
}
```

### Response Models

#### `IDSPredictionResponse`

Single prediction response model.

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `int` | Predicted class index (0-14) |
| `confidence` | `float` | Prediction confidence (0.0-1.0) |
| `class_probabilities` | `List[float]` | Probabilities for all classes |
| `predicted_class` | `str` | Human-readable class name |
| `is_attack` | `bool` | Whether prediction indicates attack |
| `processing_time_ms` | `float` | Processing time in milliseconds |
| `timestamp` | `str` | Prediction timestamp (ISO format) |

#### `HealthResponse`

Health check response model.

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `status` | `str` | Service status ("healthy", "degraded", "unhealthy") |
| `timestamp` | `str` | Check timestamp |
| `version` | `str` | API version |
| `uptime_seconds` | `float` | Service uptime |
| `model_loaded` | `bool` | Whether model is loaded |
| `predictions_served` | `int` | Total predictions served |
| `system_info` | `dict` | System resource usage |

#### `ModelInfoResponse`

Model information response model.

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Model name |
| `model_version` | `str` | Model version |
| `input_features` | `int` | Number of input features |
| `output_classes` | `int` | Number of output classes |
| `model_size_mb` | `float` | Model size in megabytes |
| `training_info` | `dict` | Training metadata |
| `class_names` | `List[str]` | Class names |
| `feature_names` | `List[str]` | Feature names |
| `loaded_at` | `str` | Model load timestamp |

---

## Usage Examples

### Python Client

```python
import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{BASE_URL}/health")
print(f"Service status: {response.json()['status']}")

# Model information
response = requests.get(f"{BASE_URL}/model/info")
model_info = response.json()
print(f"Model: {model_info['model_name']} v{model_info['model_version']}")
print(f"Classes: {len(model_info['class_names'])}")

# Single prediction
features = [0.123, 0.456, 0.789] + [0.0] * 75  # 78 features total
prediction_request = {"features": features}

response = requests.post(
    f"{BASE_URL}/predict",
    json=prediction_request
)

result = response.json()
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Attack: {result['is_attack']}")

# Batch prediction
batch_request = [
    {"features": features},
    {"features": [0.987, 0.654, 0.321] + [0.0] * 75}
]

response = requests.post(
    f"{BASE_URL}/predict/batch",
    json=batch_request
)

results = response.json()
for i, result in enumerate(results):
    print(f"Sample {i+1}: {result['predicted_class']} "
          f"({result['confidence']:.2%})")
```

### JavaScript Client

```javascript
// Health check
async function checkHealth() {
    const response = await fetch('http://localhost:8000/health');
    const health = await response.json();
    console.log(`Service status: ${health.status}`);
    return health;
}

// Single prediction
async function predictIntrusion(features) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ features: features })
    });
    
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    console.log(`Prediction: ${result.predicted_class}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
    
    return result;
}

// Usage
const features = [0.123, 0.456, 0.789, ...Array(75).fill(0)];
predictIntrusion(features)
    .then(result => console.log('Prediction successful:', result))
    .catch(error => console.error('Prediction failed:', error));
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Model information
curl -X GET "http://localhost:8000/model/info"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [0.123, 0.456, 0.789, 0.321, 0.654, 0.987, 0.147, 0.258, 0.369, 0.741, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     }'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '[
       {"features": [0.123, 0.456, ...]},
       {"features": [0.789, 0.321, ...]}
     ]'
```

## Error Handling

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2024-06-21T10:30:45.123456"
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `VALIDATION_ERROR` | 422 | Invalid input data |
| `MODEL_NOT_LOADED` | 503 | Model not available |
| `PREDICTION_FAILED` | 500 | Internal prediction error |
| `BATCH_TOO_LARGE` | 413 | Batch size exceeds limit |
| `RATE_LIMITED` | 429 | Too many requests |

### Error Examples

```json
// Invalid features
{
  "detail": "Features list cannot be empty",
  "error_code": "VALIDATION_ERROR"
}

// Model not loaded
{
  "detail": "Prediction service not initialized",
  "error_code": "MODEL_NOT_LOADED"
}

// Batch too large
{
  "detail": "Batch size exceeds maximum limit of 100",
  "error_code": "BATCH_TOO_LARGE"
}
```

## Performance

### Response Times

| Endpoint | Typical Response Time |
|----------|----------------------|
| `/health` | < 5ms |
| `/model/info` | < 10ms |
| `/predict` | 2-5ms |
| `/predict/batch` (10 samples) | 5-15ms |

### Rate Limits

- **Default**: 1000 requests/minute per IP
- **Burst**: Up to 100 requests in 10 seconds
- **Configurable** via environment variables

### Scaling

- **Horizontal**: Deploy multiple instances behind load balancer
- **Vertical**: Increase CPU/memory for better throughput
- **GPU**: Use CUDA for faster inference

## Monitoring

### Metrics Available

- Request count by endpoint
- Response time percentiles
- Error rates
- Model prediction accuracy
- System resource usage

### Health Check Details

The `/health` endpoint provides comprehensive status:

```json
{
  "status": "healthy",
  "checks": {
    "model_loaded": true,
    "memory_usage": "normal",
    "cpu_usage": "normal",
    "disk_space": "normal"
  },
  "metrics": {
    "requests_per_minute": 45,
    "avg_response_time_ms": 3.2,
    "error_rate": 0.001
  }
}
```

## See Also

- [Getting Started Guide](../getting-started.md) - API setup and usage
- [Service Module](service.md) - Implementation details
- [Deployment Guide](deployment.md) - Production deployment
