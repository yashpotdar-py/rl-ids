# API Usage Tutorial

## Overview

This comprehensive tutorial covers advanced usage patterns for the RL-IDS FastAPI service, including authentication, batch processing, monitoring, integration patterns, and production deployment strategies.

## Prerequisites

- Completed [Getting Started Guide](../getting-started.md)
- Trained RL-IDS model available
- Basic understanding of REST APIs and HTTP

## Learning Objectives

By the end of this tutorial, you will:

1. Master all API endpoints and their advanced features
2. Implement efficient batch processing workflows
3. Set up comprehensive monitoring and alerting
4. Integrate the API with various systems and frameworks
5. Deploy and scale the API in production environments

## 1. API Service Setup and Configuration

### Advanced Configuration

Start with a comprehensive API configuration:

```python
from api.config import APISettings
from pathlib import Path
import os

# Advanced API configuration
settings = APISettings(
    # Basic settings
    app_name="RL-IDS Production API",
    app_version="2.0.0",
    debug=False,
    
    # Server configuration
    host="0.0.0.0",
    port=8000,
    workers=4,  # Scale based on CPU cores
    
    # Model settings
    model_path=Path("models/dqn_model_best.pt"),
    model_backup_path=Path("models/dqn_model_backup.pt"),
    
    # Performance optimization
    max_batch_size=500,
    prediction_timeout=30.0,
    enable_model_caching=True,
    cache_size=1000,
    
    # Monitoring
    log_level="INFO",
    enable_metrics=True,
    metrics_port=9090,
    
    # Security
    enable_cors=True,
    cors_origins=["https://your-frontend.com"],
    cors_methods=["GET", "POST"],
    
    # Rate limiting
    rate_limit_enabled=True,
    rate_limit_requests_per_minute=1000,
    rate_limit_burst=50,
    
    # Health checks
    health_check_interval=30,
    model_health_check=True
)

print(f"API Configuration: {settings}")
```

### Environment-Specific Configuration

Set up different configurations for different environments:

```bash
# .env.development
RLIDS_API_DEBUG=true
RLIDS_API_LOG_LEVEL=DEBUG
RLIDS_API_WORKERS=1
RLIDS_API_RATE_LIMIT_ENABLED=false

# .env.production
RLIDS_API_DEBUG=false
RLIDS_API_LOG_LEVEL=INFO
RLIDS_API_WORKERS=8
RLIDS_API_RATE_LIMIT_ENABLED=true
RLIDS_API_MAX_BATCH_SIZE=1000
RLIDS_API_CORS_ORIGINS=["https://production-app.com"]

# .env.testing
RLIDS_API_DEBUG=true
RLIDS_API_LOG_LEVEL=DEBUG
RLIDS_API_WORKERS=1
RLIDS_API_MODEL_PATH=/tmp/test_model.pt
```

### Starting the API Service

```bash
# Development mode with auto-reload
python run_api.py --reload --log-level debug

# Production mode with multiple workers
python run_api.py --host 0.0.0.0 --port 8000 --workers 4

# Using Gunicorn for production
gunicorn api.main:app \
    --bind 0.0.0.0:8000 \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile - \
    --error-logfile - \
    --log-level info

# Docker deployment
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -e RLIDS_API_WORKERS=4 \
    rl-ids-api:latest
```

## 2. Comprehensive API Client Usage

### Advanced Python Client

Create a robust Python client with advanced features:

```python
import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from loguru import logger
import numpy as np
import pandas as pd

@dataclass
class PredictionResult:
    """Structured prediction result"""
    predicted_class: int
    predicted_class_name: str
    confidence: float
    is_attack: bool
    processing_time_ms: float
    model_version: str
    raw_probabilities: List[float]

class AdvancedIDSAPIClient:
    """Advanced API client with connection pooling, retries, and monitoring"""
    
    def __init__(self, base_url: str = "http://localhost:8000", 
                 timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = None
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0
        }
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=30,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'RL-IDS Advanced Client v2.0'}
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, endpoint: str, 
                           json_data: Dict = None, retry_count: int = 0) -> Dict:
        """Make HTTP request with retry logic"""
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            self.request_stats['total_requests'] += 1
            
            async with self.session.request(method, url, json=json_data) as response:
                response_time = (time.time() - start_time) * 1000
                self.request_stats['total_response_time'] += response_time
                
                if response.status == 200:
                    self.request_stats['successful_requests'] += 1
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    if retry_count < self.max_retries:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logger.warning(f"Rate limited, waiting {wait_time}s before retry")
                        await asyncio.sleep(wait_time)
                        return await self._make_request(method, endpoint, json_data, retry_count + 1)
                else:
                    error_text = await response.text()
                    raise aiohttp.ClientError(f"HTTP {response.status}: {error_text}")
                    
        except asyncio.TimeoutError:
            if retry_count < self.max_retries:
                logger.warning(f"Request timeout, retrying... ({retry_count + 1}/{self.max_retries})")
                return await self._make_request(method, endpoint, json_data, retry_count + 1)
            else:
                self.request_stats['failed_requests'] += 1
                raise
        except Exception as e:
            self.request_stats['failed_requests'] += 1
            if retry_count < self.max_retries:
                logger.warning(f"Request failed: {e}, retrying... ({retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(1)
                return await self._make_request(method, endpoint, json_data, retry_count + 1)
            else:
                raise
    
    async def health_check(self) -> Dict:
        """Comprehensive health check"""
        return await self._make_request('GET', '/health')
    
    async def get_model_info(self) -> Dict:
        """Get detailed model information"""
        return await self._make_request('GET', '/model/info')
    
    async def predict_single(self, features: List[float]) -> PredictionResult:
        """Single prediction with structured result"""
        
        response = await self._make_request('POST', '/predict', 
                                          {'features': features})
        
        return PredictionResult(
            predicted_class=response['predicted_class'],
            predicted_class_name=response['predicted_class_name'],
            confidence=response['confidence'],
            is_attack=response['is_attack'],
            processing_time_ms=response['processing_time_ms'],
            model_version=response['model_version'],
            raw_probabilities=response.get('probabilities', [])
        )
    
    async def predict_batch(self, features_list: List[List[float]], 
                           batch_size: int = 100) -> List[PredictionResult]:
        """Efficient batch prediction with chunking"""
        
        all_results = []
        
        # Process in chunks to respect API limits
        for i in range(0, len(features_list), batch_size):
            chunk = features_list[i:i + batch_size]
            chunk_requests = [{'features': features} for features in chunk]
            
            response = await self._make_request('POST', '/predict/batch', 
                                              chunk_requests)
            
            # Convert to structured results
            chunk_results = [
                PredictionResult(
                    predicted_class=pred['predicted_class'],
                    predicted_class_name=pred['predicted_class_name'],
                    confidence=pred['confidence'],
                    is_attack=pred['is_attack'],
                    processing_time_ms=pred['processing_time_ms'],
                    model_version=pred['model_version'],
                    raw_probabilities=pred.get('probabilities', [])
                )
                for pred in response
            ]
            
            all_results.extend(chunk_results)
            
            # Progress logging
            logger.info(f"Processed batch {i//batch_size + 1}, "
                       f"total predictions: {len(all_results)}")
        
        return all_results
    
    async def predict_dataframe(self, df: pd.DataFrame, 
                               feature_columns: List[str] = None,
                               batch_size: int = 100) -> pd.DataFrame:
        """Predict on pandas DataFrame with results integration"""
        
        if feature_columns is None:
            # Assume all numeric columns except labels
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = [col for col in feature_columns 
                             if not col.lower().startswith('label')]
        
        # Extract features
        features_list = df[feature_columns].values.tolist()
        
        # Get predictions
        predictions = await self.predict_batch(features_list, batch_size)
        
        # Add results to DataFrame
        result_df = df.copy()
        result_df['predicted_class'] = [p.predicted_class for p in predictions]
        result_df['predicted_class_name'] = [p.predicted_class_name for p in predictions]
        result_df['confidence'] = [p.confidence for p in predictions]
        result_df['is_attack'] = [p.is_attack for p in predictions]
        result_df['processing_time_ms'] = [p.processing_time_ms for p in predictions]
        
        return result_df
    
    async def stream_predictions(self, features_stream, 
                                prediction_callback=None) -> None:
        """Stream predictions for real-time processing"""
        
        async for features in features_stream:
            try:
                prediction = await self.predict_single(features)
                
                if prediction_callback:
                    await prediction_callback(features, prediction)
                else:
                    logger.info(f"Prediction: {prediction.predicted_class_name} "
                               f"(confidence: {prediction.confidence:.3f})")
                    
            except Exception as e:
                logger.error(f"Stream prediction error: {e}")
    
    def get_stats(self) -> Dict:
        """Get client performance statistics"""
        
        stats = self.request_stats.copy()
        
        if stats['total_requests'] > 0:
            stats['success_rate'] = stats['successful_requests'] / stats['total_requests']
            stats['average_response_time_ms'] = stats['total_response_time'] / stats['total_requests']
        else:
            stats['success_rate'] = 0.0
            stats['average_response_time_ms'] = 0.0
        
        return stats

# Example usage
async def advanced_client_example():
    """Comprehensive client usage example"""
    
    async with AdvancedIDSAPIClient("http://localhost:8000") as client:
        # Health check
        health = await client.health_check()
        logger.info(f"Service health: {health['status']}")
        
        # Model info
        model_info = await client.get_model_info()
        logger.info(f"Model: {model_info['model_name']} v{model_info['model_version']}")
        
        # Single prediction
        features = [0.1] * 77  # Example features
        prediction = await client.predict_single(features)
        logger.info(f"Single prediction: {prediction}")
        
        # Batch prediction
        batch_features = [[0.1] * 77 for _ in range(100)]
        batch_predictions = await client.predict_batch(batch_features, batch_size=50)
        logger.info(f"Batch predictions completed: {len(batch_predictions)} samples")
        
        # DataFrame prediction
        df = pd.DataFrame(np.random.rand(200, 77))
        result_df = await client.predict_dataframe(df)
        logger.info(f"DataFrame prediction completed: {len(result_df)} rows")
        
        # Performance stats
        stats = client.get_stats()
        logger.info(f"Client stats: {stats}")

# Run the example
asyncio.run(advanced_client_example())
```

### JavaScript/TypeScript Client

Create a robust JavaScript client for web applications:

```javascript
/**
 * Advanced RL-IDS API Client for JavaScript/TypeScript
 */
class AdvancedIDSAPIClient {
    constructor(baseUrl = 'http://localhost:8000', options = {}) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.timeout = options.timeout || 30000;
        this.maxRetries = options.maxRetries || 3;
        this.apiKey = options.apiKey || null;
        
        this.stats = {
            totalRequests: 0,
            successfulRequests: 0,
            failedRequests: 0,
            totalResponseTime: 0
        };
    }
    
    /**
     * Make HTTP request with retry logic
     */
    async makeRequest(method, endpoint, data = null, retryCount = 0) {
        const url = `${this.baseUrl}${endpoint}`;
        const startTime = performance.now();
        
        const headers = {
            'Content-Type': 'application/json',
            'User-Agent': 'RL-IDS JS Client v2.0'
        };
        
        if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
        }
        
        const config = {
            method,
            headers,
            signal: AbortSignal.timeout(this.timeout)
        };
        
        if (data) {
            config.body = JSON.stringify(data);
        }
        
        try {
            this.stats.totalRequests++;
            
            const response = await fetch(url, config);
            const responseTime = performance.now() - startTime;
            this.stats.totalResponseTime += responseTime;
            
            if (response.ok) {
                this.stats.successfulRequests++;
                return await response.json();
            } else if (response.status === 429 && retryCount < this.maxRetries) {
                // Rate limited - exponential backoff
                const waitTime = Math.pow(2, retryCount) * 1000;
                console.warn(`Rate limited, waiting ${waitTime}ms before retry`);
                await this.sleep(waitTime);
                return this.makeRequest(method, endpoint, data, retryCount + 1);
            } else {
                const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
        } catch (error) {
            this.stats.failedRequests++;
            
            if (retryCount < this.maxRetries && 
                (error.name === 'TimeoutError' || error.name === 'NetworkError')) {
                console.warn(`Request failed: ${error.message}, retrying...`);
                await this.sleep(1000);
                return this.makeRequest(method, endpoint, data, retryCount + 1);
            } else {
                throw error;
            }
        }
    }
    
    /**
     * Utility function for delays
     */
    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    /**
     * Health check endpoint
     */
    async healthCheck() {
        return this.makeRequest('GET', '/health');
    }
    
    /**
     * Get model information
     */
    async getModelInfo() {
        return this.makeRequest('GET', '/model/info');
    }
    
    /**
     * Single prediction
     */
    async predict(features) {
        return this.makeRequest('POST', '/predict', { features });
    }
    
    /**
     * Batch prediction with chunking
     */
    async predictBatch(featuresList, batchSize = 100) {
        const allResults = [];
        
        for (let i = 0; i < featuresList.length; i += batchSize) {
            const chunk = featuresList.slice(i, i + batchSize);
            const chunkRequests = chunk.map(features => ({ features }));
            
            const response = await this.makeRequest('POST', '/predict/batch', chunkRequests);
            allResults.push(...response);
            
            console.log(`Processed batch ${Math.floor(i / batchSize) + 1}, total predictions: ${allResults.length}`);
        }
        
        return allResults;
    }
    
    /**
     * Real-time prediction stream
     */
    async *streamPredictions(featuresGenerator) {
        for await (const features of featuresGenerator) {
            try {
                const prediction = await this.predict(features);
                yield { features, prediction };
            } catch (error) {
                console.error('Stream prediction error:', error);
                yield { features, error };
            }
        }
    }
    
    /**
     * Monitor API performance
     */
    getStats() {
        const stats = { ...this.stats };
        
        if (stats.totalRequests > 0) {
            stats.successRate = stats.successfulRequests / stats.totalRequests;
            stats.averageResponseTime = stats.totalResponseTime / stats.totalRequests;
        } else {
            stats.successRate = 0;
            stats.averageResponseTime = 0;
        }
        
        return stats;
    }
}

// Example usage in web application
async function webAppExample() {
    const client = new AdvancedIDSAPIClient('http://localhost:8000');
    
    try {
        // Check service health
        const health = await client.healthCheck();
        console.log('Service status:', health.status);
        
        // Get model information
        const modelInfo = await client.getModelInfo();
        console.log(`Model: ${modelInfo.model_name} v${modelInfo.model_version}`);
        
        // Real-time prediction example
        const features = Array(77).fill(0).map(() => Math.random());
        const prediction = await client.predict(features);
        
        console.log(`Prediction: ${prediction.predicted_class_name}`);
        console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`);
        console.log(`Is Attack: ${prediction.is_attack}`);
        
        // Display in UI
        updateUI(prediction);
        
    } catch (error) {
        console.error('API error:', error);
        showErrorMessage(error.message);
    }
}

function updateUI(prediction) {
    // Update web UI with prediction results
    const resultElement = document.getElementById('prediction-result');
    const confidenceElement = document.getElementById('confidence-bar');
    
    resultElement.textContent = prediction.predicted_class_name;
    resultElement.className = prediction.is_attack ? 'alert-danger' : 'alert-success';
    
    confidenceElement.style.width = `${prediction.confidence * 100}%`;
    confidenceElement.textContent = `${(prediction.confidence * 100).toFixed(1)}%`;
}

function showErrorMessage(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = `Error: ${message}`;
    errorElement.style.display = 'block';
}
```

## 3. Integration Patterns

### Integration with Monitoring Systems

#### Prometheus Metrics Integration

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

class PrometheusMetrics:
    """Prometheus metrics for API monitoring"""
    
    def __init__(self):
        # Request metrics
        self.request_count = Counter('rlids_api_requests_total', 
                                   'Total API requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('rlids_api_request_duration_seconds',
                                        'Request duration in seconds', ['method', 'endpoint'])
        
        # Prediction metrics
        self.prediction_count = Counter('rlids_predictions_total',
                                      'Total predictions made', ['predicted_class'])
        self.prediction_confidence = Histogram('rlids_prediction_confidence',
                                             'Prediction confidence scores')
        
        # Model metrics
        self.model_load_time = Gauge('rlids_model_load_time_seconds',
                                   'Model load time in seconds')
        self.model_memory_usage = Gauge('rlids_model_memory_usage_bytes',
                                      'Model memory usage in bytes')
        
        # Start metrics server
        start_http_server(9090)
    
    def record_request(self, method: str, endpoint: str, status: str, duration: float):
        """Record request metrics"""
        self.request_count.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_prediction(self, predicted_class: str, confidence: float):
        """Record prediction metrics"""
        self.prediction_count.labels(predicted_class=predicted_class).inc()
        self.prediction_confidence.observe(confidence)

# Integrate with FastAPI
from fastapi import Request
import time

metrics = PrometheusMetrics()

@app.middleware("http")
async def add_metrics_middleware(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    metrics.record_request(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
        duration=process_time
    )
    
    return response
```

#### ELK Stack Integration

```python
import json
import logging
from datetime import datetime
from loguru import logger

class ELKLogger:
    """Structured logging for ELK stack"""
    
    def __init__(self):
        # Configure structured logging
        logger.configure(
            handlers=[
                {
                    "sink": "logs/rlids_api.json",
                    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[request_id]} | {message}",
                    "serialize": True,
                    "rotation": "1 day",
                    "retention": "30 days"
                }
            ]
        )
    
    def log_prediction(self, request_id: str, features: list, prediction: dict, 
                      processing_time: float):
        """Log prediction event for ELK analysis"""
        
        log_entry = {
            "event_type": "prediction",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": {
                "class": prediction["predicted_class"],
                "class_name": prediction["predicted_class_name"],
                "confidence": prediction["confidence"],
                "is_attack": prediction["is_attack"]
            },
            "performance": {
                "processing_time_ms": processing_time,
                "feature_count": len(features)
            },
            "metadata": {
                "model_version": prediction.get("model_version"),
                "api_version": "2.0.0"
            }
        }
        
        logger.bind(request_id=request_id).info(json.dumps(log_entry))
    
    def log_error(self, request_id: str, error: str, context: dict = None):
        """Log error event"""
        
        log_entry = {
            "event_type": "error",
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "error": {
                "message": str(error),
                "type": type(error).__name__
            },
            "context": context or {}
        }
        
        logger.bind(request_id=request_id).error(json.dumps(log_entry))

elk_logger = ELKLogger()
```

### Database Integration for Prediction Storage

```python
import asyncpg
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from typing import Optional

Base = declarative_base()

class PredictionRecord(Base):
    """Database model for storing predictions"""
    
    __tablename__ = 'predictions'
    
    id = sa.Column(sa.Integer, primary_key=True)
    request_id = sa.Column(sa.String, nullable=False, index=True)
    timestamp = sa.Column(sa.DateTime, default=datetime.utcnow, index=True)
    
    # Input features (stored as JSON)
    features = sa.Column(sa.JSON)
    feature_hash = sa.Column(sa.String, index=True)  # For deduplication
    
    # Prediction results
    predicted_class = sa.Column(sa.Integer, nullable=False)
    predicted_class_name = sa.Column(sa.String, nullable=False)
    confidence = sa.Column(sa.Float, nullable=False)
    is_attack = sa.Column(sa.Boolean, nullable=False)
    probabilities = sa.Column(sa.JSON)
    
    # Performance metrics
    processing_time_ms = sa.Column(sa.Float)
    model_version = sa.Column(sa.String)
    
    # Optional ground truth for model monitoring
    true_label = sa.Column(sa.Integer, nullable=True)
    is_correct = sa.Column(sa.Boolean, nullable=True)

class PredictionStorage:
    """Database storage for predictions"""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url)
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def store_prediction(self, request_id: str, features: list, 
                             prediction: dict, processing_time: float) -> int:
        """Store prediction in database"""
        
        async with self.SessionLocal() as session:
            # Calculate feature hash for deduplication
            feature_hash = hashlib.md5(str(features).encode()).hexdigest()
            
            record = PredictionRecord(
                request_id=request_id,
                features=features,
                feature_hash=feature_hash,
                predicted_class=prediction['predicted_class'],
                predicted_class_name=prediction['predicted_class_name'],
                confidence=prediction['confidence'],
                is_attack=prediction['is_attack'],
                probabilities=prediction.get('probabilities'),
                processing_time_ms=processing_time,
                model_version=prediction.get('model_version')
            )
            
            session.add(record)
            await session.commit()
            
            return record.id
    
    async def get_prediction_stats(self, hours: int = 24) -> dict:
        """Get prediction statistics for the last N hours"""
        
        async with self.SessionLocal() as session:
            since = datetime.utcnow() - timedelta(hours=hours)
            
            # Total predictions
            total_query = sa.select(sa.func.count(PredictionRecord.id)).where(
                PredictionRecord.timestamp >= since
            )
            total_result = await session.execute(total_query)
            total_predictions = total_result.scalar()
            
            # Attack predictions
            attack_query = sa.select(sa.func.count(PredictionRecord.id)).where(
                sa.and_(
                    PredictionRecord.timestamp >= since,
                    PredictionRecord.is_attack == True
                )
            )
            attack_result = await session.execute(attack_query)
            attack_predictions = attack_result.scalar()
            
            # Average confidence
            confidence_query = sa.select(sa.func.avg(PredictionRecord.confidence)).where(
                PredictionRecord.timestamp >= since
            )
            confidence_result = await session.execute(confidence_query)
            avg_confidence = confidence_result.scalar() or 0.0
            
            return {
                'total_predictions': total_predictions,
                'attack_predictions': attack_predictions,
                'benign_predictions': total_predictions - attack_predictions,
                'attack_rate': attack_predictions / max(total_predictions, 1),
                'average_confidence': float(avg_confidence)
            }

# Integration with API
prediction_storage = PredictionStorage("postgresql+asyncpg://user:pass@localhost/rlids")

@app.post("/predict")
async def predict_with_storage(request: IDSPredictionRequest):
    # ... existing prediction logic ...
    
    # Store prediction
    await prediction_storage.store_prediction(
        request_id=str(uuid.uuid4()),
        features=request.features,
        prediction=result,
        processing_time=processing_time
    )
    
    return result
```

## 4. Performance Optimization and Scaling

### Connection Pooling and Load Balancing

```yaml
# docker-compose.yml for load balanced setup
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api-1
      - api-2
      - api-3

  api-1:
    build: .
    environment:
      - RLIDS_API_WORKERS=4
      - RLIDS_API_PORT=8000
    volumes:
      - ./models:/app/models

  api-2:
    build: .
    environment:
      - RLIDS_API_WORKERS=4
      - RLIDS_API_PORT=8000
    volumes:
      - ./models:/app/models

  api-3:
    build: .
    environment:
      - RLIDS_API_WORKERS=4
      - RLIDS_API_PORT=8000
    volumes:
      - ./models:/app/models

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  postgresql:
    image: postgres:13
    environment:
      POSTGRES_DB: rlids
      POSTGRES_USER: rlids
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream rlids_api {
        server api-1:8000;
        server api-2:8000;
        server api-3:8000;
    }
    
    server {
        listen 80;
        
        location / {
            proxy_pass http://rlids_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            
            # Connection pooling
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            
            # Timeouts
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        location /health {
            access_log off;
            proxy_pass http://rlids_api;
        }
    }
}
```

### Redis Caching Integration

```python
import redis.asyncio as redis
import json
import hashlib
from typing import Optional

class RedisCacheManager:
    """Redis-based caching for API responses"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.default_ttl = 3600  # 1 hour
        
    async def get_cached_prediction(self, features: list) -> Optional[dict]:
        """Get cached prediction if available"""
        
        # Create cache key from features
        feature_str = json.dumps(features, sort_keys=True)
        cache_key = f"prediction:{hashlib.md5(feature_str.encode()).hexdigest()}"
        
        cached_result = await self.redis.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        return None
    
    async def cache_prediction(self, features: list, prediction: dict, ttl: int = None):
        """Cache prediction result"""
        
        feature_str = json.dumps(features, sort_keys=True)
        cache_key = f"prediction:{hashlib.md5(feature_str.encode()).hexdigest()}"
        
        await self.redis.setex(
            cache_key, 
            ttl or self.default_ttl,
            json.dumps(prediction)
        )
    
    async def get_model_stats(self) -> dict:
        """Get cached model statistics"""
        
        stats = await self.redis.hgetall("model:stats")
        return {k: json.loads(v) for k, v in stats.items()}
    
    async def update_model_stats(self, stats: dict):
        """Update model statistics cache"""
        
        pipeline = self.redis.pipeline()
        for key, value in stats.items():
            pipeline.hset("model:stats", key, json.dumps(value))
        await pipeline.execute()

# Integration with API
cache_manager = RedisCacheManager()

@app.post("/predict")
async def predict_with_cache(request: IDSPredictionRequest):
    # Check cache first
    cached_result = await cache_manager.get_cached_prediction(request.features)
    if cached_result:
        logger.info("Returning cached prediction")
        return cached_result
    
    # Compute prediction
    result = await prediction_service.predict(request.features)
    
    # Cache result
    await cache_manager.cache_prediction(request.features, result)
    
    return result
```

## 5. Security and Authentication

### JWT Authentication

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

class JWTAuthManager:
    """JWT-based authentication for API"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = timedelta(hours=24)
    
    def create_access_token(self, user_id: str, permissions: list = None) -> str:
        """Create JWT access token"""
        
        payload = {
            "user_id": user_id,
            "permissions": permissions or [],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

# Setup authentication
auth_manager = JWTAuthManager(secret_key="your-secret-key")
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Dependency to get current authenticated user"""
    
    payload = auth_manager.verify_token(credentials.credentials)
    return payload

async def require_permission(permission: str):
    """Dependency to require specific permission"""
    
    def permission_checker(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user
    
    return permission_checker

# Protected endpoints
@app.post("/predict")
async def predict_protected(
    request: IDSPredictionRequest,
    user: dict = Depends(get_current_user)
):
    logger.info(f"Prediction request from user: {user['user_id']}")
    return await prediction_service.predict(request.features)

@app.get("/admin/stats")
async def admin_stats(
    user: dict = Depends(require_permission("admin"))
):
    return await get_comprehensive_stats()
```

### Rate Limiting with Redis

```python
import time
from fastapi import Request, HTTPException
import redis.asyncio as redis

class RateLimiter:
    """Redis-based rate limiting"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        
    async def check_rate_limit(self, key: str, limit: int, window: int) -> bool:
        """Check if request is within rate limit"""
        
        current_time = int(time.time())
        window_start = current_time - window
        
        pipeline = self.redis.pipeline()
        
        # Remove old entries
        pipeline.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        pipeline.zcard(key)
        
        # Add current request
        pipeline.zadd(key, {str(current_time): current_time})
        
        # Set expiry
        pipeline.expire(key, window)
        
        results = await pipeline.execute()
        request_count = results[1]
        
        return request_count < limit

rate_limiter = RateLimiter("redis://localhost:6379")

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks
    if request.url.path == "/health":
        return await call_next(request)
    
    # Create rate limit key
    client_ip = request.client.host
    rate_key = f"rate_limit:{client_ip}"
    
    # Check rate limit (100 requests per minute)
    if not await rate_limiter.check_rate_limit(rate_key, 100, 60):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
    
    return await call_next(request)
```

## 6. Comprehensive Monitoring and Alerting

### Health Check System

```python
import psutil
import torch
from datetime import datetime, timedelta
from typing import Dict, List

class HealthChecker:
    """Comprehensive health monitoring"""
    
    def __init__(self, prediction_service):
        self.prediction_service = prediction_service
        self.health_history = []
        
    async def comprehensive_health_check(self) -> Dict:
        """Perform comprehensive system health check"""
        
        health_status = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }
        
        # API responsiveness
        api_check = await self._check_api_responsiveness()
        health_status["checks"]["api"] = api_check
        
        # Model health
        model_check = await self._check_model_health()
        health_status["checks"]["model"] = model_check
        
        # System resources
        resource_check = self._check_system_resources()
        health_status["checks"]["resources"] = resource_check
        
        # Database connectivity
        db_check = await self._check_database_health()
        health_status["checks"]["database"] = db_check
        
        # Cache connectivity
        cache_check = await self._check_cache_health()
        health_status["checks"]["cache"] = cache_check
        
        # Determine overall status
        failed_checks = [k for k, v in health_status["checks"].items() 
                        if v["status"] != "healthy"]
        
        if failed_checks:
            health_status["overall_status"] = "unhealthy"
            health_status["failed_checks"] = failed_checks
        
        # Store health history
        self.health_history.append(health_status)
        if len(self.health_history) > 100:
            self.health_history.pop(0)
        
        return health_status
    
    async def _check_api_responsiveness(self) -> Dict:
        """Test API responsiveness with dummy prediction"""
        
        start_time = time.time()
        try:
            # Test prediction with dummy data
            dummy_features = [0.1] * 77
            result = await self.prediction_service.predict(dummy_features)
            
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "prediction_successful": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": (time.time() - start_time) * 1000
            }
    
    async def _check_model_health(self) -> Dict:
        """Check model loading and inference capability"""
        
        try:
            # Check if model is loaded
            if not hasattr(self.prediction_service, 'model') or self.prediction_service.model is None:
                return {
                    "status": "unhealthy",
                    "error": "Model not loaded"
                }
            
            # Check model parameters
            param_count = sum(p.numel() for p in self.prediction_service.model.parameters())
            
            # Check GPU availability if using CUDA
            gpu_available = torch.cuda.is_available()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory if gpu_available else 0
            
            return {
                "status": "healthy",
                "model_parameters": param_count,
                "gpu_available": gpu_available,
                "gpu_memory_gb": gpu_memory / (1024**3) if gpu_available else 0
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    def _check_system_resources(self) -> Dict:
        """Check system resource usage"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Determine health based on thresholds
        status = "healthy"
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = "unhealthy"
        elif cpu_percent > 80 or memory.percent > 80 or disk.percent > 80:
            status = "warning"
        
        return {
            "status": status,
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3)
        }
    
    async def _check_database_health(self) -> Dict:
        """Check database connectivity and performance"""
        
        try:
            # Test database connection
            start_time = time.time()
            stats = await prediction_storage.get_prediction_stats(hours=1)
            query_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "query_time_ms": query_time,
                "recent_predictions": stats.get("total_predictions", 0)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_cache_health(self) -> Dict:
        """Check cache connectivity and performance"""
        
        try:
            start_time = time.time()
            await cache_manager.redis.ping()
            ping_time = (time.time() - start_time) * 1000
            
            # Get cache statistics
            info = await cache_manager.redis.info()
            
            return {
                "status": "healthy",
                "ping_time_ms": ping_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": info.get("used_memory", 0) / (1024**2)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

# Enhanced health endpoint
health_checker = HealthChecker(prediction_service)

@app.get("/health/comprehensive")
async def comprehensive_health():
    """Comprehensive health check endpoint"""
    return await health_checker.comprehensive_health_check()

@app.get("/health/history")
async def health_history():
    """Get health check history"""
    return health_checker.health_history[-10:]  # Last 10 checks
```

## 7. Next Steps and Best Practices

### Production Deployment Checklist

```bash
# Production deployment checklist

# 1. Environment Configuration
export RLIDS_API_DEBUG=false
export RLIDS_API_LOG_LEVEL=INFO
export RLIDS_API_WORKERS=8
export RLIDS_API_MAX_BATCH_SIZE=1000

# 2. Security Setup
export RLIDS_API_SECRET_KEY="your-secure-secret-key"
export RLIDS_API_CORS_ORIGINS='["https://your-domain.com"]'
export RLIDS_API_RATE_LIMIT_ENABLED=true

# 3. Database Configuration
export DATABASE_URL="postgresql://user:pass@db-host:5432/rlids"
export REDIS_URL="redis://redis-host:6379"

# 4. Monitoring Setup
export PROMETHEUS_ENABLED=true
export PROMETHEUS_PORT=9090
export LOG_LEVEL=INFO

# 5. SSL/TLS Certificate
export SSL_CERT_PATH="/path/to/cert.pem"
export SSL_KEY_PATH="/path/to/key.pem"

# 6. Start production server
gunicorn api.main:app \
    --bind 0.0.0.0:8000 \
    --workers 8 \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile access.log \
    --error-logfile error.log \
    --log-level info \
    --preload \
    --max-requests 1000 \
    --max-requests-jitter 100
```

### Performance Monitoring Setup

```python
# monitoring/dashboard.py
import asyncio
import json
from datetime import datetime, timedelta

class PerformanceDashboard:
    """Real-time performance monitoring dashboard"""
    
    def __init__(self):
        self.metrics = {
            'requests_per_second': 0,
            'average_response_time': 0,
            'error_rate': 0,
            'active_connections': 0,
            'model_predictions_per_hour': 0,
            'system_resources': {}
        }
    
    async def collect_metrics(self):
        """Collect real-time metrics"""
        while True:
            try:
                # Collect API metrics
                api_stats = await self.get_api_statistics()
                
                # Collect system metrics
                system_stats = await self.get_system_statistics()
                
                # Update dashboard
                self.metrics.update({
                    **api_stats,
                    'system_resources': system_stats,
                    'last_updated': datetime.utcnow().isoformat()
                })
                
                # Send to monitoring systems
                await self.send_to_monitoring()
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def generate_performance_report(self) -> dict:
        """Generate comprehensive performance report"""
        
        # Get historical data
        historical_data = await self.get_historical_metrics(days=7)
        
        # Calculate trends
        trends = self.calculate_trends(historical_data)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(self.metrics, trends)
        
        return {
            'current_metrics': self.metrics,
            'trends': trends,
            'recommendations': recommendations,
            'report_generated': datetime.utcnow().isoformat()
        }

# Start monitoring dashboard
dashboard = PerformanceDashboard()
asyncio.create_task(dashboard.collect_metrics())
```

This tutorial provides a comprehensive foundation for advanced API usage patterns. Continue with the deployment tutorial for production-ready setups and scaling strategies.

## See Also

- [Getting Started Guide](../getting-started.md) - Basic API setup
- [Deployment Tutorial](deployment.md) - Production deployment strategies
- [Monitoring Tutorial](monitoring.md) - Advanced monitoring and alerting
- [API Reference](../api/index.md) - Complete API documentation
