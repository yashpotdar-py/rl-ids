# Python Client

The RL-IDS Python client (`api/client.py`) provides a convenient interface for interacting with the RL-IDS API from Python applications.

## Overview

The `IDSAPIClient` class offers async methods for all API endpoints, connection management, and error handling. It's designed for high-performance applications that need to integrate threat detection capabilities.

## Installation

The client is included with the RL-IDS package. No additional installation is required.

## Basic Usage

### Initialize Client

```python
from api.client import IDSAPIClient

# Initialize with default settings
client = IDSAPIClient()

# Initialize with custom endpoint
client = IDSAPIClient("http://your-api-server:8000")
```

### Health Check

```python
import asyncio
from api.client import IDSAPIClient

async def check_health():
    client = IDSAPIClient()
    
    try:
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        print(f"Model Loaded: {health['details']['model_loaded']}")
        print(f"Uptime: {health['details']['uptime_seconds']} seconds")
    except Exception as e:
        print(f"Health check failed: {e}")
    finally:
        await client.close()

# Run the async function
asyncio.run(check_health())
```

### Single Prediction

```python
import asyncio
from api.client import IDSAPIClient

async def make_prediction():
    client = IDSAPIClient()
    
    try:
        # Example with 78 CICIDS2017 features
        features = [0.1, 0.2, 0.3] + [0.0] * 75  # 78 features total
        
        prediction = await client.predict(features)
        
        if prediction['is_attack']:
            print(f"🚨 Attack detected: {prediction['predicted_class']}")
            print(f"Confidence: {prediction['confidence']:.1%}")
        else:
            print("✅ Normal traffic detected")
            
    except Exception as e:
        print(f"Prediction failed: {e}")
    finally:
        await client.close()

asyncio.run(make_prediction())
```

### Batch Predictions

```python
import asyncio
from api.client import IDSAPIClient

async def batch_predictions():
    client = IDSAPIClient()
    
    try:
        # Create batch of feature sets
        batch_data = []
        for i in range(5):
            features = [0.1 * (i + 1), 0.2 * (i + 1), 0.3 * (i + 1)] + [0.0] * 75
            batch_data.append({"features": features})
        
        results = await client.predict_batch(batch_data)
        
        for i, result in enumerate(results):
            status = "Attack" if result['is_attack'] else "Normal"
            print(f"Sample {i+1}: {status} ({result['confidence']:.1%})")
            
    except Exception as e:
        print(f"Batch prediction failed: {e}")
    finally:
        await client.close()

asyncio.run(batch_predictions())
```

## Client Class Reference

### IDSAPIClient

```python
class IDSAPIClient:
    """Client for interacting with the RL-IDS API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=30.0)
```

### Methods

#### `health_check()`

Check API health status.

```python
async def health_check(self) -> Dict[str, Any]:
    """Check API health status."""
```

**Returns:**
```python
{
    "status": "healthy",
    "timestamp": "2025-06-27T10:30:00.123456",
    "details": {
        "model_loaded": True,
        "predictions_served": 1234,
        "uptime_seconds": 3600.5,
        "memory_usage_mb": 256.7
    }
}
```

**Raises:**
- `httpx.HTTPError` - If API is unreachable
- `httpx.HTTPStatusError` - If API returns error status

#### `get_model_info()`

Get detailed model information.

```python
async def get_model_info(self) -> Dict[str, Any]:
    """Get model information."""
```

**Returns:**
```python
{
    "model_name": "DQN_IDS_Model",
    "model_version": "1.2.0",
    "model_type": "Deep Q-Network",
    "input_features": 78,
    "output_classes": 15,
    "class_names": ["BENIGN", "Web Attack – Brute Force", ...],
    "model_size_mb": 2.5
}
```

#### `predict()`

Make single prediction.

```python
async def predict(self, features: List[float]) -> Dict[str, Any]:
    """Make single prediction."""
```

**Parameters:**
- `features`: List of 78 CICIDS2017 features

**Returns:**
```python
{
    "prediction": 9,
    "confidence": 0.87,
    "predicted_class": "DoS Hulk",
    "is_attack": True,
    "class_probabilities": [0.01, 0.02, ...],
    "processing_time_ms": 12.5,
    "timestamp": "2025-06-27T10:30:45.123456"
}
```

**Raises:**
- `ValueError` - If features list is invalid
- `httpx.HTTPStatusError` - If API returns error

#### `predict_batch()`

Make batch predictions.

```python
async def predict_batch(self, batch_data: List[Dict[str, List[float]]]) -> List[Dict[str, Any]]:
    """Make batch predictions."""
```

**Parameters:**
- `batch_data`: List of dictionaries with "features" key

**Returns:**
List of prediction results (same format as single prediction)

**Raises:**
- `ValueError` - If batch data is invalid
- `httpx.HTTPStatusError` - If API returns error

#### `close()`

Close the HTTP session.

```python
async def close(self):
    """Close the HTTP session."""
    await self.session.aclose()
```

## Advanced Usage

### Context Manager

Use the client as an async context manager for automatic cleanup:

```python
import asyncio
from api.client import IDSAPIClient

async def main():
    async with IDSAPIClient() as client:
        health = await client.health_check()
        print(f"API Status: {health['status']}")
        
        # Client will be automatically closed when exiting context

asyncio.run(main())
```

### Custom Timeout

Configure request timeouts:

```python
import httpx
from api.client import IDSAPIClient

# Custom timeout settings
client = IDSAPIClient()
client.session = httpx.AsyncClient(timeout=httpx.Timeout(60.0))  # 60 second timeout
```

### Error Handling

Comprehensive error handling:

```python
import httpx
from api.client import IDSAPIClient

async def robust_prediction(features):
    client = IDSAPIClient()
    
    try:
        prediction = await client.predict(features)
        return prediction
        
    except httpx.TimeoutException:
        print("Request timed out - API may be overloaded")
        return None
        
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 422:
            print("Invalid input data")
        elif e.response.status_code == 503:
            print("API service unavailable")
        else:
            print(f"HTTP error: {e.response.status_code}")
        return None
        
    except httpx.ConnectError:
        print("Cannot connect to API server")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
        
    finally:
        await client.close()
```

### Performance Testing

Benchmark API performance:

```python
import asyncio
import time
from api.client import IDSAPIClient

async def benchmark_api_performance(num_requests=100):
    """Benchmark API prediction performance."""
    client = IDSAPIClient()
    
    # Generate test features
    test_features = [0.1, 0.2, 0.3] + [0.0] * 75
    
    try:
        start_time = time.time()
        
        # Make concurrent requests
        tasks = [client.predict(test_features) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Calculate metrics
        successful_requests = sum(1 for r in results if not isinstance(r, Exception))
        total_time = end_time - start_time
        avg_time = total_time / successful_requests if successful_requests > 0 else 0
        rps = successful_requests / total_time if total_time > 0 else 0
        
        print(f"Performance Test Results:")
        print(f"  Total Requests: {num_requests}")
        print(f"  Successful: {successful_requests}")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Average Time: {avg_time:.3f}s")
        print(f"  Requests/Second: {rps:.1f}")
        
        return {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "total_time": total_time,
            "avg_time": avg_time,
            "rps": rps
        }
        
    finally:
        await client.close()

# Run benchmark
asyncio.run(benchmark_api_performance(100))
```
