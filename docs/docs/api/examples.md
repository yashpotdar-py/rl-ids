# API Examples

This page provides comprehensive examples of using the RL-IDS API for various integration scenarios.

## Basic Usage Examples

### 1. Service Health Check

```bash
# Check if the API service is running
curl -X GET http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "details": {
    "model_loaded": true,
    "api_version": "1.2.0",
    "memory_usage": 45.6,
    "uptime": 3600
  }
}
```

### 2. Get Model Information

```bash
# Get information about the loaded model
curl -X GET http://localhost:8000/model/info

# Expected response
{
  "model_name": "DQN_IDS_Model",
  "model_type": "Deep Q-Network",
  "version": "1.2.0",
  "input_features": 78,
  "output_classes": 2,
  "class_names": ["BENIGN", "ATTACK"],
  "training_date": "2024-01-10T15:30:00Z",
  "accuracy": 0.953,
  "false_positive_rate": 0.047
}
```

### 3. Single Prediction

```bash
# Make a single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
      1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,
      2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0,
      3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0,
      4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0,
      5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0,
      6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0,
      7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8
    ]
  }'

# Expected response
{
  "prediction": {
    "class": "BENIGN",
    "confidence": 0.87,
    "probability": {
      "BENIGN": 0.87,
      "ATTACK": 0.13
    }
  },
  "timestamp": "2024-01-15T10:35:00Z",
  "processing_time": 0.045
}
```

### 4. Batch Predictions

```bash
# Make multiple predictions at once
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [0.1, 0.2, 0.3, ..., 7.8],
      [1.1, 1.2, 1.3, ..., 8.8],
      [2.1, 2.2, 2.3, ..., 9.8]
    ]
  }'

# Expected response
{
  "predictions": [
    {
      "class": "BENIGN",
      "confidence": 0.87,
      "probability": {"BENIGN": 0.87, "ATTACK": 0.13}
    },
    {
      "class": "ATTACK",
      "confidence": 0.92,
      "probability": {"BENIGN": 0.08, "ATTACK": 0.92}
    },
    {
      "class": "BENIGN",
      "confidence": 0.79,
      "probability": {"BENIGN": 0.79, "ATTACK": 0.21}
    }
  ],
  "timestamp": "2024-01-15T10:40:00Z",
  "processing_time": 0.098,
  "total_predictions": 3
}
```

## Python Integration Examples

### 1. Using Requests Library

```python
import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health():
    """Check if the API is healthy and ready."""
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return None

def get_model_info():
    """Get information about the loaded model."""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to get model info: {e}")
        return None

def make_prediction(features):
    """Make a single prediction."""
    try:
        payload = {"features": features}
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Prediction failed: {e}")
        return None

def make_batch_predictions(features_list):
    """Make multiple predictions at once."""
    try:
        payload = {"features": features_list}
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Batch prediction failed: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Check API health
    health = check_api_health()
    print(f"API Health: {health}")
    
    # Get model information
    model_info = get_model_info()
    print(f"Model Info: {model_info}")
    
    # Sample feature vector (78 features)
    sample_features = [0.1] * 78
    
    # Make single prediction
    prediction = make_prediction(sample_features)
    print(f"Prediction: {prediction}")
    
    # Make batch predictions
    batch_features = [sample_features] * 3
    batch_predictions = make_batch_predictions(batch_features)
    print(f"Batch Predictions: {batch_predictions}")
```

### 2. Using the Official Python Client

```python
from api.client import RLIDSClient
import asyncio

async def main():
    # Initialize client
    client = RLIDSClient("http://localhost:8000")
    
    try:
        # Check health
        health = await client.health_check()
        print(f"Health: {health}")
        
        # Get model info
        model_info = await client.get_model_info()
        print(f"Model: {model_info}")
        
        # Sample features
        features = [0.1] * 78
        
        # Single prediction
        prediction = await client.predict(features)
        print(f"Prediction: {prediction}")
        
        # Batch prediction
        batch_features = [features] * 5
        batch_predictions = await client.predict_batch(batch_features)
        print(f"Batch Results: {batch_predictions}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Real-time Network Monitoring Integration

```python
import asyncio
import time
from api.client import RLIDSClient
from rl_ids.make_dataset import extract_features

class NetworkMonitor:
    def __init__(self, api_url="http://localhost:8000"):
        self.client = RLIDSClient(api_url)
        self.running = False
    
    async def start_monitoring(self):
        """Start real-time network monitoring."""
        self.running = True
        print("Starting network monitoring...")
        
        try:
            while self.running:
                # Capture network data (simplified)
                network_data = self.capture_network_data()
                
                if network_data:
                    # Extract features
                    features = extract_features(network_data)
                    
                    # Make prediction
                    result = await self.client.predict(features)
                    
                    # Process result
                    await self.process_prediction(result, network_data)
                
                # Wait before next capture
                await asyncio.sleep(1.0)
                
        except KeyboardInterrupt:
            print("Monitoring stopped by user")
        except Exception as e:
            print(f"Monitoring error: {e}")
        finally:
            await self.client.close()
    
    def capture_network_data(self):
        """Capture network data (placeholder)."""
        # This would contain actual packet capture logic
        # For demonstration, return mock data
        return {
            'packets': 100,
            'bytes': 5000,
            'duration': 1.0,
            'protocols': ['TCP', 'UDP']
        }
    
    async def process_prediction(self, prediction, network_data):
        """Process prediction results."""
        if prediction['prediction']['class'] == 'ATTACK':
            confidence = prediction['prediction']['confidence']
            print(f"ALERT: Attack detected with {confidence:.2%} confidence")
            print(f"Network data: {network_data}")
            
            # Here you would implement alerting logic
            await self.send_alert(prediction, network_data)
        else:
            print(f"Normal traffic detected ({prediction['prediction']['confidence']:.2%})")
    
    async def send_alert(self, prediction, network_data):
        """Send alert for detected attack."""
        alert = {
            'timestamp': time.time(),
            'type': 'ATTACK_DETECTED',
            'confidence': prediction['prediction']['confidence'],
            'network_data': network_data
        }
        
        # Implement your alerting mechanism here
        print(f"Sending alert: {alert}")
    
    def stop_monitoring(self):
        """Stop the monitoring process."""
        self.running = False

# Usage example
if __name__ == "__main__":
    monitor = NetworkMonitor()
    asyncio.run(monitor.start_monitoring())
```

## Error Handling Examples

### 1. Handling API Errors

```python
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

def robust_prediction(features, max_retries=3):
    """Make prediction with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/predict",
                json={"features": features},
                timeout=30
            )
            
            # Check for HTTP errors
            if response.status_code == 422:
                print("Validation error: Invalid input format")
                return None
                
            response.raise_for_status()
            return response.json()
            
        except ConnectionError:
            print(f"Connection error (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                
        except Timeout:
            print(f"Request timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(1)
                
        except RequestException as e:
            print(f"Request failed: {e}")
            break
    
    print("All retry attempts failed")
    return None

# Example usage
features = [0.1] * 78
result = robust_prediction(features)
if result:
    print(f"Prediction successful: {result}")
else:
    print("Prediction failed after all retries")
```

### 2. Input Validation

```python
def validate_features(features):
    """Validate feature input before sending to API."""
    if not isinstance(features, list):
        raise ValueError("Features must be a list")
    
    if len(features) != 78:
        raise ValueError(f"Expected 78 features, got {len(features)}")
    
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)):
            raise ValueError(f"Feature {i} must be numeric, got {type(feature)}")
        
        if not (-1000 <= feature <= 1000):  # Reasonable bounds
            raise ValueError(f"Feature {i} value {feature} is out of reasonable bounds")
    
    return True

def safe_prediction(features):
    """Make prediction with input validation."""
    try:
        # Validate input
        validate_features(features)
        
        # Make prediction
        response = requests.post(
            "http://localhost:8000/predict",
            json={"features": features}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API error: {response.status_code} - {response.text}")
            return None
            
    except ValueError as e:
        print(f"Input validation error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Example usage
try:
    features = [0.1] * 78  # Valid input
    result = safe_prediction(features)
    
    invalid_features = [0.1] * 77  # Invalid input
    result = safe_prediction(invalid_features)  # Will show validation error
    
except Exception as e:
    print(f"Error: {e}")
```

## Performance Optimization Examples

### 1. Batch Processing for High Throughput

```python
import asyncio
import aiohttp
from typing import List, Dict

class BatchProcessor:
    def __init__(self, api_url: str, batch_size: int = 100):
        self.api_url = api_url
        self.batch_size = batch_size
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_large_dataset(self, features_list: List[List[float]]):
        """Process a large dataset in batches."""
        results = []
        
        for i in range(0, len(features_list), self.batch_size):
            batch = features_list[i:i + self.batch_size]
            
            try:
                batch_result = await self._process_batch(batch)
                results.extend(batch_result)
                
                print(f"Processed batch {i//self.batch_size + 1}, "
                      f"total processed: {len(results)}")
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Continue with next batch
                continue
        
        return results
    
    async def _process_batch(self, batch: List[List[float]]):
        """Process a single batch."""
        async with self.session.post(
            f"{self.api_url}/predict/batch",
            json={"features": batch}
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["predictions"]
            else:
                raise Exception(f"API error: {response.status}")

# Usage example
async def main():
    # Generate sample data
    large_dataset = [[0.1] * 78 for _ in range(1000)]
    
    async with BatchProcessor("http://localhost:8000", batch_size=50) as processor:
        results = await processor.process_large_dataset(large_dataset)
        
        print(f"Processed {len(results)} predictions")
        
        # Analyze results
        attack_count = sum(1 for r in results if r["class"] == "ATTACK")
        print(f"Detected {attack_count} attacks out of {len(results)} samples")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Connection Pooling and Reuse

```python
import aiohttp
import asyncio
from typing import Optional

class OptimizedClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        # Configure connection pool
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
        )
        
        # Configure timeout
        timeout = aiohttp.ClientTimeout(total=30, connect=5)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def predict(self, features: List[float]) -> Dict:
        """Make a single prediction with optimized connection."""
        async with self.session.post(
            f"{self.base_url}/predict",
            json={"features": features}
        ) as response:
            return await response.json()
    
    async def predict_many(self, features_list: List[List[float]]) -> List[Dict]:
        """Make multiple concurrent predictions."""
        tasks = []
        
        for features in features_list:
            task = asyncio.create_task(self.predict(features))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return valid_results

# Usage example
async def performance_test():
    features_list = [[0.1] * 78 for _ in range(100)]
    
    async with OptimizedClient("http://localhost:8000") as client:
        start_time = asyncio.get_event_loop().time()
        
        results = await client.predict_many(features_list)
        
        end_time = asyncio.get_event_loop().time()
        
        print(f"Processed {len(results)} predictions in {end_time - start_time:.2f} seconds")
        print(f"Average time per prediction: {(end_time - start_time) / len(results):.3f} seconds")

if __name__ == "__main__":
    asyncio.run(performance_test())
```

## Integration with Security Tools

### 1. SIEM Integration Example

```python
import json
import logging
from datetime import datetime
from api.client import RLIDSClient

class SIEMIntegration:
    def __init__(self, api_url: str, siem_endpoint: str):
        self.client = RLIDSClient(api_url)
        self.siem_endpoint = siem_endpoint
        self.logger = logging.getLogger(__name__)
    
    async def process_network_event(self, network_event: Dict):
        """Process network event and send to SIEM if threat detected."""
        try:
            # Extract features from network event
            features = self.extract_features(network_event)
            
            # Get prediction
            prediction = await self.client.predict(features)
            
            # Create enriched event
            enriched_event = {
                'timestamp': datetime.utcnow().isoformat(),
                'source_ip': network_event.get('source_ip'),
                'dest_ip': network_event.get('dest_ip'),
                'protocol': network_event.get('protocol'),
                'original_event': network_event,
                'ml_prediction': prediction,
                'threat_score': prediction['prediction']['confidence'],
                'classification': prediction['prediction']['class']
            }
            
            # Send to SIEM if attack detected
            if prediction['prediction']['class'] == 'ATTACK':
                await self.send_to_siem(enriched_event)
            
            return enriched_event
            
        except Exception as e:
            self.logger.error(f"Failed to process network event: {e}")
            return None
    
    def extract_features(self, network_event: Dict) -> List[float]:
        """Extract ML features from network event."""
        # This would contain actual feature extraction logic
        # For demonstration, return mock features
        return [0.1] * 78
    
    async def send_to_siem(self, event: Dict):
        """Send enriched event to SIEM."""
        try:
            # Format for SIEM (this example uses JSON)
            siem_event = {
                'timestamp': event['timestamp'],
                'event_type': 'ML_THREAT_DETECTION',
                'severity': self.calculate_severity(event),
                'source': 'RL-IDS',
                'details': event
            }
            
            # Send to SIEM endpoint
            # This would be actual SIEM API call
            self.logger.info(f"Sending to SIEM: {json.dumps(siem_event, indent=2)}")
            
        except Exception as e:
            self.logger.error(f"Failed to send to SIEM: {e}")
    
    def calculate_severity(self, event: Dict) -> str:
        """Calculate event severity based on confidence."""
        confidence = event['threat_score']
        
        if confidence >= 0.9:
            return 'HIGH'
        elif confidence >= 0.7:
            return 'MEDIUM'
        else:
            return 'LOW'

# Usage example
async def main():
    siem = SIEMIntegration(
        api_url="http://localhost:8000",
        siem_endpoint="https://siem.example.com/api/events"
    )
    
    # Sample network event
    network_event = {
        'source_ip': '192.168.1.100',
        'dest_ip': '10.0.0.1',
        'protocol': 'TCP',
        'port': 80,
        'packet_count': 1000,
        'byte_count': 50000
    }
    
    result = await siem.process_network_event(network_event)
    print(f"Processed event: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Monitoring and Alerting Examples

### 1. Health Monitoring

```python
import asyncio
import logging
from datetime import datetime, timedelta
from api.client import RLIDSClient

class HealthMonitor:
    def __init__(self, api_url: str, check_interval: int = 60):
        self.client = RLIDSClient(api_url)
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        self.alert_cooldown = {}
    
    async def start_monitoring(self):
        """Start continuous health monitoring."""
        self.logger.info("Starting health monitoring...")
        
        while True:
            try:
                await self.check_health()
                await asyncio.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Health monitoring stopped")
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def check_health(self):
        """Perform health check."""
        try:
            # Get health status
            health = await self.client.health_check()
            
            if health['status'] != 'healthy':
                await self.send_alert('UNHEALTHY', health)
                return
            
            # Check model status
            if not health['details'].get('model_loaded', False):
                await self.send_alert('MODEL_NOT_LOADED', health)
                return
            
            # Check memory usage
            memory_usage = health['details'].get('memory_usage', 0)
            if memory_usage > 80:  # 80% threshold
                await self.send_alert('HIGH_MEMORY', health)
            
            # Check response time
            start_time = asyncio.get_event_loop().time()
            await self.client.get_model_info()
            response_time = asyncio.get_event_loop().time() - start_time
            
            if response_time > 5.0:  # 5 second threshold
                await self.send_alert('SLOW_RESPONSE', {
                    'response_time': response_time,
                    'health': health
                })
            
            self.logger.info(f"Health check passed: {health['status']}")
            
        except Exception as e:
            await self.send_alert('API_ERROR', {'error': str(e)})
    
    async def send_alert(self, alert_type: str, details: Dict):
        """Send alert with cooldown to prevent spam."""
        now = datetime.utcnow()
        
        # Check cooldown
        if alert_type in self.alert_cooldown:
            last_alert = self.alert_cooldown[alert_type]
            if now - last_alert < timedelta(minutes=5):
                return  # Skip alert due to cooldown
        
        # Send alert
        alert = {
            'timestamp': now.isoformat(),
            'type': alert_type,
            'details': details,
            'severity': self.get_alert_severity(alert_type)
        }
        
        self.logger.error(f"ALERT: {json.dumps(alert, indent=2)}")
        
        # Update cooldown
        self.alert_cooldown[alert_type] = now
    
    def get_alert_severity(self, alert_type: str) -> str:
        """Get alert severity level."""
        severity_map = {
            'UNHEALTHY': 'CRITICAL',
            'MODEL_NOT_LOADED': 'CRITICAL',
            'API_ERROR': 'HIGH',
            'HIGH_MEMORY': 'MEDIUM',
            'SLOW_RESPONSE': 'LOW'
        }
        return severity_map.get(alert_type, 'MEDIUM')

# Usage example
if __name__ == "__main__":
    monitor = HealthMonitor("http://localhost:8000", check_interval=30)
    asyncio.run(monitor.start_monitoring())
```

These examples demonstrate various ways to integrate and use the RL-IDS API effectively in different scenarios, from simple predictions to complex production integrations with security tools and monitoring systems.