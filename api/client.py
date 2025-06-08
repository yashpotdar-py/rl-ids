"""Example client for testing the RL-IDS API."""

import asyncio
import json
import random
import time
from typing import List, Dict, Any

import httpx
import pandas as pd
from loguru import logger


class IDSAPIClient:
    """Client for interacting with the RL-IDS API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client."""
        self.base_url = base_url
        self.session = httpx.AsyncClient(timeout=30.0)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        try:
            response = await self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        try:
            response = await self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    async def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make single prediction."""
        try:
            payload = {"features": features}
            response = await self.session.post(f"{self.base_url}/predict", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def predict_batch(self, batch_features: List[List[float]]) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        try:
            batch_payload = [{"features": features} for features in batch_features]
            response = await self.session.post(f"{self.base_url}/predict/batch", json=batch_payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    async def close(self):
        """Close the client session."""
        await self.session.aclose()


async def test_api_endpoints():
    """Test all API endpoints."""
    client = IDSAPIClient()
    
    try:
        logger.info("🚀 Starting API tests...")
        
        # Test health check
        logger.info("1. Testing health check...")
        health = await client.health_check()
        logger.info(f"Health status: {health['status']}")
        
        # Test model info
        logger.info("2. Testing model info...")
        model_info = await client.get_model_info()
        logger.info(f"Model: {model_info['model_name']} v{model_info['model_version']}")
        logger.info(f"Input features: {model_info['input_features']}")
        
        # Generate sample features based on model requirements
        num_features = model_info['input_features']
        sample_features = [random.uniform(-1, 1) for _ in range(num_features)]
        
        # Test single prediction
        logger.info("3. Testing single prediction...")
        prediction = await client.predict(sample_features)
        logger.info(f"Prediction: {prediction['predicted_class']} (confidence: {prediction['confidence']:.3f})")
        
        # Test batch prediction
        logger.info("4. Testing batch prediction...")
        batch_features = [
            [random.uniform(-1, 1) for _ in range(num_features)]
            for _ in range(3)
        ]
        batch_results = await client.predict_batch(batch_features)
        logger.info(f"Batch predictions: {len(batch_results)} results")
        
        for i, result in enumerate(batch_results):
            logger.info(f"  Batch {i+1}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        logger.success("✅ All API tests passed!")
        
    except Exception as e:
        logger.error(f"❌ API test failed: {e}")
        raise
    finally:
        await client.close()


async def benchmark_api_performance(num_requests: int = 100):
    """Benchmark API performance."""
    client = IDSAPIClient()
    
    try:
        logger.info(f"🏁 Starting performance benchmark with {num_requests} requests...")
        
        # Get model info for feature count
        model_info = await client.get_model_info()
        num_features = model_info['input_features']
        
        # Generate test data
        test_features = [
            [random.uniform(-1, 1) for _ in range(num_features)]
            for _ in range(num_requests)
        ]
        
        # Benchmark single requests
        logger.info("Benchmarking individual requests...")
        start_time = time.time()
        individual_times = []
        
        for i, features in enumerate(test_features):
            request_start = time.time()
            await client.predict(features)
            request_time = time.time() - request_start
            individual_times.append(request_time)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_requests} requests")
        
        total_time = time.time() - start_time
        avg_request_time = sum(individual_times) / len(individual_times)
        requests_per_second = num_requests / total_time
        
        logger.info(f"📊 Performance Results:")
        logger.info(f"  Total time: {total_time:.2f} seconds")
        logger.info(f"  Average request time: {avg_request_time*1000:.2f} ms")
        logger.info(f"  Requests per second: {requests_per_second:.2f}")
        logger.info(f"  Min request time: {min(individual_times)*1000:.2f} ms")
        logger.info(f"  Max request time: {max(individual_times)*1000:.2f} ms")
        
        # Test batch processing
        logger.info("Benchmarking batch processing...")
        batch_size = 10
        batch_data = [test_features[i:i+batch_size] for i in range(0, len(test_features), batch_size)]
        
        batch_start = time.time()
        for batch in batch_data:
            await client.predict_batch(batch)
        batch_total_time = time.time() - batch_start
        
        logger.info(f"  Batch processing time: {batch_total_time:.2f} seconds")
        logger.info(f"  Batch requests per second: {num_requests/batch_total_time:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Performance benchmark failed: {e}")
        raise
    finally:
        await client.close()


async def simulate_real_traffic():
    """Simulate real network traffic patterns."""
    client = IDSAPIClient()
    
    try:
        logger.info("🌐 Simulating real network traffic...")
        
        # Get model requirements
        model_info = await client.get_model_info()
        num_features = model_info['input_features']
        
        # Simulate different traffic patterns
        patterns = {
            "normal_browsing": lambda: [random.gauss(0, 0.5) for _ in range(num_features)],
            "file_download": lambda: [random.gauss(1, 0.3) if i < 5 else random.gauss(0, 0.5) 
                                    for i in range(num_features)],
            "streaming": lambda: [random.gauss(0.5, 0.2) if i < 10 else random.gauss(0, 0.3) 
                                for i in range(num_features)],
            "potential_attack": lambda: [random.gauss(2, 0.5) if i < 3 else random.gauss(-1, 0.3) 
                                       for i in range(num_features)]
        }
        
        results = {}
        
        for pattern_name, pattern_func in patterns.items():
            logger.info(f"Testing pattern: {pattern_name}")
            
            pattern_results = []
            for _ in range(20):  # Test each pattern 20 times
                features = pattern_func()
                prediction = await client.predict(features)
                pattern_results.append(prediction)
            
            # Analyze results
            attack_count = sum(1 for r in pattern_results if r['is_attack'])
            avg_confidence = sum(r['confidence'] for r in pattern_results) / len(pattern_results)
            
            results[pattern_name] = {
                "attack_rate": attack_count / len(pattern_results),
                "avg_confidence": avg_confidence,
                "predictions": pattern_results
            }
            
            logger.info(f"  Attack rate: {results[pattern_name]['attack_rate']:.2%}")
            logger.info(f"  Avg confidence: {results[pattern_name]['avg_confidence']:.3f}")
        
        logger.success("✅ Traffic simulation completed!")
        return results
        
    except Exception as e:
        logger.error(f"❌ Traffic simulation failed: {e}")
        raise
    finally:
        await client.close()


async def main():
    """Main function to run all tests."""
    logger.info("🎯 RL-IDS API Client Test Suite")
    logger.info("=" * 50)
    
    try:
        # Basic API tests
        await test_api_endpoints()
        
        logger.info("\n" + "=" * 50)
        
        # Performance benchmark
        await benchmark_api_performance(50)
        
        logger.info("\n" + "=" * 50)
        
        # Traffic simulation
        traffic_results = await simulate_real_traffic()
        
        # Save results
        results_file = "api_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(traffic_results, f, indent=2, default=str)
        
        logger.success(f"✅ All tests completed! Results saved to {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Test suite failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())
