"""Test suite for the RL-IDS API."""

import asyncio
import pytest
import httpx
from fastapi.testclient import TestClient

from api.main import app


class TestRLIDSAPI:
    """Test suite for the RL-IDS API."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "RL-based Intrusion Detection System API"
        assert data["version"] == "1.2.0"
        assert data["status"] == "running"
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "details" in data
        assert data["details"]["model_loaded"] is True
    
    def test_model_info(self):
        """Test the model info endpoint."""
        response = self.client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "DQN_IDS_Model"
        assert data["model_type"] == "Deep Q-Network"
        assert data["input_features"] == 78
        assert data["output_classes"] > 0
        assert isinstance(data["class_names"], list)
    
    def test_single_prediction(self):
        """Test single prediction endpoint."""
        # Create valid feature vector
        features = [0.1] * 78  # 78 features as required by the model
        
        response = self.client.post("/predict", json={"features": features})
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert "predicted_class" in data
        assert "is_attack" in data
        assert "processing_time_ms" in data
        assert "timestamp" in data
        
        # Validate data types
        assert isinstance(data["prediction"], int)
        assert isinstance(data["confidence"], float)
        assert isinstance(data["class_probabilities"], list)
        assert isinstance(data["is_attack"], bool)
        assert data["confidence"] >= 0 and data["confidence"] <= 1
    
    def test_batch_prediction(self):
        """Test batch prediction endpoint."""
        # Create batch of valid feature vectors
        batch_requests = [
            {"features": [0.1] * 78},
            {"features": [0.2] * 78},
            {"features": [0.3] * 78}
        ]
        
        response = self.client.post("/predict/batch", json=batch_requests)
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 3
        
        # Validate each prediction in the batch
        for prediction in data:
            assert "prediction" in prediction
            assert "confidence" in prediction
            assert "class_probabilities" in prediction
            assert "predicted_class" in prediction
            assert "is_attack" in prediction
    
    def test_invalid_feature_count(self):
        """Test prediction with invalid feature count."""
        # Wrong number of features
        features = [0.1] * 10  # Should be 78
        
        response = self.client.post("/predict", json={"features": features})
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
        assert "Expected 78 features" in data["detail"]
    
    def test_empty_features(self):
        """Test prediction with empty features."""
        response = self.client.post("/predict", json={"features": []})
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data
    
    def test_invalid_feature_types(self):
        """Test prediction with invalid feature types."""
        # Mix of valid and invalid feature types
        features = ["invalid"] + [0.1] * 77
        
        response = self.client.post("/predict", json={"features": features})
        assert response.status_code == 422
    
    def test_large_batch_rejection(self):
        """Test that large batches are rejected."""
        # Create batch larger than the limit (100)
        large_batch = [{"features": [0.1] * 78} for _ in range(101)]
        
        response = self.client.post("/predict/batch", json=large_batch)
        assert response.status_code == 413
        
        data = response.json()
        assert "Batch size too large" in data["detail"]
    
    def test_api_documentation(self):
        """Test that API documentation is accessible."""
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        response = self.client.get("/redoc")
        assert response.status_code == 200
        
        response = self.client.get("/openapi.json")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_async_client():
    """Test the async client functionality."""
    from api.client import IDSAPIClient
    
    client = IDSAPIClient("http://localhost:8000")
    
    try:
        # Test health check
        health = await client.health_check()
        assert health["status"] == "healthy"
        
        # Test model info
        model_info = await client.get_model_info()
        assert model_info["model_name"] == "DQN_IDS_Model"
        
        # Test prediction
        features = [0.1] * 78
        prediction = await client.predict(features)
        assert "prediction" in prediction
        assert "confidence" in prediction
        
    finally:
        await client.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
