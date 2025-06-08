"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class IDSPredictionRequest(BaseModel):
    """Request model for intrusion detection prediction."""

    features: List[float] = Field(
        ...,
        description="Network traffic features for prediction",
        min_items=1
    )

    @validator('features')
    def validate_features(cls, v):
        """Validate feature values."""
        if not v:
            raise ValueError("Features list cannot be empty")

        for feature in v:
            if not isinstance(feature, (int, float)):
                raise ValueError("All features must be numeric")

        return v

    class Config:
        json_schema_extra = {
            "example": {
                "features": [
                    0.123, 0.456, 0.789, 0.321, 0.654,
                    0.987, 0.147, 0.258, 0.369, 0.741
                ]
            }
        }


class IDSPredictionResponse(BaseModel):
    """Response model for intrusion detection prediction."""

    prediction: int = Field(...,
                            description="Predicted class (0=Normal, 1=Attack)")
    confidence: float = Field(...,
                              description="Confidence score of the prediction")
    class_probabilities: List[float] = Field(
        ..., description="Probability scores for each class")
    predicted_class: str = Field(...,
                                 description="Human-readable predicted class name")
    is_attack: bool = Field(...,
                            description="Boolean indicating if traffic is malicious")
    processing_time_ms: float = Field(...,
                                      description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Prediction timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "confidence": 0.85,
                "class_probabilities": [0.15, 0.85],
                "predicted_class": "Attack",
                "is_attack": True,
                "processing_time_ms": 12.5,
                "timestamp": "2025-06-08T10:30:45.123456"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Health check timestamp")
    details: Dict[str, Any] = Field(...,
                                    description="Additional health details")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-06-08T10:30:45.123456",
                "details": {
                    "model_loaded": True,
                    "memory_usage_mb": 256.7,
                    "uptime_seconds": 3600,
                    "predictions_served": 1234
                }
            }
        }


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    model_name: str = Field(..., description="Name of the loaded model")
    model_version: str = Field(..., description="Version of the model")
    model_type: str = Field(..., description="Type of the model (e.g., DQN)")
    input_features: int = Field(..., description="Number of input features")
    output_classes: int = Field(..., description="Number of output classes")
    training_episodes: Optional[int] = Field(
        None, description="Number of training episodes")
    model_size_mb: float = Field(..., description="Model size in megabytes")
    class_names: List[str] = Field(...,
                                   description="Names of the output classes")
    feature_importance: Optional[Dict[str, float]] = Field(
        None, description="Feature importance scores")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "DQN_IDS_Model",
                "model_version": "1.0.0",
                "model_type": "Deep Q-Network",
                "input_features": 78,
                "output_classes": 2,
                "training_episodes": 200,
                "model_size_mb": 2.5,
                "class_names": ["Normal", "Attack"],
                "feature_importance": {
                    "flow_duration": 0.15,
                    "total_fwd_packets": 0.12,
                    "packet_length_mean": 0.08
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    batch_data: List[IDSPredictionRequest] = Field(
        ...,
        description="List of prediction requests",
        max_items=100
    )

    class Config:
        json_schema_extra = {
            "example": {
                "batch_data": [
                    {
                        "features": [0.1, 0.2, 0.3, 0.4, 0.5]
                    },
                    {
                        "features": [0.6, 0.7, 0.8, 0.9, 1.0]
                    }
                ]
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data provided",
                "timestamp": "2025-06-08T10:30:45.123456"
            }
        }
