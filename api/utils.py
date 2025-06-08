"""Utility functions for the API."""

import asyncio
import time
from functools import wraps
from typing import Any, Callable, Dict, List
import json

from loguru import logger
import numpy as np
import pandas as pd


def async_timer(func: Callable) -> Callable:
    """Decorator to measure execution time of async functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
        
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}ms")
        return result
    
    return wrapper


def validate_features(features: List[float], expected_count: int) -> List[float]:
    """Validate and normalize feature inputs."""
    if not features:
        raise ValueError("Features list cannot be empty")
    
    if len(features) != expected_count:
        raise ValueError(f"Expected {expected_count} features, got {len(features)}")
    
    # Check for invalid values
    for i, feature in enumerate(features):
        if not isinstance(feature, (int, float)):
            raise ValueError(f"Feature at index {i} must be numeric")
        
        if np.isnan(feature) or np.isinf(feature):
            raise ValueError(f"Feature at index {i} contains invalid value (NaN or Inf)")
    
    return features


def normalize_features(features: List[float], feature_stats: Dict[str, Any] = None) -> List[float]:
    """Normalize features using pre-computed statistics."""
    if feature_stats is None:
        return features
    
    normalized = []
    for i, feature in enumerate(features):
        if f"feature_{i}" in feature_stats:
            stats = feature_stats[f"feature_{i}"]
            mean = stats.get("mean", 0)
            std = stats.get("std", 1)
            normalized_value = (feature - mean) / std if std != 0 else feature
            normalized.append(normalized_value)
        else:
            normalized.append(feature)
    
    return normalized


def create_response_metadata(processing_time_ms: float, model_version: str = "1.0.0") -> Dict[str, Any]:
    """Create metadata for API responses."""
    return {
        "processing_time_ms": round(processing_time_ms, 2),
        "model_version": model_version,
        "timestamp": pd.Timestamp.now().isoformat(),
        "api_version": "1.0.0"
    }


def format_prediction_confidence(confidence: float) -> str:
    """Format confidence score for human-readable output."""
    if confidence >= 0.9:
        return "Very High"
    elif confidence >= 0.7:
        return "High"
    elif confidence >= 0.5:
        return "Medium"
    elif confidence >= 0.3:
        return "Low"
    else:
        return "Very Low"


def get_class_description(class_id: int, class_names: List[str] = None) -> Dict[str, str]:
    """Get detailed description for predicted class."""
    if class_names is None:
        class_names = ["Normal", "Attack"]
    
    descriptions = {
        0: {
            "name": "Normal Traffic",
            "description": "Legitimate network traffic with no malicious activity detected",
            "risk_level": "Low",
            "recommended_action": "No action required"
        },
        1: {
            "name": "Attack Traffic",
            "description": "Potential malicious network activity detected",
            "risk_level": "High", 
            "recommended_action": "Investigate and potentially block traffic"
        }
    }
    
    return descriptions.get(class_id, {
        "name": class_names[class_id] if class_id < len(class_names) else "Unknown",
        "description": "Unknown traffic classification",
        "risk_level": "Unknown",
        "recommended_action": "Manual review required"
    })


def calculate_feature_importance_scores(features: List[float], feature_names: List[str] = None) -> Dict[str, float]:
    """Calculate simple feature importance scores based on values."""
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(features))]
    
    # Simple importance based on normalized absolute values
    abs_features = [abs(f) for f in features]
    max_val = max(abs_features) if abs_features else 1
    
    importance = {}
    for i, (name, value) in enumerate(zip(feature_names, abs_features)):
        importance[name] = round(value / max_val, 3) if max_val > 0 else 0.0
    
    # Sort by importance
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))


def create_prediction_summary(prediction_result: Dict[str, Any]) -> Dict[str, Any]:
    """Create a comprehensive prediction summary."""
    class_info = get_class_description(prediction_result["prediction"])
    confidence_level = format_prediction_confidence(prediction_result["confidence"])
    
    return {
        "prediction_summary": {
            "predicted_class": prediction_result["predicted_class"],
            "confidence_score": prediction_result["confidence"],
            "confidence_level": confidence_level,
            "is_malicious": prediction_result["is_attack"],
            "risk_assessment": class_info,
            "processing_time_ms": prediction_result["processing_time_ms"]
        },
        "technical_details": {
            "class_probabilities": prediction_result["class_probabilities"],
            "prediction_id": prediction_result.get("prediction", 0),
            "model_confidence": prediction_result["confidence"]
        }
    }


async def batch_process_with_timeout(items: List[Any], process_func: Callable, timeout: float = 30.0) -> List[Any]:
    """Process a batch of items with timeout protection."""
    tasks = [asyncio.create_task(process_func(item)) for item in items]
    
    try:
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        return results
    except asyncio.TimeoutError:
        logger.error(f"Batch processing timed out after {timeout} seconds")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        raise


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize objects to JSON, handling numpy types."""
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)
    
    return json.dumps(obj, cls=NumpyEncoder)


def log_prediction_metrics(prediction_result: Dict[str, Any], request_info: Dict[str, Any] = None):
    """Log prediction metrics for monitoring."""
    metrics = {
        "prediction": prediction_result["prediction"],
        "confidence": prediction_result["confidence"],
        "processing_time_ms": prediction_result["processing_time_ms"],
        "is_attack": prediction_result["is_attack"],
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    if request_info:
        metrics.update(request_info)
    
    logger.info(f"Prediction metrics: {safe_json_serialize(metrics)}")


def health_check_database_connection() -> Dict[str, Any]:
    """Perform health check for database connections if applicable."""
    # Placeholder for database health checks
    return {
        "database_connected": True,
        "connection_pool_size": 0,
        "active_connections": 0
    }


def get_system_resources() -> Dict[str, Any]:
    """Get current system resource usage."""
    import psutil
    
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage_percent": psutil.disk_usage('/').percent,
        "available_memory_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "process_count": len(psutil.pids())
    }
