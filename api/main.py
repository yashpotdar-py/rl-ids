"""Main FastAPI application for RL-IDS service."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import pandas as pd

from rl_ids.config import MODELS_DIR, NORMALISED_DATA_FILE
from .models import IDSPredictionRequest, IDSPredictionResponse, HealthResponse, ModelInfoResponse
from .services import IDSPredictionService


# Global service instance
prediction_service: IDSPredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    global prediction_service
    
    logger.info("Starting RL-IDS API service...")
    
    # Initialize prediction service
    try:
        model_path = MODELS_DIR / "dqn_model_final.pt"
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        prediction_service = IDSPredictionService(model_path=model_path)
        await prediction_service.initialize()
        logger.success("✅ Prediction service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down RL-IDS API service...")
    if prediction_service:
        await prediction_service.cleanup()


# Create FastAPI app with lifespan
app = FastAPI(
    title="RL-based Intrusion Detection System API",
    description="RESTful API for real-time network intrusion detection using Deep Q-Learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "RL-based Intrusion Detection System API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if prediction service is available
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not initialized"
            )
        
        # Get service health status
        health_status = await prediction_service.get_health_status()
        
        return HealthResponse(
            status="healthy",
            timestamp=pd.Timestamp.now().isoformat(),
            details=health_status
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not initialized"
            )
        
        model_info = await prediction_service.get_model_info()
        return ModelInfoResponse(**model_info)
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.post("/predict", response_model=IDSPredictionResponse)
async def predict_intrusion(request: IDSPredictionRequest):
    """Predict network intrusion for given features."""
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not initialized"
            )
        
        # Make prediction
        prediction_result = await prediction_service.predict(request.features)
        
        return IDSPredictionResponse(
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            class_probabilities=prediction_result["class_probabilities"],
            predicted_class=prediction_result["predicted_class"],
            is_attack=prediction_result["is_attack"],
            processing_time_ms=prediction_result["processing_time_ms"],
            timestamp=pd.Timestamp.now().isoformat()
        )
        
    except ValueError as e:
        logger.warning(f"Invalid input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=list[IDSPredictionResponse])
async def predict_batch(requests: list[IDSPredictionRequest]):
    """Predict network intrusions for a batch of feature sets."""
    try:
        if prediction_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction service not initialized"
            )
        
        if len(requests) > 100:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Batch size too large. Maximum 100 requests allowed."
            )
        
        # Process batch predictions
        results = []
        for request in requests:
            prediction_result = await prediction_service.predict(request.features)
            results.append(IDSPredictionResponse(
                prediction=prediction_result["prediction"],
                confidence=prediction_result["confidence"],
                class_probabilities=prediction_result["class_probabilities"],
                predicted_class=prediction_result["predicted_class"],
                is_attack=prediction_result["is_attack"],
                processing_time_ms=prediction_result["processing_time_ms"],
                timestamp=pd.Timestamp.now().isoformat()
            ))
        
        return results
        
    except ValueError as e:
        logger.warning(f"Invalid batch input data: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid batch input data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error occurred"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
