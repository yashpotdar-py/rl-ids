"""Service layer for handling predictions and model management."""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import psutil
import os

from loguru import logger
import numpy as np
import pandas as pd
import torch

from rl_ids.agents.dqn_agent import DQNAgent, DQNConfig
from rl_ids.config import NORMALISED_DATA_FILE


class IDSPredictionService:
    """Service for handling IDS predictions using trained DQN model."""
    
    def __init__(self, model_path: Path):
        """Initialize the prediction service."""
        self.model_path = model_path
        self.agent: Optional[DQNAgent] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = ["Normal", "Attack"]  # Default class names
        self.feature_columns = []
        self.predictions_served = 0
        self.start_time = time.time()
        self.model_info = {}
        
        logger.info(f"Initializing prediction service with model: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    async def initialize(self) -> None:
        """Initialize the model and prepare for predictions."""
        try:
            # Load model checkpoint
            logger.info("Loading model checkpoint...")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config_dict = checkpoint["config"]
            
            # Create agent configuration
            config = DQNConfig(**config_dict)
            self.agent = DQNAgent(config=config)
            
            # Load model state
            self.agent.load_model(self.model_path)
            self.agent.model.to(self.device)
            self.agent.target_model.to(self.device)
            self.agent.epsilon = 0.0  # Pure greedy for inference
            
            # Set model to evaluation mode
            self.agent.model.eval()
            
            # Load feature information from dataset
            await self._load_feature_info()
            
            # Store model information
            self._prepare_model_info(config)
            
            logger.success("✅ Model loaded and initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def _load_feature_info(self) -> None:
        """Load feature information from the normalized dataset."""
        try:
            if NORMALISED_DATA_FILE.exists():
                # Load a small sample to get feature columns
                df_sample = pd.read_csv(NORMALISED_DATA_FILE, nrows=1)
                self.feature_columns = [col for col in df_sample.columns 
                                      if col not in ["Label", "Label_Original"]]
                logger.info(f"Loaded {len(self.feature_columns)} feature columns")
            else:
                logger.warning("Normalized data file not found. Using default feature info.")
                
        except Exception as e:
            logger.warning(f"Could not load feature info: {e}")
    
    def _prepare_model_info(self, config: DQNConfig) -> None:
        """Prepare model information dictionary."""
        model_size_bytes = sum(p.numel() * p.element_size() for p in self.agent.model.parameters())
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        self.model_info = {
            "model_name": "DQN_IDS_Model",
            "model_version": "1.0.0",
            "model_type": "Deep Q-Network",
            "input_features": config.state_dim,
            "output_classes": config.action_dim,
            "training_episodes": None,  # Could be extracted from checkpoint if stored
            "model_size_mb": round(model_size_mb, 2),
            "class_names": self.class_names,
            "feature_importance": None  # Could be computed if needed
        }
    
    async def predict(self, features: List[float]) -> Dict[str, Any]:
        """Make a prediction for given features."""
        start_time = time.time()
        
        try:
            # Validate input
            if len(features) != self.agent.config.state_dim:
                raise ValueError(
                    f"Expected {self.agent.config.state_dim} features, "
                    f"got {len(features)}"
                )
            
            # Convert to numpy array and then to tensor
            features_array = np.array(features, dtype=np.float32)
            features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                q_values = self.agent.model(features_tensor)
                probabilities = torch.softmax(q_values, dim=1)
                
                prediction = q_values.argmax().item()
                confidence = probabilities[0, prediction].item()
                class_probs = probabilities[0].cpu().numpy().tolist()
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Increment prediction counter
            self.predictions_served += 1
            
            # Prepare response
            result = {
                "prediction": prediction,
                "confidence": float(confidence),
                "class_probabilities": class_probs,
                "predicted_class": self.class_names[prediction],
                "is_attack": prediction != 0,  # Assuming 0 is normal traffic
                "processing_time_ms": round(processing_time, 2)
            }
            
            logger.debug(f"Prediction made: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get service health status."""
        try:
            # Memory usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)
            
            # GPU memory if available
            gpu_memory_mb = None
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            return {
                "model_loaded": self.agent is not None,
                "memory_usage_mb": round(memory_usage_mb, 1),
                "gpu_memory_mb": round(gpu_memory_mb, 1) if gpu_memory_mb else None,
                "uptime_seconds": round(uptime_seconds, 1),
                "predictions_served": self.predictions_served,
                "device": str(self.device),
                "model_path": str(self.model_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {
                "model_loaded": False,
                "error": str(e)
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        return self.model_info.copy()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up prediction service...")
        if self.agent:
            # Move models to CPU to free GPU memory
            if torch.cuda.is_available():
                self.agent.model.cpu()
                self.agent.target_model.cpu()
                torch.cuda.empty_cache()
        
        logger.info("✅ Prediction service cleanup completed")


class ModelManager:
    """Manager for handling multiple models and model versions."""
    
    def __init__(self, models_dir: Path):
        """Initialize model manager."""
        self.models_dir = models_dir
        self.loaded_models: Dict[str, IDSPredictionService] = {}
        
    async def load_model(self, model_name: str, model_path: Path) -> IDSPredictionService:
        """Load a model and add it to the manager."""
        if model_name in self.loaded_models:
            logger.warning(f"Model {model_name} already loaded")
            return self.loaded_models[model_name]
        
        service = IDSPredictionService(model_path)
        await service.initialize()
        self.loaded_models[model_name] = service
        
        logger.info(f"Model {model_name} loaded successfully")
        return service
    
    async def get_model(self, model_name: str) -> Optional[IDSPredictionService]:
        """Get a loaded model."""
        return self.loaded_models.get(model_name)
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            await self.loaded_models[model_name].cleanup()
            del self.loaded_models[model_name]
            logger.info(f"Model {model_name} unloaded")
            return True
        return False
    
    async def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.loaded_models.keys())
    
    async def cleanup_all(self) -> None:
        """Cleanup all loaded models."""
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
