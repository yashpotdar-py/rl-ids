"""Configuration settings for the FastAPI application."""

from pathlib import Path
from pydantic_settings import BaseSettings
from rl_ids.config import MODELS_DIR, PROCESSED_DATA_DIR


class APISettings(BaseSettings):
    """API configuration settings."""

    # API Settings
    app_name: str = "RL-IDS API"
    app_version: str = "1.2.0"
    debug: bool = False

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Model Settings
    model_path: Path = MODELS_DIR / "dqn_model_final.pt"
    data_path: Path = PROCESSED_DATA_DIR / "cicids2017_balanced.csv"

    # Performance Settings
    max_batch_size: int = 100
    prediction_timeout: float = 30.0

    # Logging Settings
    log_level: str = "INFO"
    log_format: str = "{time} | {level} | {message}"

    # CORS Settings
    cors_origins: list = ["*"]
    cors_methods: list = ["*"]
    cors_headers: list = ["*"]

    # Rate Limiting
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Health Check Settings
    health_check_timeout: float = 5.0

    class Config:
        env_prefix = "RLIDS_"
        env_file = ".env"


# Global settings instance
settings = APISettings()
