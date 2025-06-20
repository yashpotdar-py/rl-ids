import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "cicids2017_processed.csv"
NORMALISED_DATA_FILE = PROCESSED_DATA_DIR / "cicids2017_normalised.csv"
BALANCED_DATA_FILE = PROCESSED_DATA_DIR / "cicids2017_balanced.csv"

TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = PROJ_ROOT / "models"
EPISODES_DIR = MODELS_DIR / "episodes"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Logging configuration
LOG_LEVEL = os.getenv("RLIDS_LOG_LEVEL", "INFO").upper()
DEBUG_MODE = os.getenv("RLIDS_DEBUG", "false").lower() == "true"

# Configure loguru
logger.remove(0)  # Remove default handler

# If tqdm is installed, configure loguru with tqdm.write
try:
    from tqdm import tqdm

    # Set log level based on environment
    if DEBUG_MODE:
        level = "DEBUG"
    else:
        level = LOG_LEVEL

    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    logger.info(f"📊 Logging configured - Level: {level}, Debug: {DEBUG_MODE}")

except ModuleNotFoundError:
    # Fallback if tqdm is not available
    if DEBUG_MODE:
        level = "DEBUG"
    else:
        level = LOG_LEVEL

    logger.add(
        lambda msg: print(msg, end=""),
        colorize=True,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
