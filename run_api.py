#!/usr/bin/env python3
"""
Script to run the RL-IDS FastAPI server.
"""

import argparse
import sys
from pathlib import Path

import uvicorn
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.config import settings


def main():
    """Main function to start the server."""
    parser = argparse.ArgumentParser(description="RL-IDS FastAPI Server")
    parser.add_argument("--host", default=settings.host, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=settings.workers, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--log-level", default=settings.log_level.lower(), help="Log level")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper())
    
    logger.info(f"Starting RL-IDS API server on {args.host}:{args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Log level: {args.log_level}")
    
    # Check if model exists
    if not settings.model_path.exists():
        logger.error(f"Model file not found: {settings.model_path}")
        logger.error("Please ensure the model is trained and saved before starting the API")
        sys.exit(1)
    
    # Start the server
    try:
        uvicorn.run(
            "api.main:app",
            host=args.host,
            port=args.port,
            workers=args.workers if not args.reload else 1,
            reload=args.reload,
            log_level=args.log_level,
            access_log=True,
            reload_dirs=["api", "rl_ids"] if args.reload else None
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
