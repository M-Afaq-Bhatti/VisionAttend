"""
Logging configuration for VisionAttend.

Sets up structured logging with consistent formatting
across all modules.
"""

import logging
import sys
from src.config.settings import settings


def setup_logging(level: str = None) -> None:
    """
    Configure root logger with a consistent format.

    Args:
        level: Log level string (e.g., 'DEBUG', 'INFO'). Defaults to settings.
    """
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(log_level)

    # Clear existing handlers to avoid duplicates on re-init
    root.handlers.clear()
    root.addHandler(console_handler)

    # Suppress noisy third-party loggers
    for noisy in ["PIL", "matplotlib", "urllib3", "insightface", "onnxruntime"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured at level %s", log_level)
