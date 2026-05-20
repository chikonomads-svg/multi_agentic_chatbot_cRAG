"""Logging configuration for the app.

This module configures both console and rotating file handlers. It also ensures
the logs directory exists.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv


def configure_logging(level: int = logging.INFO, log_file: str | None = None) -> None:
    """Configure root logger with console and rotating file handlers.

    If log_file is None the default is ./logs/app.log relative to the package
    root.
    """
    # Load environment variables from a parent .env if present (safe; does not commit keys)
    load_dotenv(find_dotenv())

    root = Path(__file__).resolve().parents[1]
    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if log_file is None:
        log_file = str(logs_dir / "app.log")

    # Basic config to ensure handlers are present
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    # Remove existing handlers to avoid duplicate logs in repeated runs
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    root_logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    fh = RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=3)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)



