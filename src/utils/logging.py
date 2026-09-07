"""Logging helpers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> logging.Logger:
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)
    return logging.getLogger("contra-evo")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
