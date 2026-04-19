"""Logging utility for scripts and trainer components."""

from __future__ import annotations

import logging
from pathlib import Path


def create_logger(name: str, log_dir: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create or fetch a configured logger.

    Args:
        name: Logger name.
        log_dir: Optional directory to save a file log.
        level: Logging verbosity.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
