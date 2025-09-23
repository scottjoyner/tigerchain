from __future__ import annotations

import logging
from typing import Optional


def configure_logging(level: int = logging.INFO) -> None:
    """Configure structured logging once for the process."""

    if logging.getLogger().handlers:
        # Logging already configured by the runtime (e.g. uvicorn)
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
