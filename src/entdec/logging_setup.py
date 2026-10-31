"""Shared logging configuration.

The project uses :mod:`logging` everywhere - no ``print`` calls in library code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Third-party loggers that are extremely chatty at INFO level.
_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "urllib3",
    "neo4j",
    "neo4j.io",
    "neo4j.pool",
    "asyncio",
    "matplotlib",
    "streamlit",
    "watchfiles",
)

_configured = False


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once.

    Args:
        level: Log level name or numeric level.
        log_file: Optional file to also write logs to (created if missing).
        force: Reconfigure even when logging was already set up.

    Returns:
        The configured root logger.
    """
    global _configured

    root = logging.getLogger()
    if _configured and not force:
        return root

    numeric = logging.getLevelName(str(level).upper()) if isinstance(level, str) else level
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root.handlers.clear()
    root.setLevel(numeric)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream = logging.StreamHandler(stream=sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring logging with defaults on first use."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
