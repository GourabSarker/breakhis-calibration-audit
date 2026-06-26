"""Logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from src.utils.io import ensure_parent_dir


def configure_logging(level: int = logging.INFO, *, log_path: Path | None = None) -> None:
    """Configure stdout logging and optionally mirror logs to a file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path is not None:
        ensure_parent_dir(log_path)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
