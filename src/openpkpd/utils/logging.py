"""Structured logging for openpkpd."""

from __future__ import annotations

import logging
import sys
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Get a logger with openpkpd prefix."""
    return logging.getLogger(f"openpkpd.{name}")


def configure_logging(level: int = logging.INFO, verbose: bool = False) -> None:
    """Configure root openpkpd logger with a clean handler."""
    root = logging.getLogger("openpkpd")
    root.setLevel(level if not verbose else logging.DEBUG)

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)


def log_estimation_step(
    logger: logging.Logger,
    iteration: int,
    ofv: float,
    params: dict[str, Any] | None = None,
) -> None:
    """Log a standard estimation iteration line."""
    msg = f"Iter {iteration:6d}  OFV={ofv:15.6f}"
    if params:
        param_str = "  ".join(f"{k}={v:.4g}" for k, v in params.items())
        msg = f"{msg}  {param_str}"
    logger.info(msg)
