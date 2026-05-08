"""Logging configuration using loguru."""

import sys

from loguru import logger


def get_logger(name: str = "vsde", level: str = "INFO") -> logger:
    """Configure and return a logger with VSDE defaults.

    - Colored output to stderr
    - Timestamps and log level
    - Module name in output
    """
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )
    return logger


# Default logger instance
_log = get_logger()
