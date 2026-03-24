"""
Logging utilities for RAG pipeline.

Provides centralized logging configuration with support for
console and file handlers, log levels, and formatting.
"""

import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging for the RAG pipeline.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to output to console
        format_string: Optional custom log format string

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logging(log_level="DEBUG", log_file="rag.log")
        >>> logger.info("Starting RAG pipeline")
    """
    # Default format string
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logger
    logger = logging.getLogger("ragwire")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Propagate to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__ in modules)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"ragwire.{name}")


class ColoredFormatter(logging.Formatter):
    """
    Colored log formatter for better console output.

    Adds ANSI color codes to log levels for easier reading.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        # Add color to log level
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"

        return super().format(record)


def setup_colored_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging with colored console output.

    Args:
        log_level: Logging level
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger = logging.getLogger("ragwire")
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers = []

    # Colored console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter(format_string))
    logger.addHandler(console_handler)

    # File handler (no colors)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string))
        logger.addHandler(file_handler)

    logger.propagate = False

    return logger
