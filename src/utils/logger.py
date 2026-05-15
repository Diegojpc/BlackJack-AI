"""
Logging utilities for Blackjack AI.

Provides structured logging with contextual information for full traceability
across training runs, evaluations, and agent decisions.
"""

import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.config import LOGS_DIR


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Path | None = None,
) -> logging.Logger:
    """
    Configure and return a logger with console and optional file handlers.

    Args:
        name: Logger name (typically __name__ of the calling module).
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_to_file: Whether to write logs to a file.
        log_dir: Directory for log files. Defaults to LOGS_DIR.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if setup_logger is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — always active
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler — optional, rotates per session
    if log_to_file:
        log_path = log_dir or LOGS_DIR
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_path / f"{name.replace('.', '_')}_{timestamp}.log",
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG+
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("Logger initialized: name=%s, level=%s, file=%s", name, logging.getLevelName(level), log_to_file)
    return logger


class MetricsCSVWriter:
    """
    Writes training metrics to a CSV file for later analysis.

    Usage:
        writer = MetricsCSVWriter("results/training_metrics.csv", ["episode", "reward", "epsilon"])
        writer.write_row({"episode": 1000, "reward": -0.05, "epsilon": 0.8})
    """

    def __init__(self, filepath: str | Path, fieldnames: list[str]) -> None:
        """
        Initialize CSV writer.

        Args:
            filepath: Path to the CSV file.
            fieldnames: Column headers for the CSV.
        """
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["timestamp"] + fieldnames
        self.logger = logging.getLogger(f"{__name__}.MetricsCSVWriter")

        # Write header if file doesn't exist
        if not self.filepath.exists():
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            self.logger.info("Created metrics CSV: %s with fields %s", self.filepath, self.fieldnames)

    def write_row(self, data: dict) -> None:
        """
        Append a single row of metrics to the CSV.

        Args:
            data: Dictionary mapping fieldnames to values. Timestamp is auto-added.
        """
        data["timestamp"] = datetime.now().isoformat()
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(data)

    def write_rows(self, rows: list[dict]) -> None:
        """Append multiple rows of metrics at once."""
        timestamp = datetime.now().isoformat()
        with open(self.filepath, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            for row in rows:
                row["timestamp"] = timestamp
                writer.writerow(row)
        self.logger.debug("Wrote %d rows to %s", len(rows), self.filepath)
