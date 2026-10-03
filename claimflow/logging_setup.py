"""
Logging for the whole app: console plus a rotating file in logs/.

A filter on every handler replaces known API keys with [REDACTED], so a key
cannot leak into a log line even if some library prints it.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable

LOGGER_NAME = "claimflow"
_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


class RedactingFilter(logging.Filter):
    def __init__(self, secrets: Iterable[str]):
        super().__init__()
        # Very short values would redact ordinary words, so ignore them.
        self._secrets = [s for s in secrets if s and len(s) >= 6]

    def filter(self, record: logging.LogRecord) -> bool:
        if self._secrets:
            message = record.getMessage()
            for secret in self._secrets:
                message = message.replace(secret, "[REDACTED]")
            record.msg = message
            record.args = None
        return True


def setup_logging(level: str = "INFO", log_dir: Path | None = None, secrets: Iterable[str] = ()) -> logging.Logger:
    """
    Safe to call many times. Streamlit re-runs scripts constantly, so we drop
    our own old handlers first instead of stacking duplicates.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    redactor = RedactingFilter(secrets)
    formatter = logging.Formatter(_FORMAT)

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        try:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            handlers.append(
                RotatingFileHandler(Path(log_dir) / "claimflow.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
            )
        except OSError:
            # A read-only folder (some hosts) must not stop the app from starting.
            logger.warning("Could not open a log file in %s; logging to the console only.", log_dir)

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.addFilter(redactor)
        logger.addHandler(handler)

    # Library chatter at INFO level hides our own lines.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
