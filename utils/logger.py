"""Central logging configuration.

* writes diagnostics to ``logs/chatbot.log``
* prints warnings/errors to stderr for the operator
* never logs API keys, passwords or other secrets
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

import config

LOG_DIR = config.LOG_DIR
LOG_FILE = LOG_DIR / "chatbot.log"

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure the root logger once and return the app logger."""
    global _configured

    root = logging.getLogger()
    if _configured:
        return logging.getLogger("chatbot")

    root.setLevel(level)

    console = logging.StreamHandler(stream=sys.stderr)
    # only CRITICAL messages reach the terminal: stack traces and raw
    # exceptions stay in the log file and are never shown to normal users
    console.setLevel(logging.CRITICAL)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(console)

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=512_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(_FORMAT, _DATEFMT))
        root.addHandler(file_handler)
    except Exception:  # pragma: no cover - unwritable filesystem
        root.warning("Could not open log file at %s", LOG_FILE)

    # third-party libraries are noisy at DEBUG
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)

    _configured = True
    return logging.getLogger("chatbot")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
