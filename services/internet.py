"""Lightweight internet connectivity check."""

from __future__ import annotations

import logging
import socket
import time

import config

logger = logging.getLogger(__name__)

# very short cache so rapid questions do not stall on repeated TCP probes
_CACHE_TTL = 10.0
_last_check: float = 0.0
_last_result: bool = False


def _probe() -> bool:
    try:
        with socket.create_connection(
            (config.INTERNET_CHECK_HOST, config.INTERNET_CHECK_PORT),
            timeout=config.INTERNET_CHECK_TIMEOUT,
        ):
            return True
    except OSError:
        return False


def is_online(force: bool = False) -> bool:
    """Return True when the internet appears to be reachable.

    Uses a short TCP connect to ``8.8.8.8:53`` and never blocks for longer
    than ``INTERNET_CHECK_TIMEOUT`` seconds.
    """
    global _last_check, _last_result

    now = time.monotonic()
    if not force and (now - _last_check) < _CACHE_TTL:
        return _last_result

    try:
        _last_result = _probe()
    except Exception:  # pragma: no cover - defensive
        logger.exception("Internet check failed unexpectedly")
        _last_result = False

    _last_check = time.monotonic()
    if not _last_result:
        logger.info("Internet check: offline")
    return _last_result


def reset_cache() -> None:
    """Force the next call to ``is_online`` to perform a fresh probe."""
    global _last_check
    _last_check = 0.0
