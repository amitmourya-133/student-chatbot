"""Shared utilities.

Submodules are importable on demand (``utils.paths``, ``utils.logger`` ...);
the package itself stays import-light so ``config`` can depend on it without
triggering a circular import through the logger.
"""

from __future__ import annotations

from typing import Any

__all__ = ["setup_logging", "get_logger", "logger"]


def __getattr__(name: str) -> Any:
    if name in ("setup_logging", "get_logger", "logger"):
        from utils import logger as _logger

        return getattr(_logger, name)
    raise AttributeError(f"module 'utils' has no attribute {name!r}")