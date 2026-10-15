"""Path resolution for development and PyInstaller-packaged runs.

* ``resource_path(...)``  -> immutable bundled assets (data/, models/, webapp/)
* ``user_data_dir()``     -> writable per-user data (memory, logs, config)
* ``is_frozen()``         -> True inside a PyInstaller EXE

Bundled assets are read from ``sys._MEIPASS`` when frozen, otherwise from the
project root.  User data always lands in ``%LOCALAPPDATA%\\StudentAIAssistant``
in frozen mode (overridable with ``STUDENTAI_DATA_DIR`` for portability or
tests) and in ``<project>/data`` in development so nothing shifts for source
runs.  The install directory is never written to.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running inside a PyInstaller bundle."""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """Project root in dev; the bundled resource root when frozen."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Absolute path to an immutable bundled resource."""
    return app_root().joinpath(*parts)


def user_data_dir() -> Path:
    """Writable per-user data directory (never the install directory)."""
    override = os.environ.get("STUDENTAI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if not is_frozen():
        return app_root() / "data"
    local = os.environ.get("LOCALAPPDATA")
    base = Path(local) if local else (Path.home() / "AppData" / "Local")
    return base / "StudentAIAssistant"