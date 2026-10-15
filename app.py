"""Vercel entrypoint.

Vercel's Python runtime looks for a FastAPI instance named ``app`` at a
supported entrypoint (``app.py``, ``index.py``, ``server.py``, ``main.py``,
``wsgi.py`` or ``asgi.py``).  This module re-exports the real application so
the whole project deploys as a single Vercel Function without changing local
behaviour.

Local Windows development and the PyInstaller build do not use this file.
"""

from webapp.app import app  # noqa: F401

__all__ = ["app"]
