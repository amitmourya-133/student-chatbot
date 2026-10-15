#!/usr/bin/env python3
"""Local web-app launcher for the Student AI Assistant.

Starts the existing FastAPI application on ``127.0.0.1`` (never the public
interface), picks a free local port, waits until the server is responding and
then opens the default browser.  Re-launching while the server is already
running re-opens the browser instead of starting a second server.

Run::

    python web_launcher.py [--port 8000] [--no-browser]
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn

import config
from webapp.app import app

try:
    import ctypes
    import ctypes.wintypes as wintypes
except ImportError:  # pragma: no cover - non-Windows only
    ctypes = None

HOST = "127.0.0.1"
_DEFAULT_PORT_RANGE = (8000, 8020)
_READY_TIMEOUT_S = 15.0
_MUTEX_NAME = "StudentAIAssistantWebLauncher"
_PORT_FILE = config.USER_DATA_DIR / "web_launcher.port"
_ERROR_ALREADY_EXISTS = 183


def _acquire_mutex() -> Optional[int]:
    """Return a held Windows mutex handle, or None if another instance owns it."""
    if ctypes is None:
        return -1  # non-Windows: no cross-process guard, rely on port probing
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    handle = kernel32.CreateMutexW(None, False, _MUTEX_NAME)
    if not handle:
        return None
    if ctypes.get_last_error() == _ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return None
    return int(handle)


def _release_mutex(handle: Optional[int]) -> None:
    if ctypes is None or not handle:
        return
    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(
        wintypes.HANDLE(handle)
    )


def _write_port(port: int) -> None:
    """Persist the chosen port so a second launch can find the running server."""
    try:
        config.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
        _PORT_FILE.write_text(str(port), encoding="utf-8")
    except OSError:
        logging.getLogger("web_launcher").warning(
            "Could not persist port file %s", _PORT_FILE
        )


def _clean_port_file() -> None:
    try:
        _PORT_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _read_running_port() -> Optional[int]:
    try:
        if not _PORT_FILE.exists():
            return None
        return int(_PORT_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        try:
            sock.bind((HOST, port))
            return False
        except OSError:
            return True


def _poll_ready(port: int, timeout: float = _READY_TIMEOUT_S) -> bool:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://{HOST}:{port}/api/health", timeout=0.5
            ) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.2)
    return False


def _open_browser_after_ready(port: int, no_browser: bool) -> None:
    if no_browser:
        return

    def _open() -> None:
        if _poll_ready(port):
            webbrowser.open(f"http://{HOST}:{port}")
        else:
            print(f"Server did not become ready in time. URL: http://{HOST}:{port}")

    threading.Thread(target=_open, daemon=True, name="browser-opener").start()


def _pick_port(preferred: Optional[int]) -> int:
    if preferred is None:
        start = _DEFAULT_PORT_RANGE[0]
    else:
        start = preferred
    for port in range(start, start + 100):
        if not _port_in_use(port):
            return port
    raise RuntimeError("Could not find a free local port. Close some apps and retry.")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Student AI Assistant web launcher")
    parser.add_argument("--port", type=int, default=None, help="preferred local port")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    mutex = _acquire_mutex()
    if mutex is None:
        # another instance is running: re-open its browser tab and exit
        running_port = _read_running_port()
        if running_port is not None and _poll_ready(
            running_port, timeout=min(2.0, _READY_TIMEOUT_S)
        ):
            url = f"http://{HOST}:{running_port}"
            print(f"Student AI Assistant is already running at {url}")
            if not args.no_browser:
                webbrowser.open(url)
            print("Opening the existing instance and exiting.")
            return 0
        # stale mutex/port file: the old process died uncleanly, take over
        print("A previous instance appears to have stopped, starting fresh.")
        _clean_port_file()

    try:
        port = _pick_port(args.port)
    except RuntimeError as exc:
        print(f"Startup error: {exc}")
        _release_mutex(mutex)
        return 1

    url = f"http://{HOST}:{port}"
    print("Student AI Assistant - local web app")
    print(f"URL: {url}")
    print("Press Ctrl+C to stop the server.")

    _write_port(port)
    _open_browser_after_ready(port, args.no_browser)

    try:
        uvicorn.run(
            app,
            host=HOST,
            port=port,
            log_level="info",
            access_log=False,
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except SystemExit as exc:
        # uvicorn maps its own shutdown onto SystemExit(0)
        if getattr(exc, "code", None) != 0:
            raise
        print("\nServer stopped.")
    finally:
        _clean_port_file()
        _release_mutex(mutex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())