"""Cross-process safe JSON persistence helpers.

``atomic_json_write`` writes to a temporary file in the same directory and
then atomically renames it into place, so a peer process can never observe a
partially-written JSON file.  ``FileLock`` serialises writers for
read-modify-write flows (chat log append + memory update).

Both are small, broadly-compatible helpers: ``msvcrt.locking`` on Windows and
``fcntl.flock`` on POSIX.  If neither lock API is available (pure fallback),
the atomic rename still prevents torn files, only last-writer-wins applies.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:  # Windows
    import msvcrt

    _MS = True
except Exception:  # pragma: no cover - POSIX
    _MS = False

try:  # POSIX
    import fcntl

    _FCNTL = True
except Exception:  # pragma: no cover - Windows
    _FCNTL = False

_LOCK_TIMEOUT_S = 5.0
_LOCK_RETRY_S = 0.02


class FileLock:
    """Advisory cross-process lock for a sibling data file.

    Usage::

        with FileLock(target_path):
            read + mutate + atomic_json_write(...)
    """

    def __init__(self, target: Path, timeout: float = _LOCK_TIMEOUT_S) -> None:
        self.target = Path(target)
        self.lock_path = self.target.with_name(self.target.name + ".lock")
        self.timeout = timeout
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        if not _MS and not _FCNTL:
            return  # no locking backend - atomic writes still prevent torn files
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.lock_path), os.O_CREAT | os.O_RDWR)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if _MS:
                    msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    self.release()
                    raise TimeoutError(
                        f"Could not acquire lock on {self.lock_path}"
                    )
                time.sleep(_LOCK_RETRY_S)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if _MS:
                try:
                    os.lseek(self._fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            elif _FCNTL:
                try:
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(self._fd)
        except OSError:
            logger.exception("Could not close lock handle")
        finally:
            self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def atomic_json_write(path: Path, payload: Any) -> None:
    """Atomically write *payload* as JSON to *path*.

    A short retry loop absorbs transient Windows ``PermissionError`` values
    caused by antivirus scanners or a rename colliding with another process.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_path, path)
    except Exception:
        logger.exception("Atomic JSON write failed for %s", path)
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _replace_with_retry(tmp_path: Path, target: Path, attempts: int = 5) -> None:
    for attempt in range(attempts):
        try:
            os.replace(tmp_path, target)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.02 * (attempt + 1))