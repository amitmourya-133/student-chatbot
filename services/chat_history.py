"""Persistent chat history stored as JSON in ``data/chat_log.json``."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


class ChatHistory:
    """Append-only JSON log of user/assistant exchanges.

    Every public method is defensive: a missing, unreadable or corrupted
    file never crashes the application.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or config.CHAT_LOG_FILE
        self._entries: list[dict] = []
        self._load()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.path.exists():
            self._entries = []
            self._ensure_file()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if isinstance(raw, list):
                self._entries = [entry for entry in raw if isinstance(entry, dict)]
            elif isinstance(raw, dict) and isinstance(raw.get("messages"), list):
                self._entries = [
                    entry for entry in raw["messages"] if isinstance(entry, dict)
                ]
            else:
                logger.warning("Chat log has unexpected structure; resetting")
                self._entries = []
        except json.JSONDecodeError:
            logger.warning("Chat log was corrupted - backing it up and starting fresh")
            self._backup_corrupt_file()
            self._entries = []
        except Exception:
            logger.exception("Could not read chat log")
            self._entries = []

    def _backup_corrupt_file(self) -> None:
        try:
            backup = self.path.with_suffix(".corrupt.json")
            self.path.replace(backup)
            logger.info("Corrupt chat log moved to %s", backup)
        except Exception:
            logger.exception("Could not back up corrupt chat log")

    def _ensure_file(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with open(self.path, "w", encoding="utf-8") as handle:
                    json.dump([], handle, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Could not initialise chat log file")

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = self._entries[-config.HISTORY_SAVE_LIMIT:]
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
        except Exception:
            logger.exception("Could not write chat log")

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def add(self, user_message: str, assistant_reply: str, mode: str) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user": str(user_message),
            "assistant": str(assistant_reply),
            "mode": str(mode),
        }
        self._entries.append(entry)
        self._save()
        return entry

    def recent(self, limit: int = config.HISTORY_DISPLAY_LIMIT) -> list[dict]:
        if limit <= 0:
            return []
        return self._entries[-limit:]

    def all(self) -> list[dict]:
        return list(self._entries)

    def clear(self) -> None:
        """Empty the in-memory log and rewrite the file."""
        self._entries = []
        self._save()

    def __len__(self) -> int:
        return len(self._entries)
