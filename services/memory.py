"""Conversation memory: short-term context + explicit student facts.

Persists to ``data/memory.json``. Missing or corrupted data is backed up
(like the chat log) and never crashes the app.
Sensitive values (passwords, keys, bank details) are never stored.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

import config
from utils.filelock import FileLock, atomic_json_write

logger = logging.getLogger(__name__)

SESSION_LIMIT = 8

# --------------------------------------------------------------------------- #
# Keys we are willing to remember about a student
# --------------------------------------------------------------------------- #
VALID_KEYS = {
    "name",
    "subject",
    "subjects",
    "branch",
    "semester",
    "year",
    "college",
    "language",
    "difficulty",
}

# --------------------------------------------------------------------------- #
# Compiled regexes for extracting facts from free text
# --------------------------------------------------------------------------- #
# Language pattern: capture english / hindi / hinglish after "in" or "prefer .. in"
_LANG_PATTERN = re.compile(
    r"\b(?:prefer(?:red)?\s+)?(?:explain\s+)?in\s+"
    r"(?P<v>english|hindi|hinglish|hindi\s+english|english\s+hindi)\b",
    re.IGNORECASE,
)

# Difficulty pattern: capture beginner / intermediate / advanced etc.
_DIFF_PATTERN = re.compile(
    r"\b(?P<v>beginner|intermediate|advanced)\s+(?:level|difficulty)?\b|"
    r"\b(?:easy|medium|hard)\s+(?:difficulty|level|questions?)\b",
    re.IGNORECASE,
)

# --------------------------------------------------------------------------- #
# Fact-extraction patterns (key -> compiled regex)
# --------------------------------------------------------------------------- #
_FACT_PATTERNS: dict[str, re.Pattern] = {
    "name": re.compile(
        r"(?:my\s+)?name\s+is\s+(?P<v>[A-Za-z][A-Za-z .'-]{1,40})",
        re.IGNORECASE,
    ),
    "subjects": re.compile(
        r"(?:my\s+)?subjects?\s+(?:are|is)\s+(?P<v>[A-Za-z0-9 ,&+.'-]{2,80})",
        re.IGNORECASE,
    ),
    "subject": re.compile(
        r"(?:my\s+|favourite\s+|favorite\s+)?subject\s+(?:is|topic\s+is)?\s+"
        r"(?P<v>[A-Za-z][A-Za-z0-9 +().'-]{1,60})",
        re.IGNORECASE,
    ),
    "branch": re.compile(
        r"(?:my\s+)?(?:branch|stream|course)\s+is\s+(?P<v>[A-Za-z][A-Za-z0-9 /+.,'-]{1,40})",
        re.IGNORECASE,
    ),
    "semester": re.compile(
        r"(?:my\s+)?sem(?:ester)?\s+(?:is\s+)?(?P<v>\d{1,2})",
        re.IGNORECASE,
    ),
    "year": re.compile(
        r"(?:my\s+)?year(?:\s+is)?\s+(?P<v>\d{1,2})",
        re.IGNORECASE,
    ),
    "college": re.compile(
        r"(?:my\s+)?college\s+(?:is\s+)?(?P<v>[A-Za-z][^.,!?]{1,60})",
        re.IGNORECASE,
    ),
    "language": _LANG_PATTERN,
    "difficulty": _DIFF_PATTERN,
}

# --------------------------------------------------------------------------- #
# Question-patterns for direct memory queries (e.g. "What is my name?")
# --------------------------------------------------------------------------- #
_QUESTION_PATTERNS: dict[str, re.Pattern] = {
    "name": re.compile(r"what(?: is|'s| was)? my name\??", re.IGNORECASE),
    "subject": re.compile(r"what(?: is|'s| was)? my (?:favourite |favorite )?subject\??", re.IGNORECASE),
    "branch": re.compile(r"what(?: is|'s| was)? my (?:branch|stream|course)\??", re.IGNORECASE),
    "semester": re.compile(r"what(?: is|'s| was)? my sem(?:ester)?\??", re.IGNORECASE),
    "year": re.compile(r"what(?: is|'s| was)? my year\??", re.IGNORECASE),
    "college": re.compile(r"what(?: is|'s| was)? my college\??", re.IGNORECASE),
    "language": re.compile(r"what(?: is|'s| was)? my (?:preferred )?language\??", re.IGNORECASE),
    "difficulty": re.compile(r"what(?: is|'s| was)? my (?:preferred )?difficulty(?: level)?\??", re.IGNORECASE),
}

# --------------------------------------------------------------------------- #
# Words that must never be stored as facts (privacy guard)
# --------------------------------------------------------------------------- #
_SENSITIVE_WORDS = {
    "password", "otp", "pin", "mpin", "aadhaar", "adhaar", "pan card",
    "card number", "upi", "bank", "account number", "ssn", "voter",
    "passport", "secret", "token", "api key",
}


# --------------------------------------------------------------------------- #
# ConversationMemory – store / retrieve / manage short-term context + facts
# --------------------------------------------------------------------------- #
class ConversationMemory:
    """Remember student details, short-term exchanges, and answer lookup questions."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or config.MEMORY_FILE
        self.facts: dict[str, str] = {}
        self.session: list[str] = []
        self.disabled = False
        self._load()

    # --------------------------------------------------------------- #
    # persistence
    # --------------------------------------------------------------- #
    def _load(self) -> None:
        if not self.path.exists():
            self._ensure_file()
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if not isinstance(raw, dict):
                raise ValueError("memory payload is not an object")
            self.facts = {
                k: str(v)
                for k, v in (raw.get("facts") or {}).items()
                if k in VALID_KEYS
            }
            self.session = [str(x) for x in (raw.get("session") or [])][-SESSION_LIMIT:]
            self.disabled = bool(raw.get("disabled", False))
        except Exception:
            logger.warning("Memory data corrupted; backing up and starting fresh")
            self._backup_corrupt_file()
            self.facts = {}
            self.session = []
            self.disabled = False

    def _backup_corrupt_file(self) -> None:
        try:
            backup = self.path.with_suffix(".corrupt.json")
            self.path.rename(backup)
        except Exception:
            logger.exception("Could not back up corrupt memory file")

    def _ensure_file(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                self._save()
        except Exception:
            logger.exception("Could not initialise memory file")

    def _save(self) -> None:
        try:
            payload = {
                "facts": self.facts,
                "session": self.session[-SESSION_LIMIT:],
                "disabled": self.disabled,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with FileLock(self.path):
                atomic_json_write(self.path, payload)
        except Exception:
            logger.exception("Could not write memory file")

    # --------------------------------------------------------------- #
    # short-term session
    # --------------------------------------------------------------- #
    def record(self, user_message: str, assistant_reply: str) -> None:
        """Append one exchange to the bounded rolling session context."""
        if self.disabled:
            return
        self.session.extend([str(user_message), str(assistant_reply)])
        self.session = self.session[-SESSION_LIMIT:]
        self._save()

    def recent_context(self, limit: int = SESSION_LIMIT) -> list[str]:
        """Return the last *limit* user/assistant exchanges (raw strings)."""
        if limit <= 0:
            return []
        return self.session[-limit:]

    # --------------------------------------------------------------- #
    # explicit facts
    # --------------------------------------------------------------- #
    @staticmethod
    def sensitive(value: str) -> bool:
        """Return True when *value* looks like something we must never store."""
        lowered = value.lower()
        return any(word in lowered for word in _SENSITIVE_WORDS)

    def extract_facts(self, text: str) -> list[str]:
        """Pull explicit student facts out of free text.

        Returns the list of keys that were newly set. Sensitive values are
        silently discarded and never stored.
        """
        if self.disabled or not isinstance(text, str):
            return []
        stored: list[str] = []
        for key, pattern in _FACT_PATTERNS.items():
            match = pattern.search(text)
            if not match:
                continue
            value = match.group("v").strip().strip(".,!?").strip()
            if not value or self.sensitive(value):
                continue
            # Normalise language/difficulty to canonical tokens
            if key == "language":
                value = value.split()[0].lower()
            if key == "difficulty":
                lowered = value.lower()
                mapping = {
                    "beginner": "beginner",
                    "intermediate": "medium",
                    "advanced": "hard",
                    "easy": "easy",
                    "medium": "medium",
                    "hard": "hard",
                }
                value = mapping.get(lowered, value.lower())
            # Guard against value overflow
            self.facts[key] = value[:120]
            stored.append(key)
        if stored:
            self._save()
        return stored

    def get_fact(self, key: str) -> Optional[str]:
        return self.facts.get(key)

    def fact_keys(self) -> list[str]:
        return [k for k in VALID_KEYS if k in self.facts]

    def facts_dict(self) -> dict[str, str]:
        return dict(self.facts)

    def delete_fact(self, key: str) -> bool:
        existed = self.facts.pop(key, None) is not None
        if existed:
            self._save()
        return existed

    def clear(self) -> None:
        self.facts = {}
        self.session = []
        self._save()

    def set_enabled(self, enabled: bool) -> None:
        self.disabled = not enabled
        if self.disabled:
            self.clear()
        else:
            self._save()

    # --------------------------------------------------------------- #
    # answers to direct memory questions (offline)
    # --------------------------------------------------------------- #
    def answer_question(self, text: str) -> Optional[str]:
        """Answer questions like ``What is my name?`` from stored facts.

        Returns a friendly sentence or ``None`` when the answer is not known
        or memory is disabled.
        """
        if self.disabled or not self.facts:
            return None
        for key, pattern in _QUESTION_PATTERNS.items():
            if pattern.search(text):
                value = self.facts.get(key)
                if value:
                    pretty = {
                        "name": "name",
                        "subject": "subject",
                        "branch": "branch",
                        "semester": "semester",
                        "year": "year",
                        "college": "college",
                        "language": "preferred language",
                        "difficulty": "preferred difficulty",
                    }
                    return f"Your {pretty.get(key, key)} is {value}."
                return None
        return None

    # --------------------------------------------------------------- #
    # context lines for prompts (online system-injection)
    # --------------------------------------------------------------- #
    def context_lines(self) -> list[str]:
        """Non-sensitive lines describing remembered context (for prompts)."""
        if self.disabled or not self.facts:
            return []
        lines: list[str] = []
        order = ["name", "subject", "subjects", "branch", "year", "semester",
                 "college", "language", "difficulty"]
        for key in order:
            value = self.facts.get(key)
            if value:
                lines.append(f"- {key}: {value}")
        return lines


def answer_memory_question(text: str, memory: ConversationMemory) -> Optional[str]:
    """Convenience wrapper used by the CLI flow."""
    return memory.answer_question(text)