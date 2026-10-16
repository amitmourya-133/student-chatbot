"""Central configuration for the Student AI Chatbot.

All values can be overridden through environment variables or a local
``.env`` file.  No secrets are ever hardcoded here.

Path model:
* Bundled read-only assets (``data/intents.json``, ``data/offline_kb.json``,
  ``models/*.pkl``, ``webapp/``) are resolved through ``resource_path`` and
  come from ``%LOCALAPPDATA%``-independent bundle locations when frozen.
* Writable user data (memory, chat log, feature stores, logs, user config)
  lives under ``user_data_dir()``, which is ``%LOCALAPPDATA%\\StudentAIAssistant``
  in packaged mode and ``<project>/data`` in development (or
  ``STUDENTAI_DATA_DIR`` when that environment variable is set).
* In development none of the existing paths change.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from utils.paths import is_frozen, resource_path, user_data_dir

try:  # python-dotenv is optional - the app must run without it
    from dotenv import load_dotenv

    # dev keeps reading the project's .env; packaged apps read the user copy
    load_dotenv(resource_path(".env"))
    load_dotenv(user_data_dir() / ".env")
except Exception:  # pragma: no cover - dotenv missing or broken
    pass


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip()


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = resource_path()  # project root in dev, bundle root when frozen
DATA_DIR = resource_path("data")
MODELS_DIR = resource_path("models")

INTENTS_FILE = DATA_DIR / "intents.json"
KNOWLEDGE_BASE_FILE = DATA_DIR / "offline_kb.json"

VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"
INTENT_MODEL_FILE = MODELS_DIR / "intent_model.pkl"

WEBAPP_DIR = resource_path("webapp")
STATIC_DIR = WEBAPP_DIR / "static"
TEMPLATES_DIR = WEBAPP_DIR / "templates"
INDEX_TEMPLATE = TEMPLATES_DIR / "index.html"

# --------------------------------------------------------------------------- #
# Writable user data (packaged: %LOCALAPPDATA%/StudentAIAssistant)
# --------------------------------------------------------------------------- #
USER_DATA_DIR = user_data_dir()
CHAT_LOG_FILE = USER_DATA_DIR / "chat_log.json"
if is_frozen():
    LOG_DIR = USER_DATA_DIR / "logs"  # writable user dir when packaged
else:
    LOG_DIR = BASE_DIR / "logs"  # dev keeps logs/ as before

# --------------------------------------------------------------------------- #
# Feature storage (all created lazily, all git-ignored user data)
# --------------------------------------------------------------------------- #
MEMORY_FILE = USER_DATA_DIR / "memory.json"
PROFILE_FILE = USER_DATA_DIR / "profile.json"
CONVERSATIONS_FILE = USER_DATA_DIR / "conversations.json"
PROGRESS_FILE = USER_DATA_DIR / "progress.json"
FEEDBACK_FILE = USER_DATA_DIR / "feedback.json"
ANALYTICS_FILE = USER_DATA_DIR / "analytics.json"
FLASHCARDS_FILE = USER_DATA_DIR / "flashcards.json"
PLANNER_FILE = USER_DATA_DIR / "planner.json"
DOCUMENTS_DIR = USER_DATA_DIR / "uploads"

# upload / document limits
MAX_DOCUMENT_MB = 10
MAX_DOCUMENT_PAGES = 100
DOCUMENT_CHUNK_CHARS = 1500
DOCUMENT_CHUNK_OVERLAP = 200

# --------------------------------------------------------------------------- #
# Gemini / Online mode
# --------------------------------------------------------------------------- #
GEMINI_API_KEY: Optional[str] = _env("GEMINI_API_KEY") or None
# Safe default model. ``gemini-2.5-flash`` was retired for new users (404) and
# ``gemini-3.8-flash`` intermittently returns 503 UNAVAILABLE (overload).
# ``gemini-3.5-flash`` is a current, generally-available Flash model suitable
# for a student assistant. Always overridable via the GEMINI_MODEL env var.
GEMINI_MODEL: str = _env("GEMINI_MODEL", "gemini-3.5-flash") or "gemini-3.5-flash"
GEMINI_TIMEOUT: int = int(_env("GEMINI_TIMEOUT", "20") or "20")
GEMINI_MAX_HISTORY: int = 10
# Transient provider errors (429 rate limit, 408, 5xx, timeout, connection) are
# retried at most this many extra times. Kept small to avoid retry loops and
# duplicate billing.
GEMINI_RETRIES: int = int(_env("GEMINI_RETRIES", "1") or "1")
# Exponential backoff with jitter between retries. ``GEMINI_MAX_BACKOFF`` bounds
# each wait (and caps any provider ``Retry-After``) so a single request stays
# well within the 20s request timeout and Vercel's function time budget.
GEMINI_BACKOFF_BASE: float = float(_env("GEMINI_BACKOFF_BASE", "0.5") or "0.5")
GEMINI_MAX_BACKOFF: float = float(_env("GEMINI_MAX_BACKOFF", "4") or "4")
# After a provider failure, the online provider is skipped for this many
# seconds (per process) while the user's selected mode stays ONLINE.
PROVIDER_COOLDOWN_SECONDS: float = float(
    _env("PROVIDER_COOLDOWN_SECONDS", "30") or "30"
)

SYSTEM_PROMPT = (
    "You are Student AI Assistant, a friendly and expert tutor for school and "
    "college students. You help with programming (Python, Java, C, JavaScript, "
    "HTML, CSS), mathematics, science, study planning, assignments, exam "
    "preparation, coding debugging and concept explanations. Keep answers "
    "concise, accurate and easy to understand. Use short examples when they "
    "help. If the user writes in Hinglish, reply in natural Hinglish/Hindi "
    "mixed with English."
)

# --------------------------------------------------------------------------- #
# Offline mode
# --------------------------------------------------------------------------- #
INTERNET_CHECK_HOST: str = _env("INTERNET_CHECK_HOST", "8.8.8.8")
INTERNET_CHECK_PORT: int = int(_env("INTERNET_CHECK_PORT", "53") or "53")
INTERNET_CHECK_TIMEOUT: float = float(_env("INTERNET_CHECK_TIMEOUT", "2.5") or "2.5")

KB_SIMILARITY_THRESHOLD: float = float(_env("KB_SIMILARITY_THRESHOLD", "0.30") or "0.30")

# --------------------------------------------------------------------------- #
# Chat / UI
# --------------------------------------------------------------------------- #
HISTORY_DISPLAY_LIMIT: int = 10
HISTORY_SAVE_LIMIT: int = 500

BOT_NAME = "Student AI Chatbot"
BOT_SUBTITLE = "Your Personal Study Assistant"

EXIT_WORDS = {"quit", "exit", "bye", "goodbye", "q", "exit()", "quit()"}


def has_api_key() -> bool:
    """Return True when a Gemini API key is configured."""
    return bool(GEMINI_API_KEY)


def summary() -> dict:
    """Non-sensitive configuration summary for diagnostics."""
    return {
        "base_dir": str(BASE_DIR),
        "gemini_model": GEMINI_MODEL,
        "api_key_configured": has_api_key(),
        "kb_threshold": KB_SIMILARITY_THRESHOLD,
        "internet_check": f"{INTERNET_CHECK_HOST}:{INTERNET_CHECK_PORT}",
    }
