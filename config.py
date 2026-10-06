"""Central configuration for the Student AI Chatbot.

All values can be overridden through environment variables or a local
``.env`` file.  No secrets are ever hardcoded here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent

try:  # python-dotenv is optional - the app must run without it
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
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
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

INTENTS_FILE = DATA_DIR / "intents.json"
KNOWLEDGE_BASE_FILE = DATA_DIR / "offline_kb.json"
CHAT_LOG_FILE = DATA_DIR / "chat_log.json"

VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"
INTENT_MODEL_FILE = MODELS_DIR / "intent_model.pkl"

# --------------------------------------------------------------------------- #
# Gemini / Online mode
# --------------------------------------------------------------------------- #
GEMINI_API_KEY: Optional[str] = _env("GEMINI_API_KEY") or None
GEMINI_MODEL: str = _env("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
GEMINI_TIMEOUT: int = int(_env("GEMINI_TIMEOUT", "20") or "20")
GEMINI_MAX_HISTORY: int = 10

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
