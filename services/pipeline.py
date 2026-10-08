"""Shared business-logic pipeline for the CLI and the GUI.

Owns the classifier, offline knowledge base, online (Gemini) calling,
online/offline decision making, graceful fallback and source tracking.

The CLI's :class:`~main.StudentChatbot` and the desktop GUI both delegate to
this class so behaviour stays identical everywhere. The pipeline never prints
and never writes directly to the console - callers render results.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass

import config
from nlp.classifier import IntentClassifier
from services.chat_history import ChatHistory
from services.gemini import GeminiError, generate_response
from services.internet import is_online
from services.knowledge_base import NO_MATCH_MESSAGE, KnowledgeBase

logger = logging.getLogger(__name__)

SOURCE_OFFLINE_KB = "OFFLINE_KB"
SOURCE_INTENT = "INTENT_MODEL"
SOURCE_GEMINI = "GEMINI"
SOURCE_DOCUMENT = "DOCUMENT"
SOURCE_HYBRID = "HYBRID"
SOURCE_FALLBACK = "FALLBACK"

SOURCES = {
    SOURCE_OFFLINE_KB,
    SOURCE_INTENT,
    SOURCE_GEMINI,
    SOURCE_DOCUMENT,
    SOURCE_HYBRID,
    SOURCE_FALLBACK,
}


@dataclass
class ReplyResult:
    """Safe result of one user message. Never carries secrets."""

    text: str
    source: str
    online: bool
    error: str = ""
    just_switched: bool = False


@dataclass
class StartupStatus:
    """Non-sensitive startup validation result."""

    intent_model: bool
    knowledge_base: bool
    kb_entries: int
    api_key: bool
    online: bool
    directories_ok: bool = True


class AssistantPipeline:
    """Stateless-per-message business logic (state: mode + chat history)."""

    def __init__(self, force_offline: bool = False) -> None:
        self.force_offline = force_offline
        self.classifier = IntentClassifier()
        self.kb = KnowledgeBase()
        self.history = ChatHistory()
        self.intent_responses: dict[str, list[str]] = {}
        self.online = False
        self.online_disabled = False  # flipped after a Gemini failure

    # ------------------------------------------------------------------ #
    # startup
    # ------------------------------------------------------------------ #
    def startup(self) -> StartupStatus:
        self.validate_directories()
        intent_ok = self.classifier.ensure_ready()
        kb_ok = self.kb.load()
        self.load_intent_responses()
        self.refresh_mode(initial=True)
        return StartupStatus(
            intent_model=intent_ok,
            knowledge_base=kb_ok,
            kb_entries=self.kb.size,
            api_key=config.has_api_key(),
            online=self.online,
        )

    def validate_directories(self) -> None:
        for path in (config.DATA_DIR, config.MODELS_DIR):
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.warning("Could not create directory %s", path)

    def load_intent_responses(self) -> None:
        try:
            with open(config.INTENTS_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            for intent in data.get("intents", []):
                tag = intent.get("tag")
                responses = intent.get("responses") or []
                if tag and responses:
                    self.intent_responses[tag] = responses
        except Exception:
            logger.warning("Could not load canned intent responses")

    # ------------------------------------------------------------------ #
    # mode
    # ------------------------------------------------------------------ #
    def refresh_mode(self, initial: bool = False) -> bool:
        if self.force_offline or self.online_disabled:
            available = False
        else:
            available = is_online() and config.has_api_key()
        self.online = available
        return self.online

    @property
    def mode(self) -> str:
        return "online" if self.online else "offline"

    # ------------------------------------------------------------------ #
    # reply
    # ------------------------------------------------------------------ #
    def reply(self, text: str) -> ReplyResult:
        """Answer ``text`` and never raise.

        Returns a :class:`ReplyResult`. When an online attempt fails, the
        pipeline falls back to the offline engine once and reports the safe
        error category so the caller can tell the user what happened.
        """
        if self.online:
            try:
                reply_text = self._online_reply(text)
                return ReplyResult(reply_text, SOURCE_GEMINI, True)
            except GeminiError as exc:
                logger.warning("Gemini unavailable: %s", exc)
                error = str(exc)
            except Exception:
                logger.exception("Unexpected online failure")
                error = "The online AI failed unexpectedly."

            self.online_disabled = True
            self.online = False
            result = self._offline_reply(text)
            return ReplyResult(
                result.text,
                result.source,
                False,
                error=error,
                just_switched=True,
            )
        return self._offline_reply(text)

    def _offline_reply(self, text: str) -> ReplyResult:
        try:
            prediction = self.classifier.predict(text)
        except Exception:
            logger.exception("Intent classification failed")
            prediction = {"intent": "fallback", "confidence": 0.0}
        intent = prediction.get("intent", "fallback")
        confidence = float(prediction.get("confidence", 0.0))
        logger.debug("intent=%s confidence=%.2f input=%r", intent, confidence, text)

        try:
            match = self.kb.find_answer(text, intent_hint=intent)
        except Exception:
            logger.exception("KB search failed")
            match = None
        if match is not None:
            return ReplyResult(match.answer, SOURCE_OFFLINE_KB, False)

        responses = self.intent_responses.get(intent)
        if intent != "fallback" and responses:
            return ReplyResult(random.choice(responses), SOURCE_INTENT, False)

        fallback_responses = self.intent_responses.get("fallback")
        if fallback_responses:
            return ReplyResult(random.choice(fallback_responses), SOURCE_FALLBACK, False)
        return ReplyResult(NO_MATCH_MESSAGE, SOURCE_FALLBACK, False)

    def _online_reply(self, text: str) -> str:
        pairs: list[dict] = []
        for entry in self.history.recent(limit=10):
            if entry.get("user"):
                pairs.append({"role": "user", "content": str(entry["user"])})
            if entry.get("assistant"):
                pairs.append(
                    {"role": "assistant", "content": str(entry["assistant"])}
                )
        return self.generate_online(text, pairs)

    def generate_online(self, message: str, history: list[dict]) -> str:
        """Hook kept separate so modes/documents can add context later."""
        return generate_response(message, history)