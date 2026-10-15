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
import time
from dataclasses import dataclass
from pathlib import Path

import config
from fnmatch import fnmatch
from pypdf import PdfReader
from nlp.classifier import IntentClassifier
from nlp.preprocessing import preprocess
from services.chat_history import ChatHistory
from services.gemini import GeminiError, generate_response
from services.internet import is_online
from services.knowledge_base import (
    GENERIC_TERMS,
    NO_MATCH_MESSAGE,
    KnowledgeBase,
)
from services.memory import ConversationMemory
from services.response_policy import DEFAULT_STUDY_MODE, compose_system

logger = logging.getLogger(__name__)

SOURCE_OFFLINE_KB = "OFFLINE_KB"
SOURCE_INTENT = "INTENT_MODEL"
SOURCE_GEMINI = "GEMINI"
SOURCE_DOCUMENT = "DOCUMENT"
SOURCE_HYBRID = "HYBRID"
SOURCE_FALLBACK = "FALLBACK"
SOURCE_MEMORY = "MEMORY"

SOURCES = {
    SOURCE_OFFLINE_KB,
    SOURCE_INTENT,
    SOURCE_GEMINI,
    SOURCE_DOCUMENT,
    SOURCE_HYBRID,
    SOURCE_FALLBACK,
    SOURCE_MEMORY,
}

# Provider status is *separate* from the user-selected mode. A user can be in
# ONLINE mode while the provider itself is temporarily cooling down after a
# failure - that must never silently flip the mode to offline.
PROVIDER_OK = "ok"
PROVIDER_ERROR = "error"
PROVIDER_UNCONFIGURED = "unconfigured"
PROVIDER_UNREACHABLE = "unreachable"
PROVIDER_OFFLINE = "offline"

# Honest single message used when an online attempt failed *and* no local
# answer exists. Never concatenated with an unrelated template answer.
PROVIDER_FALLBACK_MESSAGE = (
    "The online assistant is temporarily unavailable, and I don't have a "
    "local answer for that yet. Please try again in a moment."
)

# Only these intents map to academic *content* answers. A canned content answer
# is safe only when the query shares a real topic word with the intent; social
# intents (greet, thanks, help, ...) are always fine to answer directly.
GROUNDED_INTENTS = frozenset({"code", "science", "math", "study"})


@dataclass
class ReplyResult:
    """Safe result of one user message. Never carries secrets."""

    text: str
    source: str
    online: bool
    error: str = ""
    fallback_used: bool = False
    provider_status: str = PROVIDER_OFFLINE
    request_id: str = ""


@dataclass
class StartupStatus:
    """Non-sensitive startup validation result."""

    intent_model: bool
    knowledge_base: bool
    kb_entries: int
    api_key: bool
    online: bool
    provider_status: str = PROVIDER_OFFLINE
    directories_ok: bool = True


class AssistantPipeline:
    """Stateless-per-message business logic (state: mode + chat history)."""

    def __init__(self, force_offline: bool = False, memory_path: Optional[Path] = None) -> None:
        self.force_offline = force_offline
        self.classifier = IntentClassifier()
        self.kb = KnowledgeBase()
        self.history = ChatHistory()
        self.memory = ConversationMemory(path=memory_path)
        self.study_mode = DEFAULT_STUDY_MODE  # Phase 3 switches this
        self.language = "english"  # Phase 12 switches this
        self.style = "normal"  # Phase 15 switches this
        self.intent_responses: dict[str, list[str]] = {}
        self.intent_tokens: dict[str, frozenset[str]] = {}
        self.online = False
        # True while the provider is cooling down after a failure. Kept as a
        # plain attribute (tests set it directly); it no longer implies the
        # user selected offline mode.
        self.online_disabled = False
        self.last_provider_error = ""
        self._provider_cooldown_until = 0.0
        self._request_counter = 0

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
            provider_status=self._compute_provider_status(),
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
            responses: dict[str, list[str]] = {}
            tokens: dict[str, frozenset[str]] = {}
            for intent in data.get("intents", []):
                tag = intent.get("tag")
                if not tag:
                    continue
                canned = intent.get("responses") or []
                if canned:
                    responses[tag] = canned
                # Content tokens from the *training patterns* let us refuse a
                # canned answer that shares no meaningful word with the query
                # (e.g. the code intent answering a physics question).
                bag: set[str] = set()
                for pattern in intent.get("patterns") or []:
                    for token in preprocess(pattern).split():
                        if token and token not in GENERIC_TERMS:
                            bag.add(token)
                tokens[tag] = frozenset(bag)
            self.intent_responses = responses
            self.intent_tokens = tokens
        except Exception:
            logger.warning("Could not load canned intent responses")

    # ------------------------------------------------------------------ #
    # mode
    # ------------------------------------------------------------------ #
    def _expire_cooldown(self) -> None:
        """Clear a stale provider cooldown once its window has elapsed."""
        if self.online_disabled and time.monotonic() >= self._provider_cooldown_until:
            self.online_disabled = False
            self.last_provider_error = ""

    def reset_provider_cooldown(self) -> None:
        """Allow an immediate provider retry (e.g. user toggles online)."""
        self.online_disabled = False
        self.last_provider_error = ""
        self._provider_cooldown_until = 0.0

    def _compute_provider_status(self) -> str:
        """Describe the *provider*, independent of the selected mode."""
        if not self.online:
            return PROVIDER_OFFLINE
        if not config.has_api_key():
            return PROVIDER_UNCONFIGURED
        if self.online_disabled:
            return PROVIDER_ERROR
        if not is_online():
            return PROVIDER_UNREACHABLE
        return PROVIDER_OK

    def refresh_mode(self, initial: bool = False) -> bool:
        self._expire_cooldown()
        if self.force_offline:
            available = False
        else:
            available = is_online() and config.has_api_key()
        self.online = available
        return self.online

    @property
    def mode(self) -> str:
        return "online" if self.online else "offline"

    @property
    def provider_status(self) -> str:
        self._expire_cooldown()
        return self._compute_provider_status()

    # ------------------------------------------------------------------ #
    # reply
    # ------------------------------------------------------------------ #
    def reply(self, text: str) -> ReplyResult:
        """Answer ``text`` and never raise.

        The user-selected mode is preserved: a provider failure enters a
        short cooldown and falls back to the offline engine for *this*
        message, but the mode stays ONLINE so the provider is retried once
        the cooldown expires.
        """
        self._expire_cooldown()
        self._request_counter += 1
        request_id = f"{int(time.time())}-{self._request_counter}"

        if self.online and not self.online_disabled:
            started = time.monotonic()
            try:
                reply_text = self._online_reply(text)
            except GeminiError as exc:
                error = str(exc)
                logger.warning(
                    "request_id=%s provider=failed category=%r",
                    request_id,
                    error,
                )
            except Exception:
                logger.exception("request_id=%s unexpected online failure", request_id)
                error = "The online AI failed unexpectedly."
            else:
                logger.info(
                    "request_id=%s mode=%s source=%s provider=%s latency=%.2fs",
                    request_id,
                    self.mode,
                    SOURCE_GEMINI,
                    PROVIDER_OK,
                    time.monotonic() - started,
                )
                return ReplyResult(
                    reply_text,
                    SOURCE_GEMINI,
                    online=True,
                    provider_status=PROVIDER_OK,
                    request_id=request_id,
                )

            self._mark_provider_error(error)
            result = self._offline_reply(text)
            logger.info(
                "request_id=%s mode=%s source=%s provider=%s fallback=true",
                request_id,
                self.mode,
                result.source,
                PROVIDER_ERROR,
            )
            if result.source == SOURCE_FALLBACK:
                # No local answer exists - use one honest message instead of a
                # generic canned template that would look unrelated.
                return ReplyResult(
                    PROVIDER_FALLBACK_MESSAGE,
                    SOURCE_FALLBACK,
                    online=False,
                    error=error,
                    fallback_used=True,
                    provider_status=PROVIDER_ERROR,
                    request_id=request_id,
                )
            return ReplyResult(
                result.text,
                result.source,
                online=False,
                error=error,
                fallback_used=True,
                provider_status=PROVIDER_ERROR,
                request_id=request_id,
            )

        result = self._offline_reply(text)
        status = PROVIDER_OFFLINE if not self.online else self._compute_provider_status()
        logger.info(
            "request_id=%s mode=%s source=%s provider=%s",
            request_id,
            self.mode,
            result.source,
            status,
        )
        return ReplyResult(
            result.text,
            result.source,
            online=False,
            fallback_used=False,
            provider_status=status,
            request_id=request_id,
        )

    def _mark_provider_error(self, error: str) -> None:
        self.last_provider_error = error
        self.online_disabled = True
        self._provider_cooldown_until = (
            time.monotonic() + config.PROVIDER_COOLDOWN_SECONDS
        )

    def _intent_is_grounded(self, intent: str, text: str) -> bool:
        """True when the query shares a content word with the intent patterns.

        Stops the canned per-intent answers from replying to off-topic
        questions that merely landed in an intent by accident.
        """
        canned_tokens = self.intent_tokens.get(intent)
        if not canned_tokens:
            return True  # no data - do not block a legitimate answer
        query_tokens = {
            token
            for token in preprocess(text).split()
            if token and token not in GENERIC_TERMS
        }
        if not query_tokens:
            return False
        return bool(query_tokens & canned_tokens)

    def _offline_reply(self, text: str) -> ReplyResult:
        # 1) direct memory question, e.g. "What is my name?"
        remembered = self.memory.answer_question(text)
        if remembered:
            return ReplyResult(remembered, SOURCE_MEMORY, False)

        # 2) remember explicit facts while offline ("my subject is DBMS")
        self.memory.extract_facts(text)

        try:
            prediction = self.classifier.predict(text)
        except Exception:
            logger.exception("Intent classification failed")
            prediction = {"intent": "fallback", "confidence": 0.0}
        intent = prediction.get("intent", "fallback")
        confidence = float(prediction.get("confidence", 0.0))
        logger.debug("intent=%s confidence=%.2f input=%r", intent, confidence, text)

        def lookup(query: str) -> object | None:
            try:
                return self.kb.find_answer(query, intent_hint=intent)
            except Exception:
                logger.exception("KB search failed")
                return None

        match = lookup(text)
        if match is not None:
            return ReplyResult(match.answer, SOURCE_OFFLINE_KB, False)

        # 3) memory-assisted retry: subject context often unlocks a KB answer
        # Use only subject fact; language effects happen online via system prompt.
        subject = self.memory.get_fact("subject")
        if subject:
            memory_query = f"{subject} {text}"
            if memory_query != text:
                match = lookup(memory_query)
                if match is not None:
                    return ReplyResult(match.answer, SOURCE_OFFLINE_KB, False)

        # 4) canned intent answer, but only when it actually relates to the
        #    question - otherwise an off-topic template would be misleading.
        responses = self.intent_responses.get(intent)
        if intent != "fallback" and responses:
            if intent not in GROUNDED_INTENTS or self._intent_is_grounded(intent, text):
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
        system = compose_system(
            memory_lines=self.memory.context_lines(),
            study_mode=self.study_mode,
            language=self.language,
            style=self.style,
        )
        return generate_response(message, history, system_instruction=system)

    # ------------------------------------------------------------------ #
    # PDF assistant
    # ------------------------------------------------------------------ #
    def load_pdf(self, file_path: str) -> dict:
        """Extract text from a PDF file and return metadata.

        Returns a dict with keys: pages, text, subject_hints.
        The extracted text is stored in memory for later querying.
        """
        try:
            reader = PdfReader(file_path)
            num_pages = len(reader.pages)
            all_text = ""
            subject_hints = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n--- Page {i+1} ---\n{page_text}"
                    # Try to extract subject hints from page text
                    text_lower = page_text.lower()
                    if "mathematics" in text_lower or "math" in text_lower:
                        subject_hints.append("mathematics")
                    if "physics" in text_lower:
                        subject_hints.append("physics")
                    if "chemistry" in text_lower:
                        subject_hints.append("chemistry")
                    if "programming" in text_lower or "python" in text_lower:
                        subject_hints.append("programming")
            # Store the extracted text in memory for querying
            if all_text.strip():
                self.memory.extract_facts(all_text)
            return {
                "pages": num_pages,
                "text_length": len(all_text),
                "subject_hints": list(set(subject_hints)),
                "status": "success",
            }
        except Exception as exc:
            logger.exception("Failed to load PDF: %s", exc)
            return {
                "pages": 0,
                "text_length": 0,
                "subject_hints": [],
                "status": f"error: {exc}",
            }

    def query_pdf(self, question: str) -> str:
        """Answer a question using extracted PDF content.

        Looks for keywords from the question in the stored PDF text
        and returns a response based on the knowledge base or memory.
        """
        # Try to find relevant content from memory first
        remembered = self.memory.answer_question(question)
        if remembered:
            return remembered

        # Try KB search
        try:
            match = self.kb.find_answer(question)
            if match is not None:
                return match.answer
        except Exception:
            logger.exception("KB search failed")

        # Try classifier + intent-based search
        try:
            prediction = self.classifier.predict(question)
            intent = prediction.get("intent", "fallback")
            confidence = float(prediction.get("confidence", 0.0))
            logger.debug("PDF query intent=%s confidence=%.2f", intent, confidence)

            # Search KB with intent hint
            match = self.kb.find_answer(question, intent_hint=intent)
            if match is not None:
                return match.answer
        except Exception:
            logger.exception("PDF query classification failed")

        # Fallback: general response
        return "I couldn't find an answer to your question in the PDF document."
