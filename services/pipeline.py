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
from pathlib import Path

import config
from fnmatch import fnmatch
from pypdf import PdfReader
from nlp.classifier import IntentClassifier
from services.chat_history import ChatHistory
from services.gemini import GeminiError, generate_response
from services.internet import is_online
from services.knowledge_base import NO_MATCH_MESSAGE, KnowledgeBase
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