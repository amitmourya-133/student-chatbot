"""Tests for the web chat API and offline knowledge-base regressions.

Covers the FastAPI chat endpoint contract, offline answer quality for the
known questions that previously fell back, the graceful fallback for unknown
questions, malformed/empty input handling, and backend error handling.
"""

from __future__ import annotations

from unittest import mock

import pytest
from fastapi.testclient import TestClient

from services.knowledge_base import KnowledgeBase
from webapp.app import app

client = TestClient(app)


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    store = KnowledgeBase()
    assert store.load()
    return store


# --------------------------------------------------------------------------- #
# chat API contract
# --------------------------------------------------------------------------- #
def test_chat_endpoint_returns_response_and_mode():
    res = client.post("/api/chat", json={"message": "what is database"})
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] == "offline"
    assert "response" in data
    assert "database" in data["response"].lower()


def test_chat_endpoint_uses_actual_backend_reply():
    """The UI renders the real pipeline reply, not a canned greeting."""
    res = client.post("/api/chat", json={"message": "what is database"})
    data = res.json()
    assert data["response"].lower().startswith("a database is")


def test_empty_message_is_rejected():
    res = client.post("/api/chat", json={"message": "   "})
    assert res.status_code == 400


def test_malformed_body_is_rejected():
    res = client.post("/api/chat", json={"not_message": "hello"})
    assert res.status_code == 422


def test_backend_error_returns_500_and_logs():
    with mock.patch("webapp.app.pipeline") as fake:
        fake.reply.side_effect = RuntimeError("boom")
        fake.mode = "offline"
        res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 500


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_status_endpoint_reports_provider_status():
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["mode"] in ("online", "offline")
    assert "online" in data
    assert "provider_status" in data
    assert "api_key_configured" in data


def test_chat_response_includes_source_and_provider():
    res = client.post("/api/chat", json={"message": "hello"})
    assert res.status_code == 200
    data = res.json()
    assert data["source"] in (
        "OFFLINE_KB",
        "INTENT_MODEL",
        "GEMINI",
        "FALLBACK",
        "MEMORY",
        "DOCUMENT",
        "HYBRID",
    )
    assert isinstance(data["online"], bool)
    assert data["provider_status"]
    assert "fallback_used" in data
    assert "request_id" in data


def test_index_page_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Student AI Assistant" in res.text


def test_unknown_question_falls_back_gracefully():
    res = client.post(
        "/api/chat",
        json={"message": "purple elephant submarine quantum banana peel zzz"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["response"].strip()
    assert data["mode"] == "offline"


# --------------------------------------------------------------------------- #
# offline knowledge base regression (previously-failing known questions)
# --------------------------------------------------------------------------- #
def test_what_is_database(kb):
    match = kb.find_answer("what is database")
    assert match is not None
    assert match.score >= kb.threshold
    assert "database" in match.question.lower()


def test_what_is_newton_ring(kb):
    match = kb.find_answer("what is newton ring")
    assert match is not None
    assert "newton" in match.question.lower()
    # regression: must be the optics phenomenon, not Newton's laws of motion
    assert "ring" in match.question.lower()
    assert "interference" in match.answer.lower()


def test_what_is_newton_first_law_still_works(kb):
    match = kb.find_answer("what is newton's first law")
    assert match is not None
    assert "first law" in match.question.lower()
    assert "inertia" in match.answer.lower()


def test_what_is_sex(kb):
    match = kb.find_answer("what is sex")
    assert match is not None
    assert match.score >= kb.threshold
    assert "sex" in match.question.lower()


def test_what_is_hindi_meaning_of_phone(kb):
    for query in ("what is hindi meaning of phone", "phone ka hindi meaning kya hai"):
        match = kb.find_answer(query)
        assert match is not None, f"no match for {query!r}"
        assert "phone" in match.question.lower()
        assert "फ़ोन" in match.answer or "फोन" in match.answer


def test_unknown_question_has_no_match(kb):
    match = kb.find_answer("purple elephant submarine quantum banana peel zzz")
    assert match is None
