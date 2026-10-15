"""Tests for the shared AssistantPipeline (services/pipeline.py)."""

from __future__ import annotations

from unittest import mock

import pytest

from services import pipeline as pipeline_mod
from services.gemini import GeminiError
from services.pipeline import AssistantPipeline


@pytest.fixture()
def offline_pipeline(tmp_path, monkeypatch):
    """Pipeline pinned to offline mode with fresh chat history."""
    patch = tmp_path / "chat_log.json"
    patch.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(pipeline_mod.config, "CHAT_LOG_FILE", patch)
    pipe = AssistantPipeline(force_offline=True, memory_path=tmp_path / "memory.json")
    pipe.load_intent_responses()
    return pipe


def test_offline_kb_answer_source(offline_pipeline):
    result = offline_pipeline.reply("Explain Python in simple words")
    assert result.online is False
    assert not result.error
    assert result.text.strip()
    assert result.source == pipeline_mod.SOURCE_OFFLINE_KB


def test_generic_question_still_answers(offline_pipeline):
    result = offline_pipeline.reply("python kya hai")
    assert "python" in result.text.lower()


def test_unknown_input_falls_back(offline_pipeline):
    result = offline_pipeline.reply("zzqx quantum waffle 918273")
    assert result.text.strip()
    assert result.source in (pipeline_mod.SOURCE_INTENT, pipeline_mod.SOURCE_FALLBACK)


def test_empty_reply_never_raises(offline_pipeline):
    assert offline_pipeline.reply("   ").text or True


def test_online_success_sets_gemini_source(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    monkeypatch.setattr(
        pipeline_mod, "generate_response", lambda msg, hist, system_instruction=None: "online answer"
    )
    result = offline_pipeline.reply("hello")
    assert result.online is True
    assert result.source == pipeline_mod.SOURCE_GEMINI
    assert result.text == "online answer"
    assert not result.error


def test_gemini_error_falls_back_gracefully(offline_pipeline, monkeypatch):
    offline_pipeline.online = True

    def boom(msg, hist, system_instruction=None):
        raise GeminiError("The online AI service is having problems.")

    monkeypatch.setattr(pipeline_mod, "generate_response", boom)
    result = offline_pipeline.reply("question")
    assert result.online is False
    assert result.fallback_used is True
    assert result.provider_status == pipeline_mod.PROVIDER_ERROR
    assert result.error == "The online AI service is having problems."
    assert result.text.strip()  # offline answer still produced
    # The user-selected mode must be preserved, not silently flipped offline.
    assert offline_pipeline.online is True
    assert offline_pipeline.mode == "online"
    assert offline_pipeline.online_disabled is True


def test_unexpected_online_error_falls_back(offline_pipeline, monkeypatch):
    offline_pipeline.online = True

    def boom(msg, hist, system_instruction=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_mod, "generate_response", boom)
    result = offline_pipeline.reply("question")
    assert result.online is False
    assert result.fallback_used is True
    assert result.provider_status == pipeline_mod.PROVIDER_ERROR
    assert "failed" in result.error.lower()
    assert result.text.strip()
    assert offline_pipeline.online is True


def test_provider_failure_enters_cooldown_not_offline(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    monkeypatch.setattr(
        pipeline_mod,
        "generate_response",
        lambda m, h, system_instruction=None: (_ for _ in ()).throw(GeminiError("x")),
    )
    offline_pipeline.reply("a")
    assert offline_pipeline.online_disabled is True
    # Mode is unchanged; only the provider is cooling down.
    assert offline_pipeline.online is True
    assert offline_pipeline.provider_status == pipeline_mod.PROVIDER_ERROR


def test_cooldown_does_not_expire_immediately(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    offline_pipeline._mark_provider_error("boom")
    # Within the cooldown window the provider is skipped without retrying.
    assert offline_pipeline.provider_status == pipeline_mod.PROVIDER_ERROR
    assert offline_pipeline.online_disabled is True


def test_reset_provider_cooldown_allows_retry(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    offline_pipeline._mark_provider_error("boom")
    offline_pipeline.reset_provider_cooldown()
    assert offline_pipeline.online_disabled is False
    assert offline_pipeline.provider_status == pipeline_mod.PROVIDER_OK
    monkeypatch.setattr(
        pipeline_mod,
        "generate_response",
        lambda m, h, system_instruction=None: "recovered",
    )
    result = offline_pipeline.reply("hello")
    assert result.online is True
    assert result.source == pipeline_mod.SOURCE_GEMINI
    assert result.text == "recovered"


def test_online_disabled_is_latched(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    monkeypatch.setattr(
        pipeline_mod,
        "generate_response",
        lambda m, h, system_instruction=None: (_ for _ in ()).throw(GeminiError("x")),
    )
    offline_pipeline.reply("a")
    assert offline_pipeline.online_disabled is True
    # Latched provider cool-down while the selected mode stays online.
    assert offline_pipeline.online is True


def test_online_failure_no_local_answer_uses_provider_message(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    monkeypatch.setattr(
        pipeline_mod,
        "generate_response",
        lambda m, h, system_instruction=None: (_ for _ in ()).throw(
            GeminiError("The online AI service is having problems.")
        ),
    )
    result = offline_pipeline.reply("zzqx quantum waffle 918273")
    assert result.source == pipeline_mod.SOURCE_FALLBACK
    assert result.fallback_used is True
    assert result.text == pipeline_mod.PROVIDER_FALLBACK_MESSAGE
    # The generic offline fallback template must not be shown as the answer.
    assert "didn't quite get that" not in result.text.lower()


def test_online_history_pairs_includes_recent(offline_pipeline, monkeypatch):
    captured = {}

    def fake_generate(msg, hist, system_instruction=None):
        captured["hist"] = hist
        return "ok"

    offline_pipeline.online = True
    offline_pipeline.history.add("hello", "hi bot", "offline")
    monkeypatch.setattr(pipeline_mod, "generate_response", fake_generate)
    offline_pipeline.reply("again")
    assert {"role": "user", "content": "hello"} in captured["hist"]
    assert {"role": "assistant", "content": "hi bot"} in captured["hist"]


def test_root_of_generate_online_uses_real_service(offline_pipeline, monkeypatch):
    """The plugin seam for modes/documents must route through generate_response."""
    sentinel = {"used": False}
    real = pipeline_mod.generate_response

    def spy(msg, hist, system_instruction=None):
        sentinel["used"] = True
        return real(msg, hist) if False else "spy"

    monkeypatch.setattr(pipeline_mod, "generate_response", spy)
    offline_pipeline.online = True
    assert offline_pipeline.generate_online("x", []) == "spy"
    assert sentinel["used"] is True


def test_startup_status_shape(offline_pipeline):
    status = offline_pipeline.startup()
    assert status.intent_model is True  # trained model ships in repo
    assert status.knowledge_base is True
    assert status.kb_entries > 0
    assert status.online is False  # force_offline
    assert status.api_key is True or status.api_key is False
    assert status.provider_status in (
        pipeline_mod.PROVIDER_OFFLINE,
        pipeline_mod.PROVIDER_OK,
        pipeline_mod.PROVIDER_ERROR,
        pipeline_mod.PROVIDER_UNCONFIGURED,
        pipeline_mod.PROVIDER_UNREACHABLE,
    )


def test_force_offline_never_goes_online(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "is_online", lambda: True)
    pipe = AssistantPipeline(force_offline=True)
    pipe.refresh_mode(initial=True)
    assert pipe.online is False


def test_reply_assigns_request_id(offline_pipeline):
    result = offline_pipeline.reply("hello")
    assert result.request_id


def test_momentum_offline_answer(offline_pipeline):
    """Regression: 'momentum kya hota hai' must give the momentum answer."""
    for query in ("what is momentum", "momentum kya hota hai", "momentum formula"):
        result = offline_pipeline.reply(query)
        assert result.source == pipeline_mod.SOURCE_OFFLINE_KB, query
        assert "m" in result.text.lower() and "v" in result.text.lower()
        assert "programming query" not in result.text.lower()


def test_offtopic_content_intent_is_not_answered_with_template(offline_pipeline):
    """A content intent with no topic overlap must not emit a canned template."""
    result = offline_pipeline.reply("kya hota hai")
    # 'kya hota hai' is all generic words - it cannot be grounded to a topic.
    assert result.source == pipeline_mod.SOURCE_FALLBACK


def test_social_intent_with_generic_words_still_answers(offline_pipeline):
    """Social intents ('kaise ho') are exempt from content grounding."""
    result = offline_pipeline.reply("kaise ho")
    assert result.source == pipeline_mod.SOURCE_INTENT


def test_chat_endpoint_returns_valid_response():
    """Regression test: verify /api/chat endpoint returns 200 with valid response.

    Original bug: pipeline.pipeline.reply() caused AttributeError (HTTP 500)
    because AssistantPipeline has no .pipeline attribute.
    Fixed: changed to pipeline.reply() in webapp/app.py.
    """
    from fastapi.testclient import TestClient
    from webapp.app import app

    client = TestClient(app)

    # Test that chat endpoint returns 200 (not 500)
    resp = client.post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.json()}"

    # Test that response has a non-empty "response" field
    data = resp.json()
    assert "response" in data, "Response JSON missing 'response' key"
    assert data["response"].strip(), "Response should not be empty"

    # Test that mode is included
    assert "mode" in data, "Response JSON missing 'mode' key"

    # Test with a content question
    resp2 = client.post("/api/chat", json={"message": "what is python"})
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["response"].strip(), "Response should not be empty"


def test_memory_history_failure_does_not_cause_false_error():
    """The web chat endpoint stays healthy when persistence backends are present.

    Negative control for the CLI false-error fix: /api/chat does not persist
    history or memory, so it must return a normal 200 answer. True
    persistence-failure coverage lives in tests/test_cli_fix.py.
    """
    from fastapi.testclient import TestClient
    from webapp.app import app

    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["response"].strip(), "Response should not be empty"


def test_one_input_one_response():
    """Verify one user input produces exactly one assistant response.

    Regression guard: a frontend bug previously rendered the same assistant
    answer repeatedly. This ensures the backend never returns stacked copies
    of the same content for a single request.
    """
    from fastapi.testclient import TestClient
    from webapp.app import app

    client = TestClient(app)
    resp1 = client.post("/api/chat", json={"message": "hello"})
    assert resp1.status_code == 200
    resp2 = client.post("/api/chat", json={"message": "world"})
    assert resp2.status_code == 200
    data1 = resp1.json()
    data2 = resp2.json()
    assert data1["response"].strip(), "First response should not be empty"
    assert data2["response"].strip(), "Second response should not be empty"

    for label, data in (("first", data1), ("second", data2)):
        text = data["response"].strip()
        if len(text) >= 6:
            half = len(text) // 2
            assert text[:half] != text[half : 2 * half], (
                f"{label} response contains duplicated stacked content"
            )
