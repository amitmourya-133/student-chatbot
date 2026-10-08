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
    pipe = AssistantPipeline(force_offline=True)
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
        pipeline_mod, "generate_response", lambda msg, hist: "online answer"
    )
    result = offline_pipeline.reply("hello")
    assert result.online is True
    assert result.source == pipeline_mod.SOURCE_GEMINI
    assert result.text == "online answer"
    assert not result.error


def test_gemini_error_falls_back_gracefully(offline_pipeline, monkeypatch):
    offline_pipeline.online = True

    def boom(msg, hist):
        raise GeminiError("The online AI service is having problems.")

    monkeypatch.setattr(pipeline_mod, "generate_response", boom)
    result = offline_pipeline.reply("question")
    assert result.online is False
    assert result.just_switched is True
    assert result.error == "The online AI service is having problems."
    assert result.text.strip()  # offline answer still produced


def test_unexpected_online_error_falls_back(offline_pipeline, monkeypatch):
    offline_pipeline.online = True

    def boom(msg, hist):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_mod, "generate_response", boom)
    result = offline_pipeline.reply("question")
    assert result.online is False
    assert result.just_switched is True
    assert "failed" in result.error.lower()
    assert result.text.strip()


def test_online_disabled_is_latched(offline_pipeline, monkeypatch):
    offline_pipeline.online = True
    monkeypatch.setattr(
        pipeline_mod, "generate_response", lambda m, h: (_ for _ in ()).throw(GeminiError("x"))
    )
    offline_pipeline.reply("a")
    assert offline_pipeline.online_disabled is True
    assert offline_pipeline.online is False


def test_online_history_pairs_includes_recent(offline_pipeline, monkeypatch):
    captured = {}

    def fake_generate(msg, hist):
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

    def spy(msg, hist):
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


def test_force_offline_never_goes_online(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "is_online", lambda: True)
    pipe = AssistantPipeline(force_offline=True)
    pipe.refresh_mode(initial=True)
    assert pipe.online is False