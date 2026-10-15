"""Tests for the Gemini (online) client error mapping and retry policy."""

from __future__ import annotations

import pytest

from services import gemini as gem


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setattr(gem.config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gem.config, "GEMINI_RETRIES", 1)
    monkeypatch.setattr(gem.time, "sleep", lambda *_: None)


@pytest.mark.parametrize(
    "message",
    [gem.MSG_PROVIDER_PROBLEM, gem.MSG_TIMEOUT, gem.MSG_UNREACHABLE],
)
def test_transient_errors_are_transient(message):
    assert gem.is_transient_error(gem.GeminiError(message)) is True


@pytest.mark.parametrize(
    "message",
    [gem.MSG_AUTH, gem.MSG_MODEL_MISSING, gem.MSG_RATE_LIMITED, gem.MSG_REQUEST_FAILED],
)
def test_permanent_errors_are_not_transient(message):
    assert gem.is_transient_error(gem.GeminiError(message)) is False


def test_model_missing_status_is_mapped():
    assert gem._message_for_status(404) == gem.MSG_MODEL_MISSING
    assert gem._message_for_status(429) == gem.MSG_RATE_LIMITED
    assert gem._message_for_status(401) == gem.MSG_AUTH
    assert gem._message_for_status(503) == gem.MSG_PROVIDER_PROBLEM
    assert gem._message_for_status(200) is None


def test_transient_error_is_retried_once_then_succeeds(configured, monkeypatch):
    calls = {"n": 0}

    def flaky(message, history, system_instruction):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gem.GeminiError(gem.MSG_PROVIDER_PROBLEM)
        return "recovered"

    monkeypatch.setattr(gem, "_attempt_online", flaky)
    assert gem.generate_response("hi") == "recovered"
    assert calls["n"] == 2


def test_permanent_error_is_not_retried(configured, monkeypatch):
    calls = {"n": 0}

    def bad_key(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_AUTH)

    monkeypatch.setattr(gem, "_attempt_online", bad_key)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_AUTH
    assert calls["n"] == 1


def test_transient_error_raises_after_retries_exhausted(configured, monkeypatch):
    calls = {"n": 0}

    def always_timeout(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_TIMEOUT)

    monkeypatch.setattr(gem, "_attempt_online", always_timeout)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_TIMEOUT
    assert calls["n"] == 2  # one original + one retry


def test_unconfigured_raises_immediately(monkeypatch):
    monkeypatch.setattr(gem.config, "GEMINI_API_KEY", "")
    with pytest.raises(gem.GeminiError):
        gem.generate_response("hi")
