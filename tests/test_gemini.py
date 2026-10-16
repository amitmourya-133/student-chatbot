"""Tests for the Gemini (online) client error mapping and retry policy.

All provider interactions are mocked; no live API quota is used here.
"""

from __future__ import annotations

import pytest

from services import gemini as gem


@pytest.fixture()
def configured(monkeypatch):
    monkeypatch.setattr(gem.config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gem.config, "GEMINI_RETRIES", 1)
    monkeypatch.setattr(gem.config, "GEMINI_BACKOFF_BASE", 0.5)
    monkeypatch.setattr(gem.config, "GEMINI_MAX_BACKOFF", 4.0)
    monkeypatch.setattr(gem.time, "sleep", lambda *_: None)


@pytest.fixture()
def sleeps(monkeypatch):
    """Record (rather than perform) backoff sleeps."""
    recorded: list[float] = []
    monkeypatch.setattr(gem.time, "sleep", recorded.append)
    return recorded


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status_code", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_transient(status_code):
    assert gem.is_transient_error(gem.GeminiError("x", status=status_code)) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404])
def test_permanent_statuses_are_not_transient(status_code):
    assert gem.is_transient_error(gem.GeminiError("x", status=status_code)) is False


@pytest.mark.parametrize(
    "message",
    [gem.MSG_PROVIDER_PROBLEM, gem.MSG_TIMEOUT, gem.MSG_UNREACHABLE, gem.MSG_RATE_LIMITED],
)
def test_transient_messages_are_transient(message):
    assert gem.is_transient_error(gem.GeminiError(message)) is True


@pytest.mark.parametrize(
    "message",
    [gem.MSG_AUTH, gem.MSG_MODEL_MISSING, gem.MSG_REQUEST_FAILED],
)
def test_permanent_messages_are_not_transient(message):
    assert gem.is_transient_error(gem.GeminiError(message)) is False


def test_model_missing_status_is_mapped():
    assert gem._message_for_status(404) == gem.MSG_MODEL_MISSING
    assert gem._message_for_status(429) == gem.MSG_RATE_LIMITED
    assert gem._message_for_status(401) == gem.MSG_AUTH
    assert gem._message_for_status(503) == gem.MSG_PROVIDER_PROBLEM
    assert gem._message_for_status(200) is None


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("5", 5.0), ("5s", 5.0), ("1.5s", 1.5), ("-1", None), ("soon", None)],
)
def test_parse_retry_after(value, expected):
    assert gem.parse_retry_after(value) == expected


# --------------------------------------------------------------------------- #
# generation + retry behavior
# --------------------------------------------------------------------------- #
def test_successful_generation_returns_text(configured, monkeypatch):
    monkeypatch.setattr(gem, "_attempt_online", lambda m, h, s=None: "hello from gemini")
    assert gem.generate_response("hi") == "hello from gemini"


def test_transient_error_is_retried_once_then_succeeds(configured, monkeypatch):
    calls = {"n": 0}

    def flaky(message, history, system_instruction):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gem.GeminiError(gem.MSG_PROVIDER_PROBLEM, status=503)
        return "recovered"

    monkeypatch.setattr(gem, "_attempt_online", flaky)
    assert gem.generate_response("hi") == "recovered"
    assert calls["n"] == 2


def test_persistent_503_raises_after_retries_exhausted(configured, monkeypatch):
    calls = {"n": 0}

    def always_503(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_PROVIDER_PROBLEM, status=503)

    monkeypatch.setattr(gem, "_attempt_online", always_503)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_PROVIDER_PROBLEM
    assert calls["n"] == 2  # one original + bounded retry


def test_429_is_retried_and_honors_retry_after(configured, sleeps, monkeypatch):
    calls = {"n": 0}

    def flaky(message, history, system_instruction):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gem.GeminiError(gem.MSG_RATE_LIMITED, status=429, retry_after=2.0)
        return "ok"

    monkeypatch.setattr(gem, "_attempt_online", flaky)
    assert gem.generate_response("hi") == "ok"
    assert calls["n"] == 2
    assert len(sleeps) == 1
    # Retry-After is honored (>= 2.0) but capped by GEMINI_MAX_BACKOFF.
    assert 2.0 <= sleeps[0] <= gem.config.GEMINI_MAX_BACKOFF


def test_429_exhausted_raises_rate_limited(configured, monkeypatch):
    def always_429(message, history, system_instruction):
        raise gem.GeminiError(gem.MSG_RATE_LIMITED, status=429, retry_after=1.0)

    monkeypatch.setattr(gem, "_attempt_online", always_429)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_RATE_LIMITED


def test_auth_error_is_not_retried(configured, monkeypatch):
    calls = {"n": 0}

    def bad_key(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_AUTH, status=401)

    monkeypatch.setattr(gem, "_attempt_online", bad_key)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_AUTH
    assert calls["n"] == 1


def test_forbidden_error_is_not_retried(configured, monkeypatch):
    calls = {"n": 0}

    def forbidden(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_AUTH, status=403)

    monkeypatch.setattr(gem, "_attempt_online", forbidden)
    with pytest.raises(gem.GeminiError):
        gem.generate_response("hi")
    assert calls["n"] == 1


def test_invalid_model_error_is_not_retried(configured, monkeypatch):
    calls = {"n": 0}

    def missing_model(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_MODEL_MISSING, status=404)

    monkeypatch.setattr(gem, "_attempt_online", missing_model)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_MODEL_MISSING
    assert calls["n"] == 1


def test_timeout_is_retried_then_raises(configured, monkeypatch):
    calls = {"n": 0}

    def always_timeout(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_TIMEOUT)

    monkeypatch.setattr(gem, "_attempt_online", always_timeout)
    with pytest.raises(gem.GeminiError) as exc:
        gem.generate_response("hi")
    assert str(exc.value) == gem.MSG_TIMEOUT
    assert calls["n"] == 2  # one original + one retry


def test_retry_count_is_bounded(monkeypatch, sleeps):
    monkeypatch.setattr(gem.config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(gem.config, "GEMINI_RETRIES", 0)
    calls = {"n": 0}

    def always(message, history, system_instruction):
        calls["n"] += 1
        raise gem.GeminiError(gem.MSG_PROVIDER_PROBLEM, status=503)

    monkeypatch.setattr(gem, "_attempt_online", always)
    with pytest.raises(gem.GeminiError):
        gem.generate_response("hi")
    assert calls["n"] == 1
    assert sleeps == []


def test_retry_delay_is_exponential_bounded_and_jittered(configured, monkeypatch):
    monkeypatch.setattr(gem.config, "GEMINI_BACKOFF_BASE", 0.5)
    monkeypatch.setattr(gem.config, "GEMINI_MAX_BACKOFF", 4.0)
    for attempt in range(8):
        delay = gem._retry_delay(attempt, gem.GeminiError("x", status=503))
        assert 0 <= delay <= 4.0
    # A large provider Retry-After is capped so we never block past the budget.
    capped = gem._retry_delay(0, gem.GeminiError("x", status=429, retry_after=1000.0))
    assert capped <= 4.0


def test_unconfigured_raises_immediately(monkeypatch):
    monkeypatch.setattr(gem.config, "GEMINI_API_KEY", "")
    with pytest.raises(gem.GeminiError):
        gem.generate_response("hi")


def test_error_status_and_retry_after_exposed():
    err = gem.GeminiError("x", status=503, retry_after=1.5)
    assert err.status == 503
    assert err.retry_after == 1.5
