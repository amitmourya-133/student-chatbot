"""Gemini (online) client using the current ``google-genai`` SDK.

Any failure raises :class:`GeminiError` with a safe, user-friendly message so
the caller can fall back to the offline pipeline. Stack traces, API keys and
raw HTTP bodies are only written to the log, never returned to the user.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Iterable, Optional

import requests

import config

logger = logging.getLogger(__name__)

REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)

try:  # recommended SDK - offline mode must keep working without it
    from google import genai as genai_sdk
    from google.genai import types as genai_types
    from google.genai import errors as genai_errors

    _SDK_AVAILABLE = True
except Exception:  # pragma: no cover - SDK not installed
    genai_sdk = None
    genai_types = None
    genai_errors = None
    _SDK_AVAILABLE = False


class GeminiError(Exception):
    """Raised when the online model cannot produce an answer.

    ``status`` carries the provider HTTP status when known and ``retry_after``
    the provider-requested delay in seconds (from ``Retry-After`` or the
    ``RetryInfo.retryDelay`` detail). Neither ever contains credentials.
    """

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        retry_after: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


# Safe, user-facing messages for failure categories. Kept in one place so the
# pipeline can tell a *transient* provider problem (worth a bounded retry)
# apart from a permanent one (bad key, unknown model, bad request).
MSG_RATE_LIMITED = "The online AI is rate limited right now."
MSG_AUTH = "The online AI could not authenticate."
MSG_MODEL_MISSING = "The configured online model is not available."
MSG_PROVIDER_PROBLEM = "The online AI service is having problems."
MSG_TIMEOUT = "The online AI timed out."
MSG_UNREACHABLE = "Could not reach the online AI service."
MSG_REQUEST_FAILED = "The online AI request failed."
MSG_EMPTY = "The model returned an empty response."
MSG_BAD_PAYLOAD = "The online AI returned an unexpected response."

# Transient HTTP statuses worth a bounded retry: rate limits, request timeout,
# server errors and gateway errors. 400/401/403/404 are permanent.
_TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Transient categories: retrying is reasonable and cheap. Message-based
# fallback for errors raised without a status (e.g. timeouts, connection loss).
_TRANSIENT_MESSAGES = frozenset(
    {MSG_PROVIDER_PROBLEM, MSG_TIMEOUT, MSG_UNREACHABLE, MSG_RATE_LIMITED}
)


def is_transient_error(exc: "GeminiError") -> bool:
    """True when ``exc`` is worth a single bounded retry.

    Prefers the provider HTTP status when present; otherwise falls back to the
    safe message category. Authentication, access and unknown-model errors are
    never transient.
    """
    status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in _TRANSIENT_STATUSES
    return str(exc) in _TRANSIENT_MESSAGES


def parse_retry_after(value: object) -> Optional[float]:
    """Parse a ``Retry-After`` header / ``retryDelay`` value into seconds.

    Accepts seconds (``"5"``) and durations (``"5s"``, ``"1.5s"``); returns
    ``None`` for missing/HTTP-date values (which we do not need here)."""
    if value is None:
        return None
    text = str(value).strip().lower().rstrip("s")
    try:
        seconds = float(text)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def is_configured() -> bool:
    """True when an API key is present (does not validate the key)."""
    return bool(config.GEMINI_API_KEY)


def is_sdk_available() -> bool:
    return _SDK_AVAILABLE


def _normalise_history(
    history: Optional[Iterable[dict]],
) -> list[dict]:
    """Convert ``[{role, content}]`` entries into Gemini history dicts."""
    cleaned: list[dict] = []
    if not history:
        return cleaned
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role") or item.get("user")
        text = item.get("content") or item.get("text") or item.get("assistant")
        if not text:
            continue
        if role in ("user", "human"):
            cleaned.append({"role": "user", "parts": [str(text)]})
        else:
            cleaned.append({"role": "model", "parts": [str(text)]})
    return cleaned[-config.GEMINI_MAX_HISTORY:]


def _message_for_status(status) -> Optional[str]:
    """Map an HTTP status code to a safe user-facing message."""
    if status == 404:
        return MSG_MODEL_MISSING
    if status == 429:
        return MSG_RATE_LIMITED
    if status in (400, 401, 403):
        return MSG_AUTH
    if isinstance(status, int) and status >= 500:
        return MSG_PROVIDER_PROBLEM
    return None


def _sdk_retry_after(exc: Exception) -> Optional[float]:
    """Best-effort extraction of a provider retry delay from an SDK error."""
    for attr in ("retry_after", "retry_delay", "retryDelay"):
        parsed = parse_retry_after(getattr(exc, attr, None))
        if parsed is not None:
            return parsed
    match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([0-9.]+)s", str(exc))
    if match:
        return parse_retry_after(match.group(1))
    return None


def _raise_mapped_sdk_error(exc: Exception) -> None:
    """Translate SDK exceptions into safe :class:`GeminiError` messages."""
    raw_status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    status = raw_status if isinstance(raw_status, int) else None
    message = _message_for_status(status)
    if message:
        raise GeminiError(message, status=status, retry_after=_sdk_retry_after(exc))
    if "api key" in str(exc).lower():
        raise GeminiError(MSG_AUTH, status=status)
    logger.warning("Gemini SDK error: %s: %s", type(exc).__name__, exc)
    raise GeminiError(MSG_REQUEST_FAILED, status=status)


def _generate_via_sdk(
    message: str,
    history: list[dict],
    system_instruction: Optional[str] = None,
) -> str:
    # the google-genai SDK takes the request timeout in milliseconds via
    # HttpOptions - generate_content() itself has no timeout parameter
    client = genai_sdk.Client(
        api_key=config.GEMINI_API_KEY,
        http_options=genai_types.HttpOptions(
            timeout=int(config.GEMINI_TIMEOUT * 1000)
        ),
    )

    contents: list[genai_types.Content] = [
        genai_types.Content(
            role=entry["role"],
            parts=[genai_types.Part.from_text(text=text) for text in entry["parts"]],
        )
        for entry in history
    ]
    contents.append(
        genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=message)])
    )

    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction or config.SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048,
            ),
        )
    except Exception as exc:
        if genai_errors is not None and isinstance(exc, genai_errors.APIError):
            _raise_mapped_sdk_error(exc)
        error_name = type(exc).__name__.lower()
        if isinstance(exc, TimeoutError) or "timeout" in error_name:
            raise GeminiError(MSG_TIMEOUT)
        if isinstance(exc, OSError) or "connect" in error_name or "network" in error_name:
            raise GeminiError(MSG_UNREACHABLE)
        raise  # unknown failure - generate_response retries through REST

    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise GeminiError(MSG_EMPTY)
    return str(text).strip()


def _rest_error(response: "requests.Response") -> GeminiError:
    """Build a safe :class:`GeminiError` from a non-OK REST response."""
    status = response.status_code
    retry_after = parse_retry_after(response.headers.get("Retry-After"))
    if retry_after is None:
        retry_after = _rest_retry_delay(response)
    message = _message_for_status(status)
    if message is None:
        return GeminiError(MSG_REQUEST_FAILED, status=status, retry_after=retry_after)
    return GeminiError(message, status=status, retry_after=retry_after)


def _rest_retry_delay(response: "requests.Response") -> Optional[float]:
    """Read ``RetryInfo.retryDelay`` from a JSON error body when present."""
    try:
        details = response.json().get("error", {}).get("details", []) or []
    except Exception:
        return None
    for item in details:
        if isinstance(item, dict) and "RetryInfo" in str(item.get("@type", "")):
            return parse_retry_after(item.get("retryDelay"))
    return None


def _generate_via_rest(
    message: str,
    history: list[dict],
    system_instruction: Optional[str] = None,
) -> str:
    # the REST API requires parts as [{"text": ...}] objects, not raw strings
    contents = [
        {
            "role": entry["role"],
            "parts": [{"text": text} for text in entry["parts"]],
        }
        for entry in history
    ]
    contents.append({"role": "user", "parts": [{"text": message}]})
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction or config.SYSTEM_PROMPT}]
        },
        "contents": contents,
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.7},
    }
    response = requests.post(
        REST_URL.format(model=config.GEMINI_MODEL),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": config.GEMINI_API_KEY or "",
        },
        json=payload,
        timeout=config.GEMINI_TIMEOUT,
    )
    if not response.ok:
        raise _rest_error(response)

    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected Gemini payload shape: %s", exc)
        raise GeminiError(MSG_BAD_PAYLOAD)

    if not text.strip():
        raise GeminiError(MSG_EMPTY)
    return text.strip()


def _attempt_online(
    message: str,
    history: list[dict],
    system_instruction: Optional[str],
) -> str:
    """One SDK-or-REST attempt. Raises :class:`GeminiError` on failure."""
    if _SDK_AVAILABLE:
        try:
            return _generate_via_sdk(message, history, system_instruction)
        except GeminiError:
            raise
        except Exception:
            logger.exception("Unexpected Gemini SDK failure")

    try:
        return _generate_via_rest(message, history, system_instruction)
    except GeminiError:
        raise
    except requests.exceptions.Timeout as exc:
        logger.warning("Gemini timeout: %s", exc)
        raise GeminiError(MSG_TIMEOUT)
    except requests.exceptions.ConnectionError as exc:
        logger.warning("Gemini connection error: %s", exc)
        raise GeminiError(MSG_UNREACHABLE)
    except Exception:
        logger.exception("Unexpected Gemini failure")
        raise GeminiError(MSG_REQUEST_FAILED)


def _retry_delay(attempt: int, error: GeminiError) -> float:
    """Exponential backoff with jitter, honoring a capped provider delay.

    The wait is bounded by ``config.GEMINI_MAX_BACKOFF`` so a single message
    stays within the request timeout and Vercel's function time budget.
    """
    base = max(0.0, config.GEMINI_BACKOFF_BASE)
    cap = max(base, config.GEMINI_MAX_BACKOFF)
    delay = min(base * (2 ** attempt), cap)
    if error.retry_after is not None:
        delay = max(delay, min(error.retry_after, cap))
    jitter = random.uniform(0, delay * 0.25)
    return min(delay + jitter, cap)


def generate_response(
    user_message: str,
    conversation_history: Optional[Iterable[dict]] = None,
    system_instruction: Optional[str] = None,
) -> str:
    """Return an answer from Gemini for ``user_message``.

    ``system_instruction`` replaces the default ``config.SYSTEM_PROMPT`` when
    provided; ``None`` keeps the current behaviour exactly.

    Transient failures (429 rate limit, 408, 5xx, timeout, connection loss) are
    retried at most ``config.GEMINI_RETRIES`` extra times with exponential
    backoff plus jitter, honoring a bounded ``Retry-After``. Permanent failures
    (bad key, unknown model, bad request) fail immediately.

    Raises :class:`GeminiError` on any failure so callers can fall back to
    the offline knowledge base.
    """
    if not is_configured():
        raise GeminiError("No API key configured.")
    if not user_message or not user_message.strip():
        raise GeminiError("Empty message.")

    history = _normalise_history(conversation_history)
    attempts = max(1, config.GEMINI_RETRIES + 1)

    last_error: Optional[GeminiError] = None
    for attempt in range(attempts):
        try:
            return _attempt_online(
                user_message.strip(), history, system_instruction
            )
        except GeminiError as exc:
            last_error = exc
            if not is_transient_error(exc) or attempt == attempts - 1:
                raise
            delay = _retry_delay(attempt, exc)
            logger.info(
                "Transient Gemini failure (attempt %d/%d, status=%s), "
                "retrying in %.2fs: %s",
                attempt + 1,
                attempts,
                exc.status,
                delay,
                exc,
            )
            time.sleep(delay)

    # Unreachable: the loop either returns or raises.
    raise last_error or GeminiError(MSG_REQUEST_FAILED)
