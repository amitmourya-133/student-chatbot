"""Gemini (online) client using the current ``google-genai`` SDK.

Any failure raises :class:`GeminiError` with a safe, user-friendly message so
the caller can fall back to the offline pipeline. Stack traces, API keys and
raw HTTP bodies are only written to the log, never returned to the user.
"""

from __future__ import annotations

import logging
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
    """Raised when the online model cannot produce an answer."""


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


def _raise_mapped_sdk_error(exc: Exception) -> None:
    """Translate SDK exceptions into safe :class:`GeminiError` messages."""
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if status == 429:
        raise GeminiError("The online AI is rate limited right now.")
    if status in (400, 401, 403):
        raise GeminiError("The online AI could not authenticate.")
    if isinstance(status, int) and status >= 500:
        raise GeminiError("The online AI service is having problems.")
    if "api key" in str(exc).lower():
        raise GeminiError("The online AI could not authenticate.")
    logger.warning("Gemini SDK error: %s: %s", type(exc).__name__, exc)
    raise GeminiError("The online AI request failed.")


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
            raise GeminiError("The online AI timed out.")
        if isinstance(exc, OSError) or "connect" in error_name or "network" in error_name:
            raise GeminiError("Could not reach the online AI service.")
        raise  # unknown failure - generate_response retries through REST

    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise GeminiError("The model returned an empty response.")
    return str(text).strip()


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
    if response.status_code == 429:
        raise GeminiError("The online AI is rate limited right now.")
    if response.status_code in (400, 401, 403):
        raise GeminiError("The online AI could not authenticate.")
    if response.status_code >= 500:
        raise GeminiError("The online AI service is having problems.")
    if not response.ok:
        raise GeminiError("The online AI request failed.")

    data = response.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Unexpected Gemini payload shape: %s", exc)
        raise GeminiError("The online AI returned an unexpected response.")

    if not text.strip():
        raise GeminiError("The model returned an empty response.")
    return text.strip()


def generate_response(
    user_message: str,
    conversation_history: Optional[Iterable[dict]] = None,
    system_instruction: Optional[str] = None,
) -> str:
    """Return an answer from Gemini for ``user_message``.

    ``system_instruction`` replaces the default ``config.SYSTEM_PROMPT`` when
    provided; ``None`` keeps the current behaviour exactly.

    Raises :class:`GeminiError` on any failure so callers can fall back to
    the offline knowledge base.
    """
    if not is_configured():
        raise GeminiError("No API key configured.")
    if not user_message or not user_message.strip():
        raise GeminiError("Empty message.")

    history = _normalise_history(conversation_history)

    if _SDK_AVAILABLE:
        try:
            return _generate_via_sdk(
                user_message.strip(), history, system_instruction
            )
        except GeminiError:
            raise
        except Exception:
            logger.exception("Unexpected Gemini SDK failure")

    try:
        return _generate_via_rest(
            user_message.strip(), history, system_instruction
        )
    except GeminiError:
        raise
    except requests.exceptions.Timeout as exc:
        logger.warning("Gemini timeout: %s", exc)
        raise GeminiError("The online AI timed out.")
    except requests.exceptions.ConnectionError as exc:
        logger.warning("Gemini connection error: %s", exc)
        raise GeminiError("Could not reach the online AI service.")
    except Exception:
        logger.exception("Unexpected Gemini failure")
        raise GeminiError("The online AI failed unexpectedly.")
