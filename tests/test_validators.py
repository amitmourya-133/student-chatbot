"""Tests for input/path/secret validation helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.validators import (
    clamp_text,
    is_safe_extension,
    is_within_directory,
    redact_secrets,
    sanitize_filename,
)


def test_clamp_text_short_unchanged():
    assert clamp_text("hi", limit=10) == "hi"


def test_clamp_text_long_truncated():
    out = clamp_text("a" * 50, limit=10)
    assert len(out) == 10


def test_clamp_text_non_string():
    assert clamp_text(None) == ""


def test_sanitize_strips_windows_path():
    out = sanitize_filename(r"C:\Users\evil\..\notes.pdf")
    assert out == "notes.pdf"
    assert "/" not in out and "\\" not in out


def test_sanitize_strips_posix_path():
    assert sanitize_filename("../../etc/passwd") in ("passwd", "etc")


def test_sanitize_removes_control_and_unsafe_chars():
    assert sanitize_filename('my<file>:*.pdf') == "myfile.pdf"


def test_sanitize_empty_returns_default():
    assert sanitize_filename("", default="doc") == "doc"
    assert sanitize_filename("   ", default="doc") == "doc"


def test_sanitize_truncates_long_name():
    out = sanitize_filename("n" * 300 + ".pdf")
    assert len(out) <= 100


def test_is_within_directory_true():
    base = Path("C:/projects") / "uploads"
    assert is_within_directory(base, base / "a.pdf")


def test_is_within_directory_false_when_escaping():
    base = Path("C:/projects") / "uploads"
    assert not is_within_directory(base, base / ".." / "secret")


def test_redact_google_key():
    text = "config key is AIzaSyBogeys1234567890abcdef and more"
    out = redact_secrets(text)
    assert "AIzaSyBogeys" not in out
    assert "[REDACTED]" in out
    assert "and more" in out


def test_redact_api_key_assignment():
    out = redact_secrets("GEMINI_API_KEY=AIzaZZZZZZZMySecret")
    assert "MySecret" not in out
    assert "GEMINI_API_KEY=" not in out or "[REDACTED]" in out


def test_redact_non_secret_text_unchanged():
    text = "Explain DBMS normalization with examples"
    assert redact_secrets(text) == text


def test_redact_none_returns():
    assert redact_secrets(None) == ""


def test_is_safe_extension():
    assert is_safe_extension(".pdf", {".pdf", ".txt"})
    assert is_safe_extension("PDF", {".pdf"})
    assert not is_safe_extension(".exe", {".pdf", ".txt"})
    assert not is_safe_extension("", {".pdf"})