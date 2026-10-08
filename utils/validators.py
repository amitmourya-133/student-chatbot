"""Input/path/secret validation helpers shared across the application.

Every helper is defensive: bad input returns a safe value instead of raising.
"""

from __future__ import annotations

import re
from pathlib import Path

MAX_INPUT_LENGTH = 2000
MAX_FILENAME_LENGTH = 100

_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)api[_-]?key\s*[=:]\s*[\"']?([A-Za-z0-9._\-]{8,})"),
]

_UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def clamp_text(text: str, limit: int = MAX_INPUT_LENGTH) -> str:
    """Truncate overly long input so prompts and logs stay bounded."""
    if not isinstance(text, str):
        return ""
    if len(text) <= limit:
        return text
    return text[:limit]


def sanitize_filename(name: str, default: str = "document") -> str:
    """Reduce user-supplied filenames to a single safe basename.

    Removes directory components, unsafe characters and control characters,
    and prevents hidden/extension-only names. Never returns a path.
    """
    if not isinstance(name, str) or not name.strip():
        return default
    # take the final path component on either separator style
    base = re.split(r"[\\/]", name.strip())[-1]
    base = _UNSAFE_FILENAME_CHARS.sub("", base)
    base = base.strip(" .")
    if not base or set(base) <= {"."}:
        return default
    if base.startswith("."):
        base = base.lstrip(".")
    if not base:
        return default
    if len(base) > MAX_FILENAME_LENGTH:
        stem, dot, ext = base.rpartition(".")
        if dot and 0 < len(ext) <= 10:
            keep = MAX_FILENAME_LENGTH - len(ext) - 1
            base = stem[:keep] + "." + ext
        else:
            base = base[:MAX_FILENAME_LENGTH]
    return base or default


def is_within_directory(base: Path, candidate: Path) -> bool:
    """True when ``candidate`` resolves inside ``base`` (path traversal guard)."""
    try:
        resolved_base = Path(base).resolve()
        resolved_candidate = Path(candidate).resolve()
    except (OSError, RuntimeError):
        return False
    try:
        resolved_candidate.relative_to(resolved_base)
        return True
    except ValueError:
        return False


def redact_secrets(text: str) -> str:
    """Strip anything that looks like a credential from free text."""
    if not isinstance(text, str) or not text:
        return "" if not isinstance(text, str) else text
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result


def is_safe_extension(ext: str, allowed: set[str]) -> bool:
    """True when ``ext`` (with or without dot) is in the allow-list."""
    if not ext:
        return False
    clean = ext.lower().lstrip(".")
    return clean in {item.lower().lstrip(".") for item in allowed}
