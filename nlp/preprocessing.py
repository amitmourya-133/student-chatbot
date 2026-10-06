"""Text preprocessing for the offline intent classifier.

Rules:
    * lower-case everything
    * strip outer whitespace and collapse repeated whitespace
    * drop repeated punctuation (``???`` -> ``?``) and trailing marks
    * remove decorative punctuation but keep programming-relevant symbols
      (``+ - = / _ . # < >``) and brackets
    * keep devanagari characters so multilingual data survives
"""

from __future__ import annotations

import re

# collapse whitespace
_WS_RE = re.compile(r"\s+")

# repeated punctuation such as !!! or ??? -> single character
_REPEAT_PUNCT_RE = re.compile(r"([!?.]){2,}")

# decorative punctuation that carries no meaning for intent matching
_STRIP_RE = re.compile(r"[\"'`~@ $%^&*|\\;,\u2018\u2019\u201c\u201d]")

# trailing sentence punctuation is dropped ("python???" -> "python")
_TRAILING_PUNCT = "?!.,;:"


def preprocess(text: str) -> str:
    """Normalise ``text`` for TF-IDF / keyword matching.

    Returns an empty string for non-string input or input that contained
    nothing but punctuation.
    """
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = _REPEAT_PUNCT_RE.sub(r"\1", text)
    text = _STRIP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    text = text.rstrip(_TRAILING_PUNCT).strip()
    return text


def tokenize(text: str) -> list[str]:
    """Split preprocessed text into lowercase word tokens."""
    cleaned = preprocess(text)
    return [token for token in cleaned.split(" ") if token]
