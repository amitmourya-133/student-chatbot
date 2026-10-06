"""Tests for nlp.preprocessing."""

from nlp.preprocessing import preprocess, tokenize


def test_lowercases_and_strips():
    assert preprocess("  How Do I Learn Python??? ") == "how do i learn python"


def test_collapses_repeated_spaces():
    assert preprocess("hello     world") == "hello world"


def test_reduces_repeated_punctuation():
    assert preprocess("wow!!!") == "wow"
    assert preprocess("what??") == "what"
    assert preprocess("really?!?!") == "really"
    assert preprocess("why?") == "why"


def test_removes_quotes_and_commas():
    assert preprocess('what is a "dictionary", really?') == "what is a dictionary really"


def test_keeps_meaningful_symbols():
    cleaned = preprocess("what is c++ and a/b?")
    assert "c++" in cleaned
    assert "a/b" in cleaned


def test_keeps_programming_keywords():
    assert "list" in preprocess("difference between list and tuple?")


def test_handles_hinglish():
    assert preprocess("  Python Kya Hai  ") == "python kya hai"


def test_handles_empty_and_non_string():
    assert preprocess("") == ""
    assert preprocess("   ") == ""
    assert preprocess(None) == ""
    assert preprocess(12345) == ""


def test_tokenize():
    assert tokenize("Hello   World") == ["hello", "world"]
    assert tokenize("!!!") == ["!"] or tokenize("!!!") == []
