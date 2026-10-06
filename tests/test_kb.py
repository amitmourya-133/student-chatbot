"""Tests for the offline knowledge base search."""

import json

import pytest

from services.knowledge_base import NO_MATCH_MESSAGE, KnowledgeBase


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    store = KnowledgeBase()
    assert store.load(), "knowledge base must load"
    return store


def test_loads_at_least_100_entries(kb):
    assert kb.size >= 100


def test_threshold_is_0_30(kb):
    assert kb.threshold == 0.30


def test_matches_python_question(kb):
    match = kb.find_answer("What is Python?")
    assert match is not None
    assert "Python" in match.answer
    assert match.score >= 0.30


def test_matches_hinglish_question(kb):
    match = kb.find_answer("python kya hai")
    assert match is not None
    assert match.score >= 0.30
    assert len(match.answer) > 30


def test_matches_sql_question(kb):
    match = kb.find_answer("what is a primary key in sql")
    assert match is not None
    assert match.score >= kb.threshold


def test_nonsense_query_below_threshold(kb):
    matches = kb.search("purple elephant submarine quantum banana peel dance zzz")
    assert matches == []
    assert kb.find_answer("purple elephant submarine quantum banana peel dance zzz") is None


def test_empty_query_returns_none(kb):
    assert kb.find_answer("") is None
    assert kb.find_answer("   ") is None


def test_search_results_sorted_by_score(kb):
    matches = kb.search("how do I learn python", top_n=3)
    assert matches
    scores = [m.score for m in matches]
    assert scores == sorted(scores, reverse=True)
    assert all(m.score >= kb.threshold for m in matches)


def test_intent_hint_can_boost(kb):
    hinted = kb.search("how do i learn python", top_n=1, intent_hint="code")
    unhinted = kb.search("how do i learn python", top_n=1)
    assert hinted and unhinted
    assert hinted[0].score >= unhinted[0].score


def test_no_match_message_is_friendly():
    assert "internet" in NO_MATCH_MESSAGE.lower()


def test_explain_python_in_simple_words_matches_python(kb):
    for hint in (None, "code"):
        match = kb.find_answer("Explain Python in simple words", intent_hint=hint)
        assert match is not None, f"no match for hint={hint}"
        assert "Python" in match.answer
        assert "responsive" not in match.question.lower()
        assert match.score >= kb.threshold


def test_explain_responsive_design_in_simple_words(kb):
    match = kb.find_answer("Explain responsive design in simple words")
    assert match is not None
    assert "responsive" in match.question.lower()
    assert match.score >= kb.threshold


def test_hinglish_database_kya_hai_matches_database(kb):
    match = kb.find_answer("database kya hai")
    assert match is not None
    assert "database" in match.question.lower()
    assert match.score >= kb.threshold


def test_hinglish_html_kya_hota_hai_matches_html(kb):
    match = kb.find_answer("html kya hota hai")
    assert match is not None
    assert "html" in match.question.lower()
    assert match.score >= kb.threshold


def test_exact_question_match_ranks_first(kb):
    match = kb.find_answer("python kya hai")
    assert match is not None
    assert match.question == "python kya hai"
    assert match.score == 1.0


def test_nonsense_query_with_intent_hint_still_below_threshold(kb):
    match = kb.find_answer(
        "purple elephant submarine quantum banana peel dance zzz",
        intent_hint="code",
    )
    assert match is None


def test_missing_file_is_handled(tmp_path):
    store = KnowledgeBase(path=tmp_path / "does_not_exist.json")
    assert store.load() is False
    assert store.find_answer("what is python") is None


def test_corrupted_json_is_handled(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json", encoding="utf-8")
    store = KnowledgeBase(path=bad)
    assert store.load() is False


def test_empty_kb_is_handled(tmp_path):
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"knowledge_base": []}), encoding="utf-8")
    store = KnowledgeBase(path=empty)
    assert store.load() is False
