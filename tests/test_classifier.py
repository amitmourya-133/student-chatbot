"""Tests for the trained intent classifier."""

import pytest

from nlp.classifier import IntentClassifier


@pytest.fixture(scope="module")
def classifier() -> IntentClassifier:
    clf = IntentClassifier()
    assert clf.ensure_ready(), "model must be loadable (auto-trains if missing)"
    return clf


def test_model_files_exist(classifier):
    assert classifier.model_files_exist


def test_greeting_intent(classifier):
    assert classifier.predict("hello")["intent"] == "greet"
    assert classifier.predict("good morning")["intent"] == "greet"


def test_code_intent_hinglish(classifier):
    assert classifier.predict("python kya hai")["intent"] == "code"
    assert classifier.predict("how do I learn python")["intent"] == "code"


def test_quiz_and_planner_intents(classifier):
    assert classifier.predict("start a quiz")["intent"] == "quiz"
    assert classifier.predict("study plan bana do")["intent"] == "planner"


def test_math_and_science_intents(classifier):
    assert classifier.predict("explain quadratic equation")["intent"] == "math"
    assert classifier.predict("explain photosynthesis")["intent"] == "science"


def test_goodbye_intent(classifier):
    assert classifier.predict("bye")["intent"] == "goodbye"


def test_gibberish_is_fallback(classifier):
    result = classifier.predict("asdfqwerzxcv 12345 !@#$%")
    assert result["intent"] == "fallback"
    assert result["fallback"] is True


def test_confidence_is_bounded(classifier):
    result = classifier.predict("hello")
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence"] >= classifier.confidence_threshold


def test_empty_input_is_fallback(classifier):
    assert classifier.predict("")["intent"] == "fallback"
    assert classifier.predict("   ")["intent"] == "fallback"


def test_known_intents_populated(classifier):
    intents = classifier.known_intents()
    assert "greet" in intents
    assert "fallback" in intents
