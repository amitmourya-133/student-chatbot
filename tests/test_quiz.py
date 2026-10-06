"""Tests for the offline quiz feature."""

import pytest

from features import quiz


def test_list_subjects():
    subjects = quiz.list_subjects()
    assert len(subjects) >= 5
    for expected in ("Computer", "General Knowledge", "Math", "Python", "Science"):
        assert expected in subjects


@pytest.mark.parametrize(
        "text,expected",
    [
        ("python", "Python"),
        ("PYTHON", "Python"),
        ("maths", "Math"),
        ("mathematics", "Math"),
        ("gk", "General Knowledge"),
        ("computer science", "Computer"),
        ("science", "Science"),
        ("pottery", None),
        ("", None),
    ],
)
def test_normalise_subject(text, expected):
    assert quiz.normalise_subject(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", 0),
        ("b", 1),
        ("C.", 2),
        ("d", 3),
        ("1", 0),
        ("4", 3),
        ("option c", 2),
        ("", None),
        ("E", None),
        ("banana", None),
    ],
)
def test_parse_choice(text, expected):
    assert quiz.parse_choice(text) == expected


def test_start_quiz_valid_subject():
    session, message = quiz.start_quiz("python")
    assert session is not None
    assert session.subject == "Python"
    assert session.total == 5
    assert "Question 1/5" in message
    assert "A." in message


def test_start_quiz_unknown_subject():
    session, message = quiz.start_quiz("underwater basket weaving")
    assert session is None
    assert "Available subjects" in message


def test_quiz_scoring_and_completion():
    session, _ = quiz.start_quiz("Python")
    # answer everything correctly using the stored answer index
    while not session.is_complete:
        question = session.current
        result = session.submit(str(question["answer"] + 1))
        assert result["status"] == "answered"
        assert result["correct"] is True
    assert session.score == session.total
    assert session.percentage == 100
    result = session.result()
    assert "Quiz Complete!" in result
    assert f"Score: {session.total}/{session.total}" in result
    assert "Percentage: 100%" in result


def test_quiz_wrong_answer_scored():
    session, _ = quiz.start_quiz("Math")
    first = session.current
    wrong = (int(first["answer"]) + 1) % 4
    result = session.submit(str(wrong + 1))
    assert result["correct"] is False
    assert result["correct_letter"] in ("A", "B", "C", "D")
    assert session.score == 0
    assert session.answered == 1


def test_quiz_invalid_answer_is_rejected():
    session, _ = quiz.start_quiz("Science")
    result = session.submit("banana")
    assert result["status"] == "invalid"
    assert session.answered == 0
    assert session.index == 0


def test_quiz_partial_score_percentage():
    session, _ = quiz.start_quiz("Computer")
    answered = 0
    while not session.is_complete:
        question = session.current
        if answered == 0:
            wrong = (int(question["answer"]) + 1) % 4
            session.submit(str(wrong + 1))
        else:
            session.submit(str(int(question["answer"]) + 1))
        answered += 1
    assert session.score == session.total - 1
    assert session.percentage == 80
    assert "Percentage: 80%" in session.result()


def test_submit_after_completion():
    session, _ = quiz.start_quiz("General Knowledge")
    while not session.is_complete:
        session.submit(str(int(session.current["answer"]) + 1))
    assert session.submit("A") == {"status": "complete"}
    assert session.current is None


def test_quiz_prompt_contains_all_options():
    session, _ = quiz.start_quiz("Python")
    prompt = session.prompt()
    for letter in ("A.", "B.", "C.", "D."):
        assert letter in prompt


def test_quiz_skip_advances_without_scoring():
    session, _ = quiz.start_quiz("Python")
    result = session.skip()
    assert result["status"] == "skipped"
    assert session.score == 0
    assert session.index == 1
    assert session.answered == 1
    assert session.history[-1]["skipped"] is True
    assert not session.is_complete


def test_quiz_skip_past_end_reports_complete():
    session, _ = quiz.start_quiz("Math")
    while not session.is_complete:
        session.skip()
    assert session.score == 0
    assert session.submit("A") == {"status": "complete"}
