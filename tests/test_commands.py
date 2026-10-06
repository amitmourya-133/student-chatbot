"""Tests for slash commands, the planner helpers and chat history."""

import json

import pytest

import config
from features import commands
from features import planner
from services.chat_history import ChatHistory


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def test_parse_simple_commands():
    assert commands.parse_command("/help") == ("/help", "")
    assert commands.parse_command("  /mode  ") == ("/mode", "")
    assert commands.parse_command("/exit") == ("/exit", "")
    assert commands.parse_command("/quiz") == ("/quiz", "")


def test_parse_command_with_arguments():
    name, args = commands.parse_command("/quiz python")
    assert name == "/quiz"
    assert args == "python"


def test_parse_plain_text_is_not_a_command():
    assert commands.parse_command("hello there") == (None, "hello there")
    assert commands.parse_command("") == (None, "")
    assert commands.is_command("what is python") is False


def test_unknown_slash_input_is_still_a_command():
    name, args = commands.parse_command("/frobnicate now")
    assert name == "/frobnicate"
    assert args == "now"
    assert name not in commands.COMMANDS


def test_slash_alone_is_not_a_command():
    assert commands.parse_command("/") == (None, "/")


def test_is_command():
    assert commands.is_command("/planner") is True
    assert commands.is_command("/history") is True
    assert commands.is_command("planner") is False


@pytest.mark.parametrize(
    "text",
    ["quit", "exit", "bye", "goodbye", "/exit", "/quit", "EXIT", "  Bye!  "],
)
def test_exit_detection(text):
    assert commands.is_exit(text) is True


@pytest.mark.parametrize(
    "text", ["hello", "exit code", "biology", "", "history", "quick"],
)
def test_non_exit_input(text):
    assert commands.is_exit(text) is False


def test_help_text_lists_every_command():
    text = commands.help_text()
    for name in commands.COMMANDS:
        assert name in text
    assert "Available Commands" in text


def test_help_text_matches_spec_layout():
    text = commands.help_text()
    assert "/help" in text
    assert "/quiz" in text
    assert "/planner" in text
    assert "/history" in text
    assert "/mode" in text
    assert "/clear" in text
    assert "/exit" in text


def test_unknown_command_message():
    message = commands.unknown_command_message("/wat")
    assert "/wat" in message
    assert "/help" in message


# --------------------------------------------------------------------------- #
# planner helpers
# --------------------------------------------------------------------------- #
def test_parse_subjects():
    assert planner.parse_subjects("python, maths and science") == [
        "python",
        "maths",
        "science",
    ]
    assert planner.parse_subjects("") == []


def test_parse_number():
    assert planner.parse_number("3") == 3
    assert planner.parse_number("2.5 hours") == 2.5
    assert planner.parse_number("two") == 2
    assert planner.parse_number("abc") is None


def test_generate_plan_output():
    plan = planner.generate_plan(["Python", "Maths"], hours_per_day=3, days=2)
    assert "DAY 1" in plan
    assert "DAY 2" in plan
    assert "Python" in plan
    assert "Maths" in plan
    assert "09:00" in plan
    assert plan.count("DAY") >= 2


def test_generate_plan_rejects_bad_input():
    with pytest.raises(ValueError):
        planner.generate_plan([], 3, 5)
    with pytest.raises(ValueError):
        planner.generate_plan(["Python"], 0, 5)
    with pytest.raises(ValueError):
        planner.generate_plan(["Python"], 3, 0)


def test_plan_slots_fit_available_hours():
    plan = planner.generate_plan(["Python"], hours_per_day=2, days=1)
    day_block = plan.split("TIPS")[0]
    starts = [
        line.split(" - ")[0]
        for line in day_block.splitlines()
        if " - " in line and ":" in line.split(" - ")[0]
    ]
    assert starts[0] == "09:00"
    # 2 hours from 09:00 must finish by 11:00 at the latest
    assert max(starts) < "11:00"


# --------------------------------------------------------------------------- #
# chat history
# --------------------------------------------------------------------------- #
def test_history_creates_file_and_appends(tmp_path):
    log = tmp_path / "chat_log.json"
    history = ChatHistory(path=log)
    assert log.exists()
    assert len(history) == 0

    history.add("hi", "hello", "offline")
    history.add("what is python", "Python is a language", "offline")
    assert len(history) == 2

    reloaded = ChatHistory(path=log)
    assert len(reloaded) == 2
    assert reloaded.recent(limit=1)[0]["user"] == "what is python"


def test_history_recent_limit(tmp_path):
    history = ChatHistory(path=tmp_path / "chat_log.json")
    for index in range(15):
        history.add(f"msg {index}", "reply", "offline")
    recent = history.recent(limit=10)
    assert len(recent) == 10
    assert recent[-1]["user"] == "msg 14"


def test_history_handles_corrupted_file(tmp_path):
    log = tmp_path / "chat_log.json"
    log.write_text("{{{ broken", encoding="utf-8")
    history = ChatHistory(path=log)
    assert len(history) == 0
    history.add("hello", "hi", "online")
    data = json.loads(log.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["mode"] == "online"


def test_history_clear(tmp_path):
    history = ChatHistory(path=tmp_path / "chat_log.json")
    history.add("a", "b", "offline")
    history.clear()
    assert len(history) == 0


def test_history_entry_shape(tmp_path):
    history = ChatHistory(path=tmp_path / "chat_log.json")
    entry = history.add("q", "a", "online")
    assert set(entry) == {"timestamp", "user", "assistant", "mode"}
    assert entry["mode"] == "online"
    assert entry["user"] == "q"
    assert entry["assistant"] == "a"


def test_config_has_no_hardcoded_key():
    assert config.GEMINI_API_KEY is None or not config.GEMINI_API_KEY.startswith("AIza")
    assert "your_api_key_here" not in (config.GEMINI_API_KEY or "")
