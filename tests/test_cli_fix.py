"""Regression tests for the CLI "Something went wrong" bug fix.

Root cause: process() printed the AI answer, then history.add() and
memory.record() called _save() which could raise. The exception reached
run()'s except block and printed "Something went wrong" after a valid
answer had already been displayed. The fix wraps those saves in try/except
inside process() so a secondary failure never misrepresents a successful
AI response.
"""
from __future__ import annotations

from unittest import mock

from main import StudentChatbot


def test_cli_no_false_warning_after_successful_answer(capsys):
    """A failure in history/memory saving must not print 'Something went wrong'."""
    bot = StudentChatbot()
    bot.pipeline.online = False
    bot.pipeline.online_disabled = False

    with mock.patch.object(bot.pipeline.history, "add", side_effect=OSError("disk full")):
        with mock.patch.object(bot.pipeline.memory, "record", side_effect=OSError("disk full")):
            with mock.patch.object(bot, "get_reply", return_value=("NEWTON ANSWER", "offline")):
                bot.process("what is database")

    captured = capsys.readouterr()
    # The assistant answer must be displayed exactly once (one input -> one response)
    assert captured.out.count("NEWTON ANSWER") == 1
    assert "Something went wrong" not in captured.out


def test_run_loop_continues_after_history_failure(capsys):
    """After a history/memory failure, the next input must still be processed."""
    bot = StudentChatbot()
    bot.pipeline.online = False

    inputs = iter(["hello", "exit"])
    with mock.patch.object(bot.pipeline.history, "add", side_effect=OSError("disk full")):
        with mock.patch.object(bot.pipeline.memory, "record", side_effect=OSError("disk full")):
            with mock.patch.object(bot, "_read", side_effect=lambda *a: next(inputs)):
                with mock.patch.object(bot, "get_reply", return_value=("OK REPLY", "offline")):
                    bot.run()

    captured = capsys.readouterr()
    assert "OK REPLY" in captured.out
    assert "Something went wrong" not in captured.out


def test_genuine_pipeline_failure_still_propagates():
    """A genuine pipeline exception should still surface (not be swallowed)."""
    bot = StudentChatbot()
    bot.pipeline.online = False

    with mock.patch.object(bot, "get_reply", side_effect=RuntimeError("real failure")):
        try:
            bot.process("boom")
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected RuntimeError to propagate")