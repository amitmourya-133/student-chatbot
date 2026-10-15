"""Tests for services/memory.py and services/response_policy.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from services import memory as memory_mod
from services.response_policy import (
    compose_system,
    DEFAULT_STUDY_MODE,
    LANGUAGES,
    STYLES,
    STUDY_MODES,
    valid_study_modes,
    valid_styles,
    valid_languages,
    study_mode_label,
)


class TestFactPatterns:
    def test_name_pattern(self):
        mem = memory_mod.ConversationMemory.__new__(memory_mod.ConversationMemory)
        # just test the compiled pattern existence
        assert memory_mod._FACT_PATTERNS["name"].pattern is not None

    def test_subjects_pattern(self):
        assert memory_mod._FACT_PATTERNS["subjects"].pattern is not None

    def test_subject_pattern(self):
        assert memory_mod._FACT_PATTERNS["subject"].pattern is not None

    def test_language_pattern(self):
        assert memory_mod._LANG_PATTERN.pattern is not None

    def test_difficulty_pattern(self):
        assert memory_mod._DIFF_PATTERN.pattern is not None


class TestConversationMemoryInit:
    def test_init_no_path(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        assert m.facts == {}
        assert m.session == []
        assert m.disabled is False

    def test_load_corrupt(self, tmp_path):
        (tmp_path / "memory.json").write_text("not json")
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        assert m.facts == {}
        assert m.disabled is False

    def test_save_and_load_cycle(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.extract_facts("my name is Anuj")
        assert "name" in m.facts
        assert m.facts["name"] == "Anuj"
        # save/reload
        m._save()
        m2 = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        assert m2.facts.get("name") == "Anuj"
        assert m2.session == []

    def test_record(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.record("hi", "hello!")
        assert len(m.session) == 2
        assert m.session[0] == "hi"
        assert m.session[1] == "hello!"


class TestQuestionPatterns:
    def test_name_question(self):
        assert memory_mod._QUESTION_PATTERNS["name"].search("What is my name?")

    def test_subject_question(self):
        assert memory_mod._QUESTION_PATTERNS["subject"].search(
            "What is my favourite subject?"
        )

    def test_language_question(self):
        assert memory_mod._QUESTION_PATTERNS["language"].search(
            "What is my preferred language?"
        )

    def test_difficulty_question(self):
        assert memory_mod._QUESTION_PATTERNS["difficulty"].search(
            "What is my preferred difficulty level?"
        )

    def test_no_question(self):
        assert not memory_mod._QUESTION_PATTERNS["name"].search("hello there")


class TestMemoryAnswerQuestion:
    def test_returns_name_when_set(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.facts["name"] = "Anuj"
        assert m.answer_question("What is my name?") == "Your name is Anuj."

    def test_returns_none_when_not_set(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        assert m.answer_question("What is my name?") is None

    def test_returns_none_when_disabled(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.set_enabled(False)
        assert m.answer_question("What is my name?") is None

    def test_returns_subject_when_set(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.facts["subject"] = "DBMS"
        result = m.answer_question("What is my subject?")
        assert "DBMS" in result

    def test_returns_language_when_set(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.facts["language"] = "hindi"
        result = m.answer_question("What is my language?")
        assert "hindi" in result

    def test_difficulty_answer(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.facts["difficulty"] = "advanced"
        result = m.answer_question("What is my preferred difficulty level?")
        assert "advanced" in result


class TestFactSensitivity:
    def test_sensitive_password_not_stored(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.extract_facts("my password is abc123")
        assert "password" not in m.facts

    def test_sensitive_credit_card_not_stored(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.extract_facts("my credit card number is 1234-5678")
        # "credit card" triggers sensitivity so stored value should be skipped
        assert "credit" not in m.facts and "card" not in m.facts

    def test_not_sensitive_normal_text(self, tmp_path):
        m = memory_mod.ConversationMemory(path=tmp_path / "memory.json")
        m.extract_facts("explain normalization in dbms")
        # no facts triggered, fine
        assert "normalization" not in str(m.facts)


class TestResponsePolicy:
    def test_compose_system_uses_base(self):
        out = compose_system(base="YOU ARE A HELPFUL ASSISTANT")
        assert out.startswith("YOU ARE A HELPFUL ASSISTANT")

    def test_compose_system_ignores_empty_extra(self):
        out = compose_system()
        assert "You are Student AI Assistant" in out  # defaults from config

    def test_compose_system_includes_memory(self):
        out = compose_system(
            memory_lines=["name: Anuj", "subject: DBMS"],
            study_mode="general",
            language="english",
            style="normal",
        )
        assert "name: Anuj" in out or "subject: DBMS" in out

    def test_compose_system_includes_study_mode(self):
        out = compose_system(
            study_mode="mathematics",
        )
        assert "Mathematics" in out or "Study mode" in out

    def test_valid_study_modes(self):
        modes = valid_study_modes()
        assert "general" in modes
        assert "mathematics" in modes

    def test_valid_styles(self):
        styles = valid_styles()
        assert "normal" in styles
        assert "short" in styles

    def test_valid_languages(self):
        langs = valid_languages()
        assert "english" in langs
        assert "hindi" in langs

    def test_study_mode_label(self):
        assert study_mode_label("mathematics") == "Mathematics"
        assert study_mode_label("general") == "General"


class TestLangsStylesDicts:
    def test_langs_dict_has_expected_keys(self):
        for lang in LANGUAGES:
            assert lang in LANGUAGES

    def test_styles_dict_has_expected_keys(self):
        for s in STYLES:
            assert s in STYLES

    def test_study_modes_dict_has_expected_keys(self):
        for mode in STUDY_MODES:
            assert mode in STUDY_MODES


class TestLangsStylesDefaults:
    def test_default_study_mode(self):
        assert DEFAULT_STUDY_MODE == "general"

    def test_default_language(self):
        assert "english" in LANGUAGES

    def test_default_style(self):
        assert "normal" in STYLES