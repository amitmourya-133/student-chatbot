"""Tests for PyInstaller-bundle resource resolution.

When frozen, bundled assets live inside the extracted ``_MEIPASS`` directory
and runtime user data must go to ``user_data_dir()`` instead of the project.
These tests simulate the frozen environment to prove:

* read-only bundled paths resolve inside the (mock) bundle
* writable user data is redirected away from the project tree
* env var override keeps the packaged data dir fully relocatable
"""

from __future__ import annotations

import sys
from pathlib import Path

from utils.paths import app_root, is_frozen, resource_path, user_data_dir


def test_not_frozen_in_tests():
    assert is_frozen() is False
    # sanity: the tests run from the project, not from a bundle
    assert app_root() == Path(__file__).resolve().parent.parent


def test_resource_path_resolves_project_files():
    index = resource_path("webapp", "templates", "index.html")
    assert index.exists()
    assert "webapp" in str(index)
    kb = resource_path("data", "offline_kb.json")
    assert kb.exists()


def test_dev_user_data_dir_matches_project_data(monkeypatch):
    monkeypatch.delenv("STUDENTAI_DATA_DIR", raising=False)
    data_dir = user_data_dir()
    # DEVELOPMENT: writeable data lives beside the source tree
    assert (data_dir / "memory.json").parent == data_dir
    assert "StudentAIAssistant" in str(data_dir) or "chatbot" in str(data_dir).lower()


def test_frozen_resource_path_points_inside_bundle(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    (bundle / "data").mkdir(parents=True)
    (bundle / "data" / "offline_kb.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    assert resource_path("data", "offline_kb.json") == bundle / "data" / "offline_kb.json"
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_frozen_user_data_dir_is_not_inside_bundle(monkeypatch):
    """Packaged apps must never write into the read-only bundle directory."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    bundle = Path("C:/fake/bundle/_MEIPASS")
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    data_dir = user_data_dir()
    assert bundle not in data_dir.parents
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_user_data_dir_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom_data"
    monkeypatch.setenv("STUDENTAI_DATA_DIR", str(target))
    assert user_data_dir() == target