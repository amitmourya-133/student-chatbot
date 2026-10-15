#!/usr/bin/env python3
"""Reproducible Windows build script for the Student AI Assistant.

Builds:
  dist/StudentAIAssistant/StudentAIAssistant.exe   (console CLI)
  dist/StudentAIAssistantWeb/StudentAIAssistantWeb.exe (web launcher)

Run from the project root:

    python scripts/build_exe.py

Requirements: ``pyinstaller`` installed in the active environment.
The EXE must never embed the developer's ``.env``; the ``datas`` lists in the
spec files explicitly exclude it (and all private memory/chat logs).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPECS = [
    "StudentAIAssistant.spec",
    "StudentAIAssistantWeb.spec",
]

FORBIDDEN_IN_DIST = {
    ".env",
    "chat_log.json",
    "memory.json",
    "profile.json",
    "conversations.json",
    "progress.json",
    "feedback.json",
    "analytics.json",
    "flashcards.json",
    "planner.json",
    ".git",
}


def _fail(message: str) -> int:
    print(f"BUILD FAILED: {message}", file=sys.stderr)
    return 1


def main() -> int:
    if not sys.platform.startswith("win"):
        print("NOTE: PyInstaller is supported on Windows; this is the target OS.")

    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        return _fail("pyinstaller is not installed. Run: pip install pyinstaller")

    for spec in SPECS:
        spec_path = ROOT / spec
        if not spec_path.exists():
            return _fail(f"missing spec file: {spec_path}")
        print(f"[build] {spec} -> dist/{Path(spec).stem or ''}")
        result = subprocess.run(
            [sys.executable, "-m", "PyInstaller", str(spec_path), "--noconfirm", "--clean"],
            cwd=str(ROOT),
            check=False,
        )
        if result.returncode != 0:
            return _fail(f"PyInstaller failed for {spec} (exit {result.returncode})")

    _verify_clean_dist()
    print("\nBUILD COMPLETE")
    print("  dist/StudentAIAssistant/StudentAIAssistant.exe")
    print("  dist/StudentAIAssistantWeb/StudentAIAssistantWeb.exe")
    return 0


def _verify_clean_dist() -> None:
    """Ensure no .env or private JSON payloads ended up in the build output."""
    issues: list[str] = []
    for dist_name in ("StudentAIAssistant", "StudentAIAssistantWeb"):
        folder = ROOT / "dist" / dist_name
        if not folder.exists():
            issues.append(f"missing dist folder: {folder}")
            continue
        for forbidden in FORBIDDEN_IN_DIST:
            if list(folder.rglob(forbidden)):
                issues.append(f"{dist_name} contains forbidden '{forbidden}'")
    if issues:
        print("\nSECURITY CHECK:")
        for issue in issues:
            print(f"  !! {issue}")
        print("  (investigate before distributing)")
    else:
        print("\nSECURITY CHECK: no .env or private JSON payloads found in dist output.")


if __name__ == "__main__":
    raise SystemExit(main())