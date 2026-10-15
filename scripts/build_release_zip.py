"""Build the release ZIP containing BOTH packaged EXEs.

Layout (each onedir build stays self-contained so its ``_internal`` tree does
not collide):

    StudentAIAssistant/StudentAIAssistant.exe
    StudentAIAssistant/_internal/...
    StudentAIAssistantWeb/StudentAIAssistantWeb.exe
    StudentAIAssistantWeb/_internal/...

The previous ZIP is preserved with a timestamped ``.bak`` suffix before it is
replaced. Forbidden payloads (.env, memory/chat/feature JSON, .git) are never
added, and the resulting archive is integrity-checked.

Usage: python scripts/build_release_zip.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
ZIP_PATH = DIST / "StudentAIAssistant_release.zip"

PACKAGES = [
    ("StudentAIAssistant", "StudentAIAssistant.exe"),
    ("StudentAIAssistantWeb", "StudentAIAssistantWeb.exe"),
]

FORBIDDEN_NAMES = {
    ".env",
    "memory.json",
    "chat_log.json",
    "profile.json",
    "conversations.json",
    "progress.json",
    "feedback.json",
    "analytics.json",
    "flashcards.json",
    "planner.json",
    "web_launcher.port",
}
FORBIDDEN_SUBSTR = ("\\.git\\", "/.git/", "__pycache__", ".pytest_cache")


def _forbidden(rel: Path) -> bool:
    name = rel.name
    joined = "\\" + str(rel).replace("/", "\\") + "\\"
    if name in FORBIDDEN_NAMES:
        return True
    if name == ".env.example":
        return False
    if name.startswith(".") and name not in (".env.example",):
        return True
    return any(s in joined for s in FORBIDDEN_SUBSTR)


def main() -> int:
    missing = [
        f"{pkg}/{exe}" for pkg, exe in PACKAGES
        if not (DIST / pkg / exe).exists()
    ]
    if missing:
        print("BUILD FAILED: missing build outputs: " + ", ".join(missing))
        print("Run: python scripts/build_exe.py")
        return 1

    # 1. preserve the previous known-good release
    if ZIP_PATH.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = ZIP_PATH.with_suffix(ZIP_PATH.suffix + f".bak-{stamp}")
        shutil.copy2(ZIP_PATH, backup)
        print(f"[backup] {ZIP_PATH.name} -> {backup.name}")

    added = 0
    skipped = []
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for pkg, exe in PACKAGES:
            base = DIST / pkg
            for path in sorted(base.rglob("*")):
                if path.is_dir():
                    continue
                rel = path.relative_to(DIST)  # keeps pkg/ prefix
                if _forbidden(rel):
                    skipped.append(str(rel))
                    continue
                zf.write(path, rel.as_posix())
                added += 1

    # 2. integrity + structural checks
    zf = zipfile.ZipFile(ZIP_PATH)
    bad = zf.testzip()
    names = zf.namelist()
    has_cli = any(n.endswith("StudentAIAssistant/StudentAIAssistant.exe") for n in names)
    has_web = any(n.endswith("StudentAIAssistantWeb/StudentAIAssistantWeb.exe") for n in names)

    digest = hashlib.sha256()
    with ZIP_PATH.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    print(f"[zip] {ZIP_PATH}")
    print(f"[zip] files={added} size={ZIP_PATH.stat().st_size:,} bytes")
    print(f"[zip] integrity={'OK' if bad is None else 'FAIL ' + str(bad)}")
    print(f"[zip] has_cli_exe={has_cli} has_web_exe={has_web}")
    print(f"[zip] sha256={digest.hexdigest()}")
    if skipped:
        print(f"[zip] skipped forbidden files ({len(skipped)}): {skipped[:5]}")
    if bad is not None or not (has_cli and has_web):
        print("BUILD FAILED: archive is missing an EXE or is corrupt")
        return 1
    print("RELEASE ZIP OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())