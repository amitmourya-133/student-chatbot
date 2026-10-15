"""Phase 3: verify the release ZIP integrity, structure, and secret-free contents.

Usage: python scripts/release_verify_zip.py <zip_path> <report_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

FORBIDDEN = {
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
FORBIDDEN_PREFIXES = (".",)
FORBIDDEN_SUBSTR = ("/.git", "/.pytest_cache", "/__pycache__")


def scan(zip_path: str, report_dir: Path) -> dict:
    result: dict = {}
    zf = zipfile.ZipFile(zip_path)
    infos = zf.infolist()
    result["zip_path"] = zip_path
    result["file_count"] = len(infos)
    result["size_bytes"] = Path(zip_path).stat().st_size
    result["bomb_check"] = max((i.file_size for i in infos), default=0) <= 4 * 1024 * 1024 * 1024

    integrity_errors = zf.testzip()
    result["integrity"] = "OK" if integrity_errors is None else f"FAIL: {integrity_errors}"

    names = [i.filename for i in infos]
    result["exe_files"] = [n for n in names if n.endswith(".exe")]
    result["has_cli_exe"] = any(n.endswith("StudentAIAssistant.exe") for n in names)
    result["has_web_exe"] = any(n.endswith("StudentAIAssistantWeb.exe") for n in names)
    result["has_models"] = any("models/" in n and n.endswith(".pkl") for n in names)
    result["has_kb"] = any(n.endswith("offline_kb.json") for n in names)
    result["has_intents"] = any(n.endswith("intents.json") for n in names)
    result["has_webapp"] = any("webapp/templates/index.html" in n for n in names)
    result["has_static"] = any("webapp/static" in n for n in names)

    root_entries = sorted({n.split("/")[0] for n in names})
    result["top_level_entries"] = root_entries

    forbidden = [n for n in names if Path(n).name in FORBIDDEN or n.lower().startswith(FORBIDDEN_PREFIXES) or any(s in n for s in FORBIDDEN_SUBSTR)]
    result["forbidden"] = forbidden

    digest = hashlib.sha256()
    with Path(zip_path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    result["sha256"] = digest.hexdigest()

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "zip_report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    zip_path = sys.argv[1]
    report_dir = Path(sys.argv[2])
    scan(zip_path, report_dir)