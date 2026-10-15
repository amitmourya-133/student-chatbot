# Building the Windows EXE (Student AI Assistant)

The assistant is a **console** app by design. Two EXE targets are produced:

| Target | Entry point | What it runs |
| --- | --- | --- |
| `dist/StudentAIAssistant/StudentAIAssistant.exe` | `main.py` | The interactive CLI chatbot (recommended default) |
| `dist/StudentAIAssistantWeb/StudentAIAssistantWeb.exe` | `web_launcher.py` | Local web app on `http://127.0.0.1:8000` |

## Prerequisites

- Windows 10/11, 64-bit
- Python 3.11+ on PATH as `python`
- Install build + runtime dependencies:

```powershell
pip install -r requirements.txt
pip install pyinstaller>=6.0
```

## Build

From the project root:

```powershell
python scripts/build_exe.py
```

This runs `pyinstaller StudentAIAssistant.spec --noconfirm --clean` and
`pyinstaller StudentAIAssistantWeb.spec --noconfirm --clean` and then checks
that no `.env` or private JSON payloads leaked into `dist\`.

Output:
- `dist\StudentAIAssistant\StudentAIAssistant.exe`
- `dist\StudentAIAssistantWeb\StudentAIAssistantWeb.exe`

Or build either one on its own:

```powershell
pyinstaller StudentAIAssistant.spec --noconfirm --clean
pyinstaller StudentAIAssistantWeb.spec --noconfirm --clean
```

## What is bundled

Everything the app needs at runtime, versioned by the `.spec` `datas` list:

- `data/intents.json`, `data/offline_kb.json`
- `models/vectorizer.pkl`, `models/intent_model.pkl`
- `webapp/templates/`, `webapp/static/` (web EXE)
- Python runtime + dependencies (scikit-learn, numpy, google-genai, pypdf,
  fastapi, uvicorn, ...)

**Never bundled**: `.env`, API keys, passwords, Git metadata, chat logs,
personal memory, test artifacts. The build script verifies this after each
build. The `.env` lives next to the EXE (see below).

## Writable data location

Bundled resources are **read-only**. All runtime user data goes to the same
location in both modes:

- Developer mode (running from source): `data\` inside the project
- Packaged EXE: `%LOCALAPPDATA%\StudentAIAssistant\`

`config.py` decides via `utils/paths.py`. The choice is exposed as the env var
`STUDENTAI_DATA_DIR` if you need to relocate it. Memory, chat history,
progress, flashcards and uploaded PDFs live there and are never recreated into
the bundle.

## API key setup (recommended)

On first run of a packaged EXE, start it from a terminal:

```powershell
mkdir "%LOCALAPPDATA%\StudentAIAssistant" 2>NUL
copy NUL "%LOCALAPPDATA%\StudentAIAssistant\.env" 2>NUL
notepad "%LOCALAPPDATA%\StudentAIAssistant\.env"
```

Add your key (obtain it from https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your_key_here
```

The EXE reads a `.env` that sits **next to the EXE** too. If neither exists or
the key is invalid/missing, the assistant logs a sanitised error and falls back
to offline knowledge-base mode; it never prints the key.

## Offline mode

- Launch with `--offline` to skip all network checks.
- `--check` runs a startup self-test (directories, model, knowledge base,
  API-key presence) and exits — handy for verifying a package works.

## Files created/modified by this packaging

```
scripts/build_exe.py          reproducible build script + security check
StudentAIAssistant.spec       PyInstaller spec (console EXE)
StudentAIAssistantWeb.spec    PyInstaller spec (web EXE)
utils/paths.py                frozen/onedir vs source path resolution
utils/filelock.py             cross-process lock + atomic JSON writes
config.py                     path model (read-only bundle vs user data)
utils/logger.py               log dir now config-driven
services/chat_history.py       lock + atomic writes
services/memory.py            lock + atomic writes
web_launcher.py               single-instance 127.0.0.1 launcher
webapp/app.py                 absolute asset paths via config
requirements.txt              fastapi/uvicorn/pypdf added
```

## Troubleshooting

- **Missing DLL / `api-ms-win-crt` errors**: install the latest Visual C++
  Redistributable (x64) from Microsoft; PyInstaller bundles the VC runtime but
  a broken system VC run can still surface this.
- **`PermissionError` writing memory/history**: the data dir
  (`%LOCALAPPDATA%\StudentAIAssistant`) is blocked by policy or an AV lock.
  Move it via `STUDENTAI_DATA_DIR`; the app degrades gracefully and will not
  crash on an unwritable data dir.
- **Antivirus false positives**: rebuild without UPX
  (`upx=False` in the spec) because some AVs flag UPX-packed files. Do **not**
  disable your AV entirely.
- **Model missing in bundle**: `--check` reports it; the classifier falls back
  to offline keyword matching and logs a warning.
- **Web app won't open**: it binds `127.0.0.1` only. Confirm the port with the
  printed URL; re-running the launcher re-opens the existing instance instead
  of starting a second server (single-instance guard via named mutex).