"""Launch web EXE via subprocess (like monitor), then run the Playwright UI tests."""
from __future__ import annotations

import os
import subprocess
import sys
import time

EXE = r"D:\chatbot ai\dist\StudentAIAssistantWeb\StudentAIAssistantWeb.exe"
DATA = os.path.join(os.environ["TEMP"], "StudentAIData_UI3")
PORT = 8014
BASE = f"http://127.0.0.1:{PORT}"

os.makedirs(DATA, exist_ok=True)
env = dict(os.environ)
env["STUDENTAI_DATA_DIR"] = DATA

proc = subprocess.Popen(
    [EXE, "--no-browser", "--port", str(PORT)],
    env=env,
    stdout=open(os.path.join(DATA, "ui3_out.log"), "w"),
    stderr=open(os.path.join(DATA, "ui3_err.log"), "w"),
    cwd=DATA,
)
print(f"launched pid={proc.pid}")
try:
    deadline = time.monotonic() + 40
    ready = False
    while time.monotonic() < deadline:
        try:
            import urllib.request
            with urllib.request.urlopen(BASE + "/api/health", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)
    if not ready:
        print("SERVER NOT READY")
        sys.exit(2)
    print("server ready, running UI tests...")
    code = subprocess.call([sys.executable, r"D:\chatbot ai\scripts\release_verify_ui.py", BASE, r"D:\chatbot ai\reports\release-verify-20261013-001118\ui_report.json"])
    print(f"ui_script_exit={code} server_alive={proc.poll() is None}")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"server final exit={proc.returncode}")