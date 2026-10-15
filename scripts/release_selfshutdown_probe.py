"""Controlled reproduction: does the packaged web EXE self-shutdown?

Launches exactly one instance via subprocess (cleanest equivalent of double-
click/Start-Process), tracks the exact PID, makes no requests, and waits.

Replicates multiple trials to measure the self-shutdown rate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

EXE = r"D:\chatbot ai\dist\StudentAIAssistantWeb\StudentAIAssistantWeb.exe"

def trial(n: int, seconds: int) -> dict:
    data = os.path.join(os.environ["TEMP"], f"StudentAITrial_{n}")
    os.makedirs(data, exist_ok=True)
    env = dict(os.environ)
    env["STUDENTAI_DATA_DIR"] = data
    port = 8020 + n
    # launch WITHOUT any stdio redirection -> the EXE gets its own console
    # (closest to an end-user double-click)
    log = open(os.path.join(data, "trial_out.log"), "w")
    err = open(os.path.join(data, "trial_err.log"), "w")
    proc = subprocess.Popen(
        [EXE, "--no-browser", "--port", str(port)],
        env=env, cwd=data, stdout=log, stderr=err,
    )
    pid = proc.pid
    start = time.monotonic()
    exited_at = None
    code = None
    while time.monotonic() - start < seconds:
        rc = proc.poll()
        if rc is not None:
            exited_at = round(time.monotonic() - start, 2)
            code = rc
            break
        time.sleep(0.2)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        code = proc.returncode
    log.close()
    err.close()
    result = {
        "trial": n,
        "pid": pid,
        "survived": exited_at is None,
        "exited_at_s": exited_at,
        "exit_code": code,
        "stderr_tail": open(os.path.join(data, "trial_err.log"), encoding="utf-8", errors="replace").read()[-400:],
    }
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    all_results = [trial(i, seconds) for i in range(trials)]
    died = [r for r in all_results if not r["survived"]]
    print(f"\nSUMMARY: {len(died)}/{trials} self-shutdown within {seconds}s")
    for r in died:
        print(f"  trial {r['trial']}: exit after {r['exited_at_s']}s code={r['exit_code']}")
        print(r["stderr_tail"])