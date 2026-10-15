"""Method B comparator: subprocess.Popen lifecycle trials."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

exe = sys.argv[1]
trials = int(sys.argv[2])
hold_seconds = int(sys.argv[3])

results = []
for i in range(trials):
    data = os.path.join(os.environ["TEMP"], "LaunchB_%d" % i)
    os.makedirs(data, exist_ok=True)
    port = 8040 + i + 100
    env = dict(os.environ)
    env["STUDENTAI_DATA_DIR"] = data
    out = open(os.path.join(data, "stdout.log"), "w")
    err = open(os.path.join(data, "stderr.log"), "w")
    proc = subprocess.Popen(
        [exe, "--no-browser", "--port", str(port)],
        env=env, cwd=data, stdout=out, stderr=err,
    )
    start = time.monotonic()
    exited = None
    health = False
    while time.monotonic() - start < hold_seconds:
        if proc.poll() is not None:
            exited = round(time.monotonic() - start, 2)
            break
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:%d/api/health" % port, timeout=1
            ) as r:
                if r.status == 200:
                    health = True
        except Exception:
            pass
        time.sleep(0.5)
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except Exception:
            proc.kill()
    out.close()
    err.close()
    tail = open(
        os.path.join(data, "stderr.log"), encoding="utf-8", errors="replace"
    ).read()[-400:]
    results.append({
        "method": "subprocess.Popen",
        "trial": i,
        "pid": proc.pid,
        "stayed_alive": exited is None,
        "exited_early_s": exited,
        "exit_code": proc.returncode,
        "health_seen": health,
        "stderr_tail": tail,
    })
    print("B trial %d pid=%d alive=%s health=%s exit=%s"
          % (i, proc.pid, exited is None, health, proc.returncode))

print("RESULT_JSON=" + json.dumps(results))