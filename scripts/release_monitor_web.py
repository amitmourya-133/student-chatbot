"""Monitor the packaged web EXE to detect self-shutdown.

Launches the EXE, polls /api/health continuously, and records the moment the
process dies on its own (no external termination). Use to reproduce the
intermittent self-shutdown seen during UI verification.

Args: <exe_path> <port> <data_dir> <mintor_seconds>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def main() -> int:
    exe, port, data_dir, monitor_s = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4])
    env = dict(os.environ)
    env["STUDENTAI_DATA_DIR"] = data_dir
    out = open(os.path.join(data_dir, "mon_out.log"), "w")
    err = open(os.path.join(data_dir, "mon_err.log"), "w")
    proc = subprocess.Popen([exe, "--no-browser", "--port", str(port)], env=env, stdout=out, stderr=err, cwd=data_dir)
    base = f"http://127.0.0.1:{port}"
    print(f"launched pid={proc.pid}")

    # request load: a burst of chat requests every few seconds (like a UI session)
    start = time.monotonic()
    last_status = "starting"
    exits_at = None
    while time.monotonic() - start < monitor_s:
        if proc.poll() is not None:
            exits_at = time.monotonic() - start
            last_status = f"PROCESS EXITED after {exits_at:.1f}s (exit={proc.returncode})"
            break
        # health check
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=2) as r:
                health = "up" if r.status == 200 else f"status={r.status}"
        except Exception as e:
            health = f"down({type(e).__name__})"
        # burst of chat
        burst = 0
        if health == "up":
            try:
                body = json.dumps({"message": "what is database"}).encode()
                req = urllib.request.Request(base + "/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=10) as r:
                    burst = r.status
            except Exception as e:
                burst = -1
        line = f"t={time.monotonic()-start:6.2f}s health={health} chat={burst} alive={proc.poll() is None}"
        if line != last_status:
            print(line)
            last_status = line
        time.sleep(0.8)
    print(f"final: alive={proc.poll() is None}" if proc.poll() is None else f"final: exited exit={proc.returncode}")
    out.close()
    err.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())