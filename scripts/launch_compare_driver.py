"""Decisive launch-method comparison using psutil-level process monitoring.

Method A: real PowerShell Start-Process (PID captured from -PassThru).
Method B: python subprocess.Popen.
Both launch the SAME EXE, SAME flags, SAME data dir pattern and are monitored
IDENTICALLY for N seconds: psutil status, CPU, health endpoint, and the exact
time of death + exit code. Totals the full child tree (Start-Process may return
a PID whose children carry the actual server) and kills whole trees.

Usage: python scripts/launch_compare_driver.py <trials> <hold_seconds> <json_out>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

import psutil

EXE = r"D:\chatbot ai\dist\StudentAIAssistantWeb\StudentAIAssistantWeb.exe"


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/api/health" % port, timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def launch_via_startprocess(port: int, data_dir: str, env: dict) -> dict:
    pid_file = os.path.join(data_dir, "launcher.pid")
    ps = (
        "$p = Start-Process -FilePath %s -ArgumentList @('--no-browser','--port','%d') "
        "-WorkingDirectory '%s' -RedirectStandardOutput '%s\\stdout.log' "
        "-RedirectStandardError '%s\\stderr.log' -PassThru; "
        "$p.Id | Out-File -Encoding ascii '%s'"
        % ('"%s"' % (EXE.replace("'", "''")), port, data_dir.replace("'", "''"),
           data_dir, data_dir, pid_file.replace("'", "''"))
    )
    child_env = dict(env)
    child_env["STUDENTAI_DATA_DIR"] = data_dir
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         timeout=60, env=child_env)
    if out.returncode != 0:
        raise RuntimeError("Start-Process failed rc=%d" % out.returncode)
    for _ in range(50):
        if os.path.exists(pid_file):
            break
        time.sleep(0.1)
    if not os.path.exists(pid_file):
        raise RuntimeError("Start-Process wrote no PID file")
    root = int(open(pid_file, encoding="ascii").read().strip())
    tree = {p.pid: p for p in psutil.Process(root).children(recursive=True)}
    tree[root] = psutil.Process(root)
    desc = []
    for p in list(tree.values()):
        try:
            desc.append({"pid": p.pid, "name": p.name(), "ppid": p.ppid(),
                         "cmd": " ".join(p.cmdline()[-4:])})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            desc.append({"pid": p.pid, "name": "gone", "ppid": None, "cmd": ""})
    return {"root": root, "tree": desc, "children": list(tree.keys())}


def kill_tree(pids: list[int]) -> None:
    for pid in reversed(pids):
        try:
            pr = psutil.Process(pid)
            pr.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        except Exception:
            pass
    time.sleep(0.3)


def monitor(pids: list[int], port: int, tag: str, hold: int, data_dir: str) -> dict:
    any_dead_early = False
    death_by_root = None
    health_seen = False
    start = time.monotonic()
    last_mark = ""
    while time.monotonic() - start < hold:
        alive_root = False
        for pid in pids:
            try:
                p = psutil.Process(pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    alive_root = True
                    break
            except psutil.NoSuchProcess:
                continue
        if not alive_root:
            any_dead_early = True
            death_by_root = round(time.monotonic() - start, 2)
            break
        h = health(port)
        health_seen = health_seen or h
        try:
            cpu = psutil.Process(pids[0]).cpu_percent(interval=None)
        except Exception:
            cpu = -1
        mark = f"alive cpu={cpu:.0f} health={h}"
        if mark != last_mark:
            print(f"  [{tag}] tree={pids} t={time.monotonic()-start:.1f}s {mark}", flush=True)
            last_mark = mark
        time.sleep(0.5)

    exit_codes = []
    alive_now = []
    last_ref = None
    for pid in pids:
        try:
            p = psutil.Process(pid)
            p._name = None
            if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                alive_now.append(pid)
            else:
                exit_codes.append({"pid": pid, "code": "not-running"})
        except psutil.NoSuchProcess:
            exit_codes.append({"pid": pid, "code": "gone"})
    if alive_now:
        kill_tree(alive_now)

    time.sleep(2)
    stderr = ""
    err_file = os.path.join(data_dir, "stderr.log")
    if os.path.exists(err_file):
        stderr = open(err_file, encoding="utf-8", errors="replace").read()
    return {
        "method": tag,
        "pids": pids,
        "died_within_hold": any_dead_early,
        "seconds_alive": death_by_root,
        "exit_codes": exit_codes,
        "health_seen": health_seen,
        "graceful_shutdown_log": bool(stderr and "Shutting down" in stderr),
        "stderr_tail": stderr[-300:],
    }


def main() -> int:
    trials = int(sys.argv[1])
    hold = int(sys.argv[2])
    out_path = sys.argv[3]
    results = []
    base_env = dict(os.environ)
    for i in range(trials):
        port_a = 8220 + 2 * i
        data_a = os.path.join(os.environ["TEMP"], "CompA_%d" % i)
        os.makedirs(data_a, exist_ok=True)
        print(f"START trial {i} method=Start-Process port={port_a}", flush=True)
        info = launch_via_startprocess(port_a, data_a, base_env)
        print(f"  Start-Process tree={info['tree']}", flush=True)
        r = monitor(info["children"], port_a, "A.Start-Process", hold, data_a)
        results.append(r)
        print("  ", json.dumps(r), flush=True)
        time.sleep(1)

        port_b = 8220 + 2 * i + 1
        data_b = os.path.join(os.environ["TEMP"], "CompB_%d" % i)
        os.makedirs(data_b, exist_ok=True)
        env_b = dict(base_env)
        env_b["STUDENTAI_DATA_DIR"] = data_b
        out = open(os.path.join(data_b, "stdout.log"), "w")
        err = open(os.path.join(data_b, "stderr.log"), "w")
        print(f"START trial {i} method=Popen port={port_b}", flush=True)
        proc = subprocess.Popen(
            [EXE, "--no-browser", "--port", str(port_b)],
            env=env_b, cwd=data_b, stdout=out, stderr=err,
        )
        r = monitor([proc.pid], port_b, "B.Popen", hold, data_b)
        results.append(r)
        print("  ", json.dumps(r), flush=True)
        time.sleep(1)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    print("JSON ->", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())