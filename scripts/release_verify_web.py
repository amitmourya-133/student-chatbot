"""Phase 4/5: start the packaged Web EXE, verify startup, homepage, health, and chat.

Args: <exe_path> <port> <data_dir> <report_file>
Sends the 10 chat cases from the master prompt and prints sanitized results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "127.0.0.1"


def _url(port: int, path: str) -> str:
    return f"http://{HOST}:{port}{path}"


def http(method: str, url: str, body: bytes | None = None, headers: dict | None = None, timeout: int = 30):
    req = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, raw
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except urllib.error.URLError as e:
        return -1, str(e).encode()


REQUIRED = [
    ("what is database", "database"),
    ("what is newton ring", "ring"),
    ("what is hindi meaning of phone", "फ़ोन"),
    ("what is sex", "sex"),
]


def main() -> int:
    exe, port, data_dir, report_file = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
    env = dict(os.environ)
    env["STUDENTAI_DATA_DIR"] = data_dir
    out_path = os.path.join(data_dir, "web_stdout.log")
    err_path = os.path.join(data_dir, "web_stderr.log")

    proc = subprocess.Popen(
        [exe, "--no-browser", "--port", str(port)],
        env=env, stdout=open(out_path, "w"), stderr=open(err_path, "w"), cwd=data_dir,
    )

    results: list[dict] = []

    def rec(name, status, ok, detail=""):
        results.append({"test": name, "status": status, "ok": ok, "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {status} {detail}")

    try:
        # wait for readiness
        ready = False
        deadline = time.monotonic() + 40
        while time.monotonic() < deadline:
            status, body = http("GET", _url(port, "/api/health"), timeout=2)
            if status == 200:
                ready = True
                break
            time.sleep(0.5)
        rec("startup_ready", ready, ready, f"pid={proc.pid}")

        status, body = http("GET", _url(port, "/"))
        ok = status == 200 and b"Student AI Assistant" in body
        rec("homepage", status, ok, f"{len(body)} bytes")

        status, body = http("GET", _url(port, "/api/health"))
        sanitized = body[:200].decode(errors="replace")
        rec("health", status, status == 200 and b'"status":"ok"' in body, sanitized)

        # 4 known questions
        for q, marker in REQUIRED:
            body = json.dumps({"message": q}).encode()
            status, resp = http("POST", _url(port, "/api/chat"), body, {"Content-Type": "application/json"})
            ok = False
            detail = ""
            if status == 200:
                data = json.loads(resp.decode())
                text = data.get("response", "")
                marker_l = marker.lower()
                ok = bool(text.strip()) and (marker_l in text.lower() or marker in text) and "first law" not in text.lower()
                detail = f"mode={data.get('mode')} len={len(text)}"
            rec(f"kb_question:{q}", status, ok, detail)

        # 4. Newton's rings must NOT be first law
        body = json.dumps({"message": "what is newton ring"}).encode()
        status, resp = http("POST", _url(port, "/api/chat"), body, {"Content-Type": "application/json"})
        if status == 200:
            text = json.loads(resp.decode()).get("response", "")
            ok = "interference" in text.lower()
            rec("newton_not_first_law", status, ok, "interference present" if ok else text[:80])
        else:
            rec("newton_not_first_law", status, False)

        # unknown question
        body = json.dumps({"message": "explain quantum computing"}).encode()
        status, resp = http("POST", _url(port, "/api/chat"), body, {"Content-Type": "application/json"})
        ok = status == 200 and json.loads(resp.decode()).get("response", "").strip()
        rec("unknown_fallback", status, ok, resp[:120].decode(errors="replace"))

        # empty string
        status, resp = http("POST", _url(port, "/api/chat"), json.dumps({"message": ""}).encode(), {"Content-Type": "application/json"})
        rec("empty_message", status, status in (400, 422), resp[:120].decode(errors="replace"))

        # whitespace-only
        status, resp = http("POST", _url(port, "/api/chat"), json.dumps({"message": "   "}).encode(), {"Content-Type": "application/json"})
        rec("whitespace_message", status, status in (400, 422), resp[:120].decode(errors="replace"))

        # moderately long valid message
        long_msg = "What is the difference between a database and a data warehouse in data management " * 3
        status, resp = http("POST", _url(port, "/api/chat"), json.dumps({"message": long_msg}).encode(), {"Content-Type": "application/json"})
        ok = status == 200 and json.loads(resp.decode()).get("response", "").strip()
        rec("long_message", status, ok, f"len={len(resp)}")

        # repeated identical
        statuses = []
        for _ in range(2):
            status, resp = http("POST", _url(port, "/api/chat"), json.dumps({"message": "what is database"}).encode(), {"Content-Type": "application/json"})
            statuses.append(status)
        rec("repeated_requests", 200, statuses == [200, 200], str(statuses))

        # malformed JSON
        status, resp = http("POST", _url(port, "/api/chat"), b"{not json", {"Content-Type": "application/json"})
        rec("malformed_json", status, status in (400, 422), resp[:120].decode(errors="replace"))

        # missing required field
        status, resp = http("POST", _url(port, "/api/chat"), json.dumps({"msg": "hello"}).encode(), {"Content-Type": "application/json"})
        rec("missing_field", status, status in (400, 422), resp[:120].decode(errors="replace"))

        # error must not leak stack/keys
        status, resp = http("POST", _url(port, "/api/chat"), json.dumps({"message": "what is database"}).encode(), {"Content-Type": "application/json"})
        txt = resp.decode(errors="replace")
        rec("no_secret_leak", status, "api_key" not in txt.lower() and "traceback" not in txt.lower() and "generativelanguage" not in txt, ok)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    rec("clean_exit", proc.returncode, True if proc.returncode in (0, 1) else False, f"exit={proc.returncode}")

    with open(report_file, "w", encoding="utf-8") as fh:
        json.dump({"pid": proc.pid, "results": results}, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())