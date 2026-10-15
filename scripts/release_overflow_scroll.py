"""Dedicated overflow-scroll regression test for the packaged Web EXE.

Drives the REAL packaged executable (subprocess.Popen launch), then verifies
chat-container scrolling under heavy message volume:

  1. 20-30 unique messages are sent, each send waiting for its own
     user + assistant response pair to complete before the next one.
  2. The real scroll container (`.messages` / `#chat-messages`) overflows:
     scrollHeight > clientHeight.
  3. The newest message is visible: its bounding rect intersects the
     container's visible rect within a small pixel tolerance.
  4. Auto-scroll lands near the bottom when the user is already at the
     bottom (scrollTop + clientHeight >= scrollHeight - tolerance).
  5. The newest message is not clipped behind the composer/input area.
  6. MANUAL SCROLL-UP PRESERVATION: after scrolling up to read history,
     an incoming response must NOT yank the viewport back to the bottom.

Failures print precise diagnostics (rects, scroll metrics, message counts).

Usage: python scripts/release_overflow_scroll.py --exe <web.exe> --port N
                                            [--base-url http://127.0.0.1:N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

TOLERANCE = 2.0  # px
VIEWPORT = {"width": 768, "height": 900}
MESSAGES = 24
TIME_TEXT = (
    "Describe how interference fringes form in Newton's rings experiment and "
    "what causes the pattern of alternating bright and dark rings to appear"
)


def _wait_ready(base: str) -> None:
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/api/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("Server did not become ready")


def _send_and_wait(page, text: str, expected_total: int) -> None:
    page.fill("#message-input", text)
    page.click("#send-btn")
    # wait for BOTH the user and assistant messages for this send
    page.wait_for_function(
        "document.querySelectorAll('.message').length >= %d" % expected_total,
        timeout=20000,
    )
    # wait until the send button is re-enabled (request fully finished) and a
    # brief settle so the final assistant bubble is rendered & laid out
    page.wait_for_function(
        "!document.querySelector('#send-btn').disabled && "
        "!document.querySelector('#send-btn').classList.contains('disabled')",
        timeout=20000,
    )
    page.wait_for_timeout(400)


def _metrics(page) -> dict:
    return page.evaluate(
        """() => {
          const c = document.querySelector('#chat-messages');
          const list = document.querySelector('#chat-messages-list');
          const msgs = Array.from(document.querySelectorAll('.message'));
          const last = msgs[msgs.length - 1];
          const pen = msgs[msgs.length - 2] || last;
          const cr = c.getBoundingClientRect();
          const lr = last.getBoundingClientRect();
          const composer = document.querySelector('.composer');
          const comr = composer ? composer.getBoundingClientRect() : null;
          const input = document.querySelector('#message-input');
          const inr = input ? input.getBoundingClientRect() : null;
          return {
            count: msgs.length,
            scrollHeight: c.scrollHeight,
            clientHeight: c.clientHeight,
            scrollTop: c.scrollTop,
            messageListVisible: !!list && list.getBoundingClientRect().height > 0,
            lastTop: lr.top, lastBottom: lr.bottom,
            lastInView: lr.bottom <= cr.bottom + 2 && lr.top >= cr.top - 2,
            lastWithinContainer: lr.top >= cr.top - 2 - Math.max(cr.top, lr.top) * 0 + 2 && lr.bottom <= cr.bottom + 2,
            containerTop: cr.top, containerBottom: cr.bottom,
            notBehindComposer: !comr || !inr || lr.bottom <= inr.top + 2 || lr.bottom <= comr.top + 2,
            penTop: pen.getBoundingClientRect().top,
            userScrolledUpAtLeastOnce: c.scrollTop < c.scrollHeight - c.clientHeight - 10,
          };
        }"""
    )


def run_overflow_check(page, base: str) -> list[dict]:
    results: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append({"test": name, "ok": ok, "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")

    page.goto(base, wait_until="networkidle", timeout=25000)

    # --- 1. send N unique messages, each fully awaited --------------------
    for i in range(MESSAGES):
        text = TIME_TEXT if i % 3 == 0 else (
            "database question number %d exploring normalised tables" % i
            if i % 3 == 1 else "question %d about newton rings experiment" % i
        )
        _send_and_wait(page, text, (i + 1) * 2)
    page.wait_for_timeout(800)

    m = _metrics(page)
    count_ok = m["count"] == MESSAGES * 2
    record("overflow_send_all", count_ok, f"messages={m['count']} (want {MESSAGES*2})")

    # --- 2. container overflows -------------------------------------------
    overflow_ok = m["scrollHeight"] > m["clientHeight"]
    record(
        "overflow_container_overflow",
        overflow_ok,
        f"scrollHeight={m['scrollHeight']} > clientHeight={m['clientHeight']}",
    )

    # --- 3. newest message visible within container ------------------------
    record(
        "overflow_newest_visible",
        m["lastInView"],
        f"lastBottom={m['lastBottom']:.1f} vs containerBottom={m['containerBottom']:.1f} "
        f"lastTop={m['lastTop']:.1f} vs containerTop={m['containerTop']:.1f}",
    )

    # --- 4. auto-scroll near bottom ----------------------------------------
    at_bottom = (
        m["scrollTop"] + m["clientHeight"] >= m["scrollHeight"] - TOLERANCE
    )
    record(
        "overflow_near_bottom",
        at_bottom,
        f"scrollTop={m['scrollTop']} + clientHeight={m['clientHeight']} >= scrollHeight={m['scrollHeight']}",
    )

    # --- 5. newest not clipped behind composer/input -----------------------
    record(
        "overflow_no_clip_behind_composer",
        m["notBehindComposer"],
        f"lastBottom={m['lastBottom']:.1f} inputTop~{m['containerBottom']:.1f}",
    )

    # --- 6. manual scroll-up preservation -----------------------------------
    # User scrolls up to read history while a response is still pending; the
    # arriving response must NOT yank the viewport back to the bottom.
    # A short artificial network delay makes the pending window deterministic.
    def delay_chat(route):
        time.sleep(0.8)
        route.continue_()

    page.route("**/api/chat", delay_chat)
    before = page.evaluate("document.querySelectorAll('.message').length")
    page.evaluate("() => { const c = document.querySelector('#chat-messages'); c.scrollTop = c.scrollHeight; }")
    page.fill("#message-input", "history question while a response is pending")
    page.click("#send-btn")
    page.wait_for_function("document.querySelector('#send-btn').disabled", timeout=10000)
    # user bubble rendered, response delayed -> confirmed pending window
    page.wait_for_function(
        "document.querySelectorAll('.message').length == %d" % (before + 1),
        timeout=10000,
    )
    page.evaluate(
        "() => { const c = document.querySelector('#chat-messages'); c.scrollTop = 0; }"
    )
    page.wait_for_timeout(200)
    # now let the pending response render (assistant bubble appended)
    page.wait_for_function(
        "document.querySelectorAll('.message').length == %d" % (before + 2),
        timeout=20000,
    )
    page.unroute("**/api/chat")
    page.wait_for_timeout(600)
    m2 = _metrics(page)
    preserved = m2["scrollTop"] < m2["scrollHeight"] - m2["clientHeight"] - 10
    record(
        "overflow_manual_scrollup_preserved",
        preserved,
        f"scrollTop={m2['scrollTop']} maxScroll={m2['scrollHeight'] - m2['clientHeight']} "
        f"(stayed up by {max(0, m2['scrollHeight'] - m2['clientHeight'] - m2['scrollTop']):.0f}px)",
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    data = os.path.join(os.environ["TEMP"], "StudentAIData_Overflow")
    os.makedirs(data, exist_ok=True)
    env = dict(os.environ)
    env["STUDENTAI_DATA_DIR"] = data

    proc = subprocess.Popen(
        [args.exe, "--no-browser", "--port", str(args.port)],
        env=env,
        cwd=data,
        stdout=open(os.path.join(data, "stdout.log"), "w"),
        stderr=open(os.path.join(data, "stderr.log"), "w"),
    )
    print(f"launched pid={proc.pid}", flush=True)
    results: list[dict] = []
    try:
        _wait_ready(base)
        print("server ready", flush=True)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport=VIEWPORT)
            errs: list[str] = []
            page.on("pageerror", lambda e: errs.append(str(e)))
            results = run_overflow_check(page, base)
            browser.close()
        if errs:
            results.append({"test": "overflow_js_errors", "ok": False, "detail": "; ".join(errs[:5])})
            print("[FAIL] overflow_js_errors:", errs[:5])
        else:
            results.append({"test": "overflow_js_errors", "ok": True, "detail": "none"})
            print("[OK] overflow_js_errors: none")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print(f"server final exit={proc.returncode}", flush=True)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
    failed = [r for r in results if not r["ok"]]
    print(f"RESULT: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())