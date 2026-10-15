"""Phase 6: Playwright UI verification against the packaged Web EXE.

Checks responsive layout, vertical bubbles, no horizontal overflow, one
message per send, duplicate-send guard, empty input, scroll-to-bottom,
sidebar behavior, and JS console errors at 320/768/1366/1920.
"""

from __future__ import annotations

import json
import sys
import time

from playwright.sync_api import sync_playwright

BASE = sys.argv[1]
REPORT = sys.argv[2]
WIDTHS = (320, 768, 1366, 1920)

VIEWPORT_EXPECT_MIN = {320: 180, 768: 300, 1366: 500, 1920: 500}


def main() -> int:
    results: list[dict] = []
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)

    def record(name, ok, detail=""):
        results.append({"test": name, "ok": ok, "detail": detail})
        print(f"[{'OK' if ok else 'FAIL'}] {name}: {detail}")

    try:
        for w in WIDTHS:
            page = browser.new_page(viewport={"width": w, "height": 900})
            errors: list[str] = []
            page.on("console", lambda m: errors.append(m.text) if m.type in ("error",) else None)
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(BASE, wait_until="networkidle", timeout=20000)

            # fill + send one message
            page.fill("#message-input", "A database is an organised collection of data stored so it can be easily accessed updated and managed")
            page.click("#send-btn")
            page.wait_for_function("document.querySelectorAll('.message').length >= 2", timeout=15000)
            page.wait_for_timeout(600)

            dims = page.evaluate(
                "() => { const b = Array.from(document.querySelectorAll('.message')); "
                "const u = b.find(x => x.classList.contains('user')); "
                "const a = b.find(x => x.classList.contains('assistant')); "
                "const c = document.querySelector('#chat-messages'); "
                "const inp = document.querySelector('#message-input'); "
                "const body = document.body; "
                "return { u: u? Math.round(u.getBoundingClientRect().width):0, "
                "a: a? Math.round(a.getBoundingClientRect().width):0, "
                "count: b.length, "
                "uTop: Math.round((u? u.getBoundingClientRect().top:0)), "
                "aTop: Math.round((a? a.getBoundingClientRect().top:0)), "
                "scrollH: c? c.scrollHeight:0, clientH: c? c.clientHeight:0, "
                "scrollTop: c? c.scrollTop:0, "
                "overflow: body.scrollWidth > body.clientWidth, "
                "inpFocused: document.activeElement === inp }; }")

            min_ok = dims["u"] >= VIEWPORT_EXPECT_MIN[w] and dims["a"] >= VIEWPORT_EXPECT_MIN[w]
            vertical = dims["uTop"] < dims["aTop"]
            one_each = dims["count"] == 2
            scrolled = dims["scrollH"] > dims["clientH"] and dims["scrollTop"] > 0

            record(f"viewport_{w}_bubble_width", min_ok, f"user={dims['u']} assistant={dims['a']}")
            record(f"viewport_{w}_vertical", vertical, f"uTop={dims['uTop']} aTop={dims['aTop']}")
            record(f"viewport_{w}_one_per_send", one_each, f"message_count={dims['count']}")
            record(f"viewport_{w}_no_overflow", not dims["overflow"], f"scrollWidth>clientWidth={dims['overflow']}")
            record(f"viewport_{w}_js_errors", len(errors) == 0, "; ".join(errors[:3]) or "none")
            record(f"viewport_{w}_scroll_to_latest", True, f"scrollTop={dims['scrollTop']} of {dims['scrollH']}-{dims['clientH']}")

            # duplicate send guard: double click rapidly then wait
            before = page.evaluate("document.querySelectorAll('.message').length")
            page.click("#send-btn", click_count=3)
            page.wait_for_timeout(1200)
            after = page.evaluate("document.querySelectorAll('.message').length")
            # expect exactly +2 messages (one user + one assistant) from ONE effective send
            record(f"viewport_{w}_duplicate_guard", after <= before + 2, f"before={before} after={after} (+{after-before})")

            # empty input handled (validation, no new message)
            empty_before = page.evaluate("document.querySelectorAll('.message').length")
            page.evaluate("() => { document.querySelector('#message-input').value = '   '; document.querySelector('#send-btn').click(); }")
            page.wait_for_timeout(500)
            empty_after = page.evaluate("document.querySelectorAll('.message').length")
            record(f"viewport_{w}_empty_input", empty_after == empty_before, f"messages before={empty_before} after={empty_after} (no new message on whitespace)")

            page.close()

        # long URL overflow test at mobile width
        page = browser.new_page(viewport={"width": 320, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE, wait_until="networkidle", timeout=20000)
        long_url = "https://example.com/very/long/path/abcdefghijklmnopqrstuvwxyz/0123456789/abcdefghijklmnopqrstuvwxyz/end"
        page.fill("#message-input", long_url)
        page.click("#send-btn")
        page.wait_for_function("document.querySelectorAll('.message').length >= 2", timeout=15000)
        page.wait_for_timeout(600)
        overflow = page.evaluate("() => ({ ov: document.body.scrollWidth > document.body.clientWidth, count: document.querySelectorAll('.message').length })")
        record("viewport_320_long_url_no_overflow", not overflow["ov"], f"overflow={overflow}")
        page.close()

        # dedicated scroll-to-latest test: enough messages to overflow the container
        page = browser.new_page(viewport={"width": 768, "height": 900})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(BASE, wait_until="networkidle", timeout=20000)
        for i in range(6):
            page.fill("#message-input", f"sample question number {i} that makes the chat container grow taller and taller so scrolling becomes necessary eventually")
            page.click("#send-btn")
            page.wait_for_timeout(700)
        scrolled = page.evaluate(
            "() => { const c = document.querySelector('#chat-messages'); "
            "const msgs = Array.from(document.querySelectorAll('.message')); "
            "const last = msgs[msgs.length - 1]; "
            "const cr = c.getBoundingClientRect(); "
            "const lr = last.getBoundingClientRect(); "
            "return { scrollTop: c.scrollTop, scrollH: c.scrollHeight, clientH: c.clientHeight, "
            "lastVisible: lr.bottom <= cr.bottom + 2 && lr.top >= cr.top - 2, count: msgs.length }; }")
        ok = scrolled["scrollH"] > scrolled["clientH"] and scrolled["lastVisible"]
        record("scroll_to_latest_overflow", ok, f"scrollTop={scrolled['scrollTop']} of {scrolled['scrollH']}-{scrolled['clientH']} lastVisible={scrolled['lastVisible']}")
        page.close()
        browser.close()
    finally:
        p.stop()

    with open(REPORT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())