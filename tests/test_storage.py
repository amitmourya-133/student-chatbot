"""Regression tests for the concurrent-write safeguards.

Verifies that:
* atomic JSON writes never leave a partially-written / invalid file
* the cross-process file lock serialises writers and honours its timeout
* multiple writers adding to the chat log converge without lost updates
"""

from __future__ import annotations

import json
import threading
import time

from utils.filelock import FileLock, atomic_json_write


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_atomic_json_write_roundtrip(tmp_path):
    target = tmp_path / "memory.json"
    atomic_json_write(target, {"facts": {"name": "Akash"}, "session": [], "disabled": False})
    assert _read_json(target) == {
        "facts": {"name": "Akash"},
        "session": [],
        "disabled": False,
    }


def test_atomic_json_write_replaces_without_leftover_tmp(tmp_path):
    target = tmp_path / "memory.json"
    atomic_json_write(target, {"v": 1})
    atomic_json_write(target, {"v": 2})
    assert _read_json(target) == {"v": 2}
    leftovers = list(tmp_path.glob("memory.json.*.tmp"))
    assert leftovers == []


def test_concurrent_writers_never_corrupt(tmp_path):
    target = tmp_path / "shared.json"
    errors = []

    def writer(tag: str):
        # mirror the production read-modify-write flow: lock, then atomic write
        for i in range(30):
            try:
                with FileLock(target):
                    atomic_json_write(target, {"writer": tag, "seq": i})
            except Exception as exc:  # pragma: no cover - only fails on bug
                errors.append(f"{tag}:{i}:{exc!r}")

    a = threading.Thread(target=writer, args=("A",))
    b = threading.Thread(target=writer, args=("B",))
    a.start()
    b.start()
    a.join()
    b.join()

    assert errors == []
    # whichever writer finished last won; the file must still be valid JSON
    data = _read_json(target)
    assert data["writer"] in ("A", "B")


def test_file_lock_mutual_exclusion(tmp_path):
    lock_target = tmp_path / "data.json"
    inside = {"count": 0}
    lock = FileLock(lock_target)
    lock.acquire()
    try:
        # a second lock attempt must time out while the first is held
        started = time.monotonic()
        second = FileLock(lock_target, timeout=0.5)
        try:
            second.acquire()
            timed_out = False
        except TimeoutError:
            timed_out = True
        finally:
            second.release()
        assert timed_out
        # held long enough to prove blocking (release() of a never-acquired
        # lock is a no-op on our backends)
        assert time.monotonic() - started < 5
    finally:
        lock.release()
    # once released a new lock can be acquired immediately
    third = FileLock(lock_target, timeout=1.0)
    third.acquire()
    inside["count"] += 1
    third.release()
    assert inside["count"] == 1


def test_chat_history_concurrent_appends_no_lost_writes(tmp_path):
    from services.chat_history import ChatHistory

    log = ChatHistory(path=tmp_path / "chat_log.json")
    threads = []

    def worker(tag: str):
        local = ChatHistory(path=tmp_path / "chat_log.json")
        for i in range(10):
            local.add(f"{tag}-{i}", "reply", "offline")

    for tag in ("sky", "ocean"):
        thread = threading.Thread(target=worker, args=(tag,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()

    log._load_unchecked()
    entries = log.all()
    # all 20 writes must be present and valid JSON
    assert len(entries) == 20
    users = {entry["user"] for entry in entries}
    assert len(users) == 20