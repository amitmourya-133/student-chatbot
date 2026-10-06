"""Offline study planner.

``/planner`` asks for subjects, hours per day and number of days, then
generates a day-wise timetable with slots, breaks and revision blocks.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

DEFAULT_START_HOUR = 9
SLOT_MINUTES = 60
SHORT_BREAK_MINUTES = 15
LONG_BREAK_MINUTES = 30
MIN_SLOT_MINUTES = 30

_STOPWORDS = {
    "and",
    "or",
    "plus",
    "with",
    "my",
    "the",
    "all",
    "subjects",
    "subject",
    "ka",
    "ke",
    "aur",
    "sath",
}


def parse_subjects(text: str) -> list[str]:
    """Split free-form input like ``python, maths and science`` into subjects."""
    if not text:
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"\band\b|\baur\b|/", ",", cleaned, flags=re.IGNORECASE)
    parts = re.split(r"[,;]+", cleaned)
    subjects: list[str] = []
    for part in parts:
        name = part.strip().strip("?.!")
        name = re.sub(r"\s+", " ", name)
        if not name:
            continue
        if name.lower() in _STOPWORDS:
            continue
        if len(name) > 40:
            continue
        if name not in subjects:
            subjects.append(name)
    return subjects


def parse_number(text: str) -> float | None:
    """Extract a positive number (``2``, ``2.5``, ``two``) from input."""
    if text is None:
        return None
    raw = text.strip().lower()
    if not raw:
        return None

    words = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "half": 0.5,
    }
    if raw in words:
        return float(words[raw])

    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        value = float(match.group(0))
    except ValueError:
        return None
    return value if value > 0 else None


def _format_time(moment: datetime) -> str:
    return moment.strftime("%H:%M")


def _build_day(day_number: int, subjects: list[str], total_minutes: int) -> list[str]:
    """Return display lines for a single day."""
    lines = [f"DAY {day_number}"]
    current = datetime(2000, 1, 1, DEFAULT_START_HOUR, 0)
    minutes_left = total_minutes
    cursor = 0
    slot_count = 0

    while minutes_left >= MIN_SLOT_MINUTES:
        is_revision = slot_count > 0 and (slot_count + 1) % 3 == 0
        duration = min(SLOT_MINUTES, minutes_left)
        if is_revision and duration > MIN_SLOT_MINUTES:
            duration = MIN_SLOT_MINUTES

        end = current + timedelta(minutes=duration)
        topic = "Revision & Practice" if is_revision else subjects[cursor % len(subjects)]
        lines.append(f"{_format_time(current)} - {_format_time(end)}  {topic}")

        minutes_left -= duration
        current = end
        slot_count += 1

        if minutes_left >= MIN_SLOT_MINUTES:
            break_minutes = (
                LONG_BREAK_MINUTES if slot_count % 4 == 0 else SHORT_BREAK_MINUTES
            )
            if minutes_left < break_minutes:
                break
            current += timedelta(minutes=break_minutes)
            minutes_left -= break_minutes

        if not is_revision:
            cursor += 1

    if minutes_left > 0:
        end = current + timedelta(minutes=minutes_left)
        lines.append(
            f"{_format_time(current)} - {_format_time(end)}  "
            "Light revision / flashcards"
        )
    return lines


def generate_plan(
    subjects: list[str],
    hours_per_day: float,
    days: int,
) -> str:
    """Build a plain-text day-wise study plan."""
    if not subjects:
        raise ValueError("At least one subject is required")
    if hours_per_day <= 0:
        raise ValueError("Hours per day must be positive")
    if days <= 0:
        raise ValueError("Number of days must be positive")

    total_minutes = int(round(hours_per_day * 60))
    if total_minutes < MIN_SLOT_MINUTES:
        total_minutes = MIN_SLOT_MINUTES

    header = [
        "📚 YOUR STUDY PLAN",
        f"Subjects : {', '.join(subjects)}",
        f"Hours/day: {hours_per_day:g}   Days: {days}",
        "=" * 42,
    ]

    body: list[str] = []
    for day in range(1, int(days) + 1):
        body.extend(_build_day(day, subjects, total_minutes))
        body.append("")

    tips = [
        "-" * 42,
        "TIPS",
        "• Take a 5-10 minute break every hour and stretch.",
        "• Do the hardest subject when your energy is highest.",
        "• Revise the previous day's topics before starting today.",
        "• Sleep 7-8 hours - memory consolidates during sleep.",
        "• End each day by writing what you learnt in 3 bullet points.",
    ]

    return "\n".join(header + body + tips)


def planner_intro() -> str:
    return (
        "📚 Study Planner\n\n"
        "Answer 3 quick questions and I will build your schedule.\n\n"
        "1) What subjects do you want to study?\n"
        "   Example: python, maths and science"
    )


def ask_hours() -> str:
    return "2) How many hours can you study each day?\n   Example: 3 or 2.5"


def ask_days() -> str:
    return "3) For how many days?\n   Example: 7"
