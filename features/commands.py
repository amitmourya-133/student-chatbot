"""Slash-command parsing and the built-in help text."""

from __future__ import annotations

import re

import config

COMMANDS: dict[str, str] = {
    "/help": "Show this help menu",
    "/quiz": "Start a quiz",
    "/planner": "Create a study plan",
    "/history": "Show recent chat history",
    "/mode": "Show current mode",
    "/clear": "Clear the screen",
    "/exit": "Exit the chatbot",
    "/quit": "Exit the chatbot",
}

_COMMAND_RE = re.compile(r"^\s*(?P<name>/[a-z]+)(?P<args>.*)$", re.IGNORECASE)


def help_text() -> str:
    width = max(len(name) for name in COMMANDS) + 2
    lines = ["Available Commands", ""]
    for name, description in COMMANDS.items():
        lines.append(f"{name.ljust(width)}{description}")
    lines.append("")
    lines.append("You can also ask study questions directly, e.g.")
    lines.append("  python kya hai")
    lines.append("  explain photosynthesis")
    return "\n".join(lines)


def parse_command(text: str) -> tuple[str | None, str]:
    """Split input into ``(command, args)``.

    Returns ``(None, original_text)`` when the input is not a slash command.
    Command names are returned *without* an alias, lower-cased, and may be
    unknown - callers decide what to do with an unrecognized command.
    """
    if not isinstance(text, str):
        return None, ""
    stripped = text.strip()
    if not stripped:
        return None, ""
    if not stripped.startswith("/"):
        return None, stripped

    match = _COMMAND_RE.match(stripped)
    if not match:
        return None, stripped

    name = match.group("name").lower()
    args = match.group("args").strip()
    return name, args


def is_command(text: str) -> bool:
    name, _ = parse_command(text)
    return name is not None


def is_exit(text: str) -> bool:
    """True for ``/exit``, ``/quit`` and plain ``quit``/``bye`` style input."""
    if not isinstance(text, str):
        return False
    stripped = text.strip().lower().rstrip("!. ")
    if not stripped:
        return False
    name, _ = parse_command(stripped)
    if name in ("/exit", "/quit"):
        return True
    return stripped in config.EXIT_WORDS


def unknown_command_message(text: str) -> str:
    token = text.strip().split()[0] if text.strip() else text
    return (
        f"I don't recognize the command `{token}`.\n"
        "Type /help to see the available commands."
    )
