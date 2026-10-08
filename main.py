#!/usr/bin/env python3
"""Student AI Chatbot - entry point.

Run::

    python main.py

The bot detects internet connectivity and answers through Gemini when it is
online, otherwise it falls back to the local TF-IDF + Naive Bayes intent
model and the offline knowledge base.  Nothing here requires the internet.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from features import commands as command_utils
from features import planner as planner_feature
from features import quiz as quiz_feature
from services.pipeline import AssistantPipeline
from utils.logger import setup_logging

logger = logging.getLogger("chatbot")

RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"

USE_COLOR = False

BANNER = """
╔══════════════════════════════════════════════╗
║        🎓  STUDENT AI CHATBOT                ║
║        Your Personal Study Assistant         ║
╚══════════════════════════════════════════════╝
"""

GOODBYE = "👋 Goodbye!\nKeep learning and keep building. 🚀"


def _init_color() -> None:
    """Enable ANSI colours only on real, colour-capable terminals."""
    global USE_COLOR
    if os.environ.get("NO_COLOR"):
        USE_COLOR = False
        return
    try:
        if os.name == "nt":
            os.system("")  # enables VT/ANSI processing on Windows 10+
        USE_COLOR = bool(getattr(sys.stdout, "isatty", lambda: False)())
    except Exception:
        USE_COLOR = False


def _init_io() -> None:
    """Never crash when the console codepage cannot encode a glyph."""
    for stream in (sys.stdout, sys.stderr):
        try:
            encoding = (getattr(stream, "encoding", None) or "").lower()
            if "utf" not in encoding and hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
        except Exception:
            pass


def paint(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{RESET}"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


class StudentChatbot:
    """Wires together the model, knowledge base, services and features."""

    # ------------------------------------------------------------------ #
    # startup
    # ------------------------------------------------------------------ #
    def __init__(self, force_offline: bool = False) -> None:
        self.pipeline = AssistantPipeline(force_offline=force_offline)
        self.running = True
        self._pending_exit = False

    # convenient read-only views into the shared pipeline
    @property
    def force_offline(self) -> bool:
        return self.pipeline.force_offline

    @property
    def online(self) -> bool:
        return self.pipeline.online

    @property
    def online_disabled(self) -> bool:
        return self.pipeline.online_disabled

    @property
    def classifier(self):
        return self.pipeline.classifier

    @property
    def kb(self):
        return self.pipeline.kb

    @property
    def history(self):
        return self.pipeline.history

    @property
    def intent_responses(self) -> dict[str, list[str]]:
        return self.pipeline.intent_responses

    def validate_directories(self) -> None:
        self.pipeline.validate_directories()

    def load_intent_responses(self) -> None:
        self.pipeline.load_intent_responses()

    def startup(self) -> None:
        """Validate everything and prepare the runtime. Never raises."""
        logger.info("Starting %s", config.BOT_NAME)
        logger.info("Config: %s", config.summary())
        status = self.pipeline.startup()

        if not status.intent_model:
            print(
                paint("\n⚠️  Intent model not found and training failed.\n", YELLOW)
                + "Run:\n"
                + paint("    python nlp/train.py", BOLD)
                + "\n"
            )
        if not status.knowledge_base:
            print(
                paint(
                    "\n⚠️  Offline knowledge base could not be loaded.\n", YELLOW
                )
                + f"Check that {config.KNOWLEDGE_BASE_FILE} exists and is valid JSON.\n"
            )

        if not config.has_api_key() and not self.force_offline:
            logger.info("No Gemini API key configured - offline only")

    def refresh_mode(self, initial: bool = False) -> bool:
        """Re-check connectivity and update ``self.online``."""
        previous = self.online
        available = self.pipeline.refresh_mode(initial=initial)
        if not initial and previous != available:
            logger.info("Mode changed -> %s", self.mode_label())
            print()
            print(self.mode_line())
        return available

    # ------------------------------------------------------------------ #
    # display helpers
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> str:
        return "online" if self.online else "offline"

    def mode_label(self) -> str:
        return "ONLINE" if self.online else "OFFLINE"

    def mode_icon(self) -> str:
        return "🌐" if self.online else "📴"

    def mode_line(self) -> str:
        color = GREEN if self.online else YELLOW
        return paint(f"{self.mode_icon()} Mode: {self.mode_label()}", color)

    def show_banner(self) -> None:
        print()
        print(paint(BANNER, CYAN))
        print(self.mode_line())
        if not self.online and not self.force_offline:
            if not config.has_api_key():
                print(
                    paint(
                        "   (no GEMINI_API_KEY configured - see .env.example)",
                        YELLOW,
                    )
                )
            else:
                print(paint("   (internet unavailable)", YELLOW))
        print()
        print("Type " + paint("/help", BOLD) + " for commands.")
        print()

    def get_reply(self, text: str) -> tuple[str, str]:
        """Return ``(reply, mode)`` - never raises."""
        result = self.pipeline.reply(text)
        if result.just_switched:
            print(
                paint(
                    "⚠️ Online AI unavailable. Switching to offline mode.",
                    YELLOW,
                )
            )
            print(self.mode_line())
        return result.text, ("online" if result.online else "offline")

    # ------------------------------------------------------------------ #
    # interactive features
    # ------------------------------------------------------------------ #
    @staticmethod
    def _read(prompt: str) -> Optional[str]:
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return None

    def run_quiz(self, subject_arg: str = "") -> None:
        session = None
        if subject_arg:
            session, message = quiz_feature.start_quiz(subject_arg)
            print()
            print(paint("🧠 QUIZ MODE", MAGENTA))
            print(message)
        else:
            print()
            print(paint("🧠 QUIZ MODE", MAGENTA))
            subjects = ", ".join(quiz_feature.list_subjects())
            print("Choose a subject:\n   " + subjects)

        while session is None:
            answer = self._read("Subject: ")
            if answer is None or answer.strip().lower() in ("/cancel", "/skip", ""):
                if answer is None:
                    self._pending_exit = True
                print("Quiz cancelled. Type /quiz to start again.")
                return
            if answer.strip().lower() in ("/exit", "/quit"):
                self._pending_exit = True
                print(GOODBYE)
                self.running = False
                return
            session, message = quiz_feature.start_quiz(answer)
            print(message)

        while not session.is_complete:
            answer = self._read("Your answer (A/B/C/D): ")
            if answer is None:
                self._pending_exit = True
                print("Quiz interrupted.")
                return
            cleaned = answer.strip().lower()
            if cleaned in ("/skip", "skip"):
                session.skip()
                print(f"Skipped. Score: {session.score}/{session.answered}")
                if not session.is_complete:
                    print()
                    print(session.prompt())
                continue
            if cleaned in ("/cancel", "/exit", "/quit"):
                print("Quiz ended early.")
                return

            result = session.submit(answer)
            if result.get("status") == "invalid":
                print("Please answer with A, B, C or D (or type /skip).")
                continue
            if result["correct"]:
                print(paint("✅ Correct!", GREEN))
            else:
                print(
                    paint("❌ Incorrect.", RED)
                    + f" Correct answer: {result['correct_letter']}. "
                    + f"{result['correct_answer']}"
                )
            print(f"Score: {session.score}/{session.answered}")
            if not session.is_complete:
                print()
                print(session.prompt())

        print()
        print(paint("📊 RESULT", MAGENTA))
        print(session.result())
        self.history.add("/quiz", session.result(), self.mode)

    def run_planner(self) -> None:
        print()
        print(planner_feature.planner_intro())
        subjects_text = self._read("Subjects: ")
        if subjects_text is None:
            self._pending_exit = True
            return
        if subjects_text.strip().lower() in ("/cancel", "/exit", "/quit", ""):
            print("Planner cancelled.")
            return
        subjects = planner_feature.parse_subjects(subjects_text)
        if not subjects:
            print("I could not read those subjects. Example: python, maths, science")
            return

        print()
        print(planner_feature.ask_hours())
        hours_text = self._read("Hours: ")
        if hours_text is None:
            self._pending_exit = True
            return
        hours = planner_feature.parse_number(hours_text)
        if hours is None or hours <= 0 or hours > 24:
            print("Please enter a number like 3 or 2.5 (max 24).")
            return

        print()
        print(planner_feature.ask_days())
        days_text = self._read("Days: ")
        if days_text is None:
            self._pending_exit = True
            return
        days_value = planner_feature.parse_number(days_text)
        if days_value is None or days_value <= 0 or days_value > 365:
            print("Please enter a number of days between 1 and 365.")
            return

        try:
            plan = planner_feature.generate_plan(subjects, hours, int(days_value))
        except ValueError as exc:
            print(f"Could not build the plan: {exc}")
            return
        print()
        print(plan)
        self.history.add(
            f"/planner {', '.join(subjects)} ({hours:g}h x {int(days_value)}d)",
            plan.splitlines()[0],
            self.mode,
        )

    def show_history(self) -> None:
        entries = self.history.recent(config.HISTORY_DISPLAY_LIMIT)
        if not entries:
            print("No chat history yet.")
            return
        print(paint(f"🕘 Last {len(entries)} messages", MAGENTA))
        print("-" * 46)
        for entry in entries:
            stamp = str(entry.get("timestamp", ""))[:19]
            mode = entry.get("mode", "?")
            icon = "🌐" if mode == "online" else "📴"
            user = str(entry.get("user", ""))
            assistant = str(entry.get("assistant", ""))
            print(f"[{stamp}] {icon}")
            print(f"  You:  {user}")
            print(f"  Bot:  {assistant}")
            print("-" * 46)

    # ------------------------------------------------------------------ #
    # command handling
    # ------------------------------------------------------------------ #
    def handle_command(self, text: str) -> bool:
        """Handle slash commands. Returns True when input was a command."""
        name, args = command_utils.parse_command(text)
        if name is None:
            return False

        if name in ("/exit", "/quit"):
            print(GOODBYE)
            self.running = False
            return True
        if name == "/help":
            print()
            print(paint(command_utils.help_text(), CYAN))
            return True
        if name == "/mode":
            print(self.mode_line())
            if self.online:
                print(f"   Model: {config.GEMINI_MODEL}")
            else:
                print("   Engine: TF-IDF + Naive Bayes + offline KB")
            return True
        if name == "/clear":
            clear_screen()
            self.show_banner()
            return True
        if name == "/history":
            self.show_history()
            return True
        if name == "/quiz":
            if not self.running:
                return True
            self.run_quiz(args)
            return True
        if name == "/planner":
            if not self.running:
                return True
            self.run_planner()
            return True

        print(command_utils.unknown_command_message(text))
        return True

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #
    def process(self, text: str) -> None:
        """Handle one user utterance end to end."""
        stripped = text.strip()
        if not stripped:
            return

        if command_utils.is_exit(stripped) and not command_utils.parse_command(stripped)[0]:
            print(GOODBYE)
            self.running = False
            return

        if self.handle_command(text):
            return

        reply, mode = self.get_reply(stripped)
        icon = "🌐" if mode == "online" else "📴"
        print()
        print(paint(f"🤖 Assistant [{icon}]:", BOLD))
        print(reply)
        print()
        self.history.add(stripped, reply, mode)

    def run(self) -> None:
        self.show_banner()
        while self.running:
            try:
                text = self._read(paint("You: ", BOLD))
            except KeyboardInterrupt:
                print()
                text = None

            if text is None:
                print(GOODBYE)
                break
            if not text.strip():
                continue

            try:
                self.process(text)
            except Exception:
                logger.exception("Unhandled error while processing input")
                print(
                    paint(
                        "⚠️ Something went wrong. Please try again.", YELLOW
                    )
                )

            if self._pending_exit:
                if self.running:
                    print(GOODBYE)
                break

            if self.running:
                self.refresh_mode()

        logger.info("Chatbot stopped")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student AI Chatbot")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="force offline mode (skip connectivity check and Gemini)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="run startup validation and exit",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _init_io()
    setup_logging()
    _init_color()

    bot = StudentChatbot(force_offline=args.offline)
    bot.startup()

    if args.check:
        print("Startup validation")
        print(f"  directories : OK ({config.DATA_DIR}, {config.MODELS_DIR})")
        print(
            f"  intent model: "
            f"{'OK' if bot.classifier.loaded else 'MISSING (run python nlp/train.py)'}"
        )
        print(
            f"  knowledge   : "
            f"{'OK (' + str(bot.kb.size) + ' entries)' if bot.kb.loaded else 'MISSING'}"
        )
        print(f"  chat log    : {len(bot.history)} stored messages")
        print(f"  api key     : {'configured' if config.has_api_key() else 'not set'}")
        print(f"  mode        : {bot.mode_label()}")
        return 0

    try:
        bot.run()
    except KeyboardInterrupt:
        print()
        print(GOODBYE)
    except Exception:
        logger.exception("Fatal error")
        print(
            paint(
                "\n⚠️ The chatbot hit an unexpected error. Details are in logs/chatbot.log.",
                YELLOW,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
