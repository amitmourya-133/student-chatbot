"""Centralized response policy.

Composes the system instruction handed to Gemini from:

* the base system prompt (``config.SYSTEM_PROMPT``)
* remembered student context (memory/profile)
* the active study mode
* the preferred language
* the response style (normal/short/detailed/exam/beginner/teacher)

Study modes steer *structure and terminology*; they never hard-code subject
content. The same policy class is used by the CLI, the GUI and the pipeline.
"""

from __future__ import annotations

import config

# --------------------------------------------------------------------------- #
# study modes
# --------------------------------------------------------------------------- #
STUDY_MODES = {
    "general": {
        "label": "General",
        "instruction": "",
    },
    "mathematics": {
        "label": "Mathematics",
        "instruction": (
            "Study mode: Mathematics. Structure maths answers as: Given, "
            "Formula/Concept, Calculation, Final Answer. Show each step, keep "
            "units/exponents explicit, and verify arithmetic where possible."
        ),
    },
    "physics": {
        "label": "Physics",
        "instruction": (
            "Study mode: Physics. Structure physics answers as: Concept, "
            "Formula, Units, Derivation/Example, Final Result. State "
            "assumptions and keep units consistent (SI preferred)."
        ),
    },
    "chemistry": {
        "label": "Chemistry",
        "instruction": (
            "Study mode: Chemistry. Structure chemistry answers as: Concept, "
            "Reaction/Process, Balanced Equation (when relevant), Observation, "
            "Conclusion. Use proper chemical formulae and state symbols."
        ),
    },
    "programming": {
        "label": "Programming",
        "instruction": (
            "Study mode: Programming. Structure coding answers as: "
            "Explanation, Code (clean, runnable), Code Walkthrough, Common "
            "Mistakes. Prefer Python unless the user asks for another language."
        ),
    },
    "exam": {
        "label": "Exam Preparation",
        "instruction": (
            "Study mode: Exam Preparation. Answer like an exam question: "
            "start with a crisp definition, then key points, then a short "
            "example or diagram description, ending with 'Important Exam "
            "Points' when useful."
        ),
    },
}
DEFAULT_STUDY_MODE = "general"

# --------------------------------------------------------------------------- #
# languages
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "english": "",
    "hindi": (
        "Language: user prefers Hindi. Reply in clear, natural Hindi. "
        "Keep technical terms in English where that helps."
    ),
    "hinglish": (
        "Language: user prefers Hinglish. Reply mixing Hindi and English "
        "naturally (e.g. 'normalization ka concept aise samjho')."
    ),
}
DEFAULT_LANGUAGE = "english"

# --------------------------------------------------------------------------- #
# response styles
# --------------------------------------------------------------------------- #
STYLES = {
    "normal": "",
    "short": (
        "Style: Short. Give the direct answer in 2-4 short lines. Skip "
        "unnecessary detail."
    ),
    "detailed": (
        "Style: Detailed. Explain thoroughly with sections, examples and "
        "context. Treat the user as a student who wants full understanding."
    ),
    "exam": (
        "Style: Exam. Answer precisely and completely enough for full marks, "
        "as an examiner would expect (definition, points, example, conclusion)."
    ),
    "beginner": (
        "Style: Beginner. Assume no prior knowledge. Use simple words, small "
        "sentences and analogies. Avoid jargon, or explain each term the "
        "first time it appears."
    ),
    "teacher": (
        "Style: Teacher. Patient, encouraging, structured. End with one "
        "checking question the student can answer to confirm understanding."
    ),
}
DEFAULT_STYLE = "normal"


def valid_study_modes() -> list[str]:
    return sorted(STUDY_MODES)


def valid_styles() -> list[str]:
    return sorted(STYLES)


def valid_languages() -> list[str]:
    return sorted(LANGUAGES)


def study_mode_label(mode: str) -> str:
    entry = STUDY_MODES.get(mode)
    return entry["label"] if entry else STUDY_MODES[DEFAULT_STUDY_MODE]["label"]


def compose_system(
    base: str = config.SYSTEM_PROMPT,
    memory_lines: list[str] | None = None,
    profile_lines: list[str] | None = None,
    study_mode: str = DEFAULT_STUDY_MODE,
    language: str = DEFAULT_LANGUAGE,
    style: str = DEFAULT_STYLE,
) -> str:
    """Build one system instruction from reusable pieces."""
    blocks = [base]
    context = list(memory_lines or []) + list(profile_lines or [])
    if context:
        blocks.append(
            "Student context (use only when relevant, never say it unless asked):\n"
            + "\n".join(context)
        )
    mode_instruction = STUDY_MODES.get(study_mode, {}).get("instruction")
    if mode_instruction:
        blocks.append(mode_instruction)
    language_instruction = LANGUAGES.get(language)
    if language_instruction:
        blocks.append(language_instruction)
    style_instruction = STYLES.get(style)
    if style_instruction:
        blocks.append(style_instruction)
    return "\n\n".join(block for block in blocks if block)