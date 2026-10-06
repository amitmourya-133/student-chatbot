"""Offline quiz feature.

Works entirely from local data - no internet required.

Flow::

    /quiz
     -> pick a subject
     -> answer A/B/C/D questions one at a time
     -> score after every question
     -> final result with percentage
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

QUESTIONS_PER_QUIZ = 5

# subject -> list of {"question", "options" (4), "answer" (index 0-3)}
QUIZ_BANK: dict[str, list[dict]] = {
    "Python": [
        {
            "question": "What is the output of: print(2 + 3)?",
            "options": ["4", "5", "6", "23"],
            "answer": 1,
        },
        {
            "question": "Which keyword defines a function in Python?",
            "options": ["func", "def", "function", "define"],
            "answer": 1,
        },
        {
            "question": "Which of these is a mutable type in Python?",
            "options": ["tuple", "string", "list", "int"],
            "answer": 2,
        },
        {
            "question": "What does len('python') return?",
            "options": ["5", "6", "7", "Error"],
            "answer": 1,
        },
        {
            "question": "Which symbol starts a single-line comment in Python?",
            "options": ["//", "#", "/*", "--"],
            "answer": 1,
        },
        {
            "question": "What is the type of 3 / 2 in Python 3?",
            "options": ["int", "float", "complex", "str"],
            "answer": 1,
        },
        {
            "question": "Which method adds an item to the end of a list?",
            "options": ["add()", "append()", "insert_all()", "extend_one()"],
            "answer": 1,
        },
        {
            "question": "What does 'if x == 5' check?",
            "options": [
                "Assignment of 5 to x",
                "Whether x equals 5",
                "Whether x is not 5",
                "A syntax error",
            ],
            "answer": 1,
        },
        {
            "question": "Which of these creates an empty dictionary?",
            "options": ["[]", "{}", "()", "set()"],
            "answer": 1,
        },
        {
            "question": "What is the correct file extension for Python files?",
            "options": [".pt", ".python", ".py", ".pyt"],
            "answer": 2,
        },
    ],
    "Science": [
        {
            "question": "What is the chemical formula of water?",
            "options": ["CO2", "H2O", "O2", "NaCl"],
            "answer": 1,
        },
        {
            "question": "Which planet is known as the Red Planet?",
            "options": ["Venus", "Jupiter", "Mars", "Saturn"],
            "answer": 2,
        },
        {
            "question": "What gas do plants absorb during photosynthesis?",
            "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
            "answer": 2,
        },
        {
            "question": "What is the speed of light approximately?",
            "options": [
                "3,000 km/s",
                "300,000 km/s",
                "30,000 km/s",
                "300 km/s",
            ],
            "answer": 1,
        },
        {
            "question": "Which organ pumps blood through the body?",
            "options": ["Liver", "Lungs", "Brain", "Heart"],
            "answer": 3,
        },
        {
            "question": "What is the boiling point of water at sea level?",
            "options": ["50 C", "90 C", "100 C", "120 C"],
            "answer": 2,
        },
        {
            "question": "Which force pulls objects toward the Earth?",
            "options": ["Magnetism", "Friction", "Gravity", "Inertia"],
            "answer": 2,
        },
        {
            "question": "The smallest unit of life is the:",
            "options": ["Atom", "Cell", "Molecule", "Tissue"],
            "answer": 1,
        },
        {
            "question": "Which metal is liquid at room temperature?",
            "options": ["Iron", "Mercury", "Aluminium", "Copper"],
            "answer": 1,
        },
        {
            "question": "Sound travels fastest through:",
            "options": ["Vacuum", "Air", "Water", "Steel"],
            "answer": 3,
        },
    ],
    "Math": [
        {
            "question": "What is 15% of 200?",
            "options": ["15", "25", "30", "35"],
            "answer": 2,
        },
        {
            "question": "What is the square root of 144?",
            "options": ["10", "11", "12", "14"],
            "answer": 2,
        },
        {
            "question": "Solve: 3x + 6 = 21. What is x?",
            "options": ["3", "4", "5", "6"],
            "answer": 2,
        },
        {
            "question": "What is the value of pi (to 2 decimal places)?",
            "options": ["3.14", "3.41", "2.14", "3.16"],
            "answer": 0,
        },
        {
            "question": "How many degrees are in a triangle?",
            "options": ["90", "180", "270", "360"],
            "answer": 1,
        },
        {
            "question": "What is 7 factorial (7!)?",
            "options": ["5040", "720", "40320", "360"],
            "answer": 0,
        },
        {
            "question": "Which of these is a prime number?",
            "options": ["21", "27", "29", "33"],
            "answer": 2,
        },
        {
            "question": "What is the LCM of 4 and 6?",
            "options": ["2", "12", "24", "10"],
            "answer": 1,
        },
        {
            "question": "If a bag has 3 red and 5 blue balls, probability of red?",
            "options": ["3/5", "3/8", "5/8", "1/3"],
            "answer": 1,
        },
        {
            "question": "What is the derivative of x squared?",
            "options": ["x", "2x", "2", "x/2"],
            "answer": 1,
        },
    ],
    "Computer": [
        {
            "question": "What does CPU stand for?",
            "options": [
                "Central Processing Unit",
                "Computer Personal Unit",
                "Central Program Utility",
                "Core Processing User",
            ],
            "answer": 0,
        },
        {
            "question": "Which of these is an operating system?",
            "options": ["Chrome", "Windows", "Google", "Python"],
            "answer": 1,
        },
        {
            "question": "What does RAM stand for?",
            "options": [
                "Read Access Memory",
                "Random Access Memory",
                "Run All Memory",
                "Rapid Access Module",
            ],
            "answer": 1,
        },
        {
            "question": "Which language runs natively in web browsers?",
            "options": ["Python", "C++", "JavaScript", "Java"],
            "answer": 2,
        },
        {
            "question": "What is 1010 in binary equal to in decimal?",
            "options": ["8", "10", "12", "14"],
            "answer": 1,
        },
        {
            "question": "Which company develops the Git version control system?",
            "options": ["Microsoft", "Google", "Linux community", "Meta"],
            "answer": 2,
        },
        {
            "question": "What does HTML stand for?",
            "options": [
                "HyperText Markup Language",
                "High Tech Modern Language",
                "Hyperlink Text Machine Language",
                "Home Tool Markup Language",
            ],
            "answer": 0,
        },
        {
            "question": "Which device stores data permanently?",
            "options": ["RAM", "CPU", "Hard disk / SSD", "Cache"],
            "answer": 2,
        },
        {
            "question": "What is the extension of a Python source file?",
            "options": [".py", ".pt", ".txt", ".exe"],
            "answer": 0,
        },
        {
            "question": "Which of these is NOT a web browser?",
            "options": ["Firefox", "Edge", "Opera", "Photoshop"],
            "answer": 3,
        },
    ],
    "General Knowledge": [
        {
            "question": "What is the capital of India?",
            "options": ["Mumbai", "New Delhi", "Kolkata", "Chennai"],
            "answer": 1,
        },
        {
            "question": "Which is the largest ocean on Earth?",
            "options": [
                "Atlantic Ocean",
                "Indian Ocean",
                "Pacific Ocean",
                "Arctic Ocean",
            ],
            "answer": 2,
        },
        {
            "question": "Who wrote the Indian national anthem?",
            "options": [
                "Bankim Chandra Chatterjee",
                "Rabindranath Tagore",
                "Sarojini Naidu",
                "Mahatma Gandhi",
            ],
            "answer": 1,
        },
        {
            "question": "Which is the longest river in the world?",
            "options": ["Ganga", "Amazon", "Nile", "Yangtze"],
            "answer": 2,
        },
        {
            "question": "How many players are in a cricket team?",
            "options": ["9", "10", "11", "12"],
            "answer": 2,
        },
        {
            "question": "Which gas makes up most of Earth's atmosphere?",
            "options": ["Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"],
            "answer": 2,
        },
        {
            "question": "The Great Wall is located in which country?",
            "options": ["India", "Japan", "China", "Nepal"],
            "answer": 2,
        },
        {
            "question": "Which planet is closest to the Sun?",
            "options": ["Venus", "Mercury", "Earth", "Mars"],
            "answer": 1,
        },
        {
            "question": "ISO stands for:",
            "options": [
                "International Student Organisation",
                "International Organization for Standardization",
                "Indian Standard Organisation",
                "Internal Standard Operation",
            ],
            "answer": 1,
        },
        {
            "question": "Which vitamin is produced by the skin in sunlight?",
            "options": ["Vitamin A", "Vitamin B12", "Vitamin C", "Vitamin D"],
            "answer": 3,
        },
    ],
}

LETTERS = ("A", "B", "C", "D")


def list_subjects() -> list[str]:
    return sorted(QUIZ_BANK.keys())


def normalise_subject(text: str) -> str | None:
    """Map free-form user input to a quiz subject name."""
    cleaned = text.strip().lower().rstrip("?.!")
    if not cleaned:
        return None
    aliases = {
        "py": "Python",
        "python": "Python",
        "sci": "Science",
        "science": "Science",
        "math": "Math",
        "maths": "Math",
        "mathematics": "Math",
        "computer": "Computer",
        "cs": "Computer",
        "it": "Computer",
        "computer science": "Computer",
        "gk": "General Knowledge",
        "general": "General Knowledge",
        "general knowledge": "General Knowledge",
        "general awareness": "General Knowledge",
        "quiz": "General Knowledge",
    }
    if cleaned in aliases:
        return aliases[cleaned]
    for subject in QUIZ_BANK:
        if cleaned == subject.lower():
            return subject
    return None


def parse_choice(text: str) -> int | None:
    """Parse an answer like ``A``, ``b``, ``2`` into an option index."""
    cleaned = text.strip().upper().rstrip(".!")
    if not cleaned:
        return None
    if cleaned in LETTERS:
        return LETTERS.index(cleaned)
    if cleaned.isdigit():
        number = int(cleaned)
        if 1 <= number <= 4:
            return number - 1
    # tolerate answers like "(a)" or "option b"
    match = re.search(r"\b([A-Da-d1-4])\b", text.strip())
    if match:
        token = match.group(1).upper()
        return LETTERS.index(token) if token in LETTERS else int(token) - 1
    return None


@dataclass
class QuizSession:
    """State machine for one quiz run."""

    subject: str
    questions: list[dict] = field(default_factory=list)
    index: int = 0
    score: int = 0
    answered: int = 0
    history: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.questions)

    @property
    def is_complete(self) -> bool:
        return self.index >= self.total

    @property
    def current(self) -> dict | None:
        if self.is_complete:
            return None
        return self.questions[self.index]

    def prompt(self) -> str:
        question = self.current
        if question is None:
            return ""
        lines = [
            f"Question {self.index + 1}/{self.total}",
            question["question"],
            "",
        ]
        for letter, option in zip(LETTERS, question["options"]):
            lines.append(f"{letter}. {option}")
        return "\n".join(lines)

    def submit(self, answer_text: str) -> dict:
        """Record an answer. ``choice`` may be ``A-D`` or ``1-4``."""
        question = self.current
        if question is None:
            return {"status": "complete"}

        choice = parse_choice(answer_text)
        if choice is None:
            return {"status": "invalid"}

        correct_index = int(question["answer"])
        correct = choice == correct_index
        if correct:
            self.score += 1
        self.answered += 1

        self.history.append(
            {
                "question": question["question"],
                "chosen": LETTERS[choice],
                "correct_letter": LETTERS[correct_index],
                "correct": correct,
            }
        )
        self.index += 1
        return {
            "status": "answered",
            "correct": correct,
            "correct_letter": LETTERS[correct_index],
            "correct_answer": question["options"][correct_index],
            "chosen": LETTERS[choice],
            "score": self.score,
            "answered": self.answered,
            "complete": self.is_complete,
        }

    def skip(self) -> dict:
        """Advance without scoring - used by the ``/skip`` command."""
        question = self.current
        if question is None:
            return {"status": "complete"}
        self.history.append(
            {
                "question": question["question"],
                "chosen": "-",
                "correct_letter": LETTERS[int(question["answer"])],
                "correct": False,
                "skipped": True,
            }
        )
        self.index += 1
        self.answered += 1
        return {
            "status": "skipped",
            "score": self.score,
            "answered": self.answered,
            "complete": self.is_complete,
        }

    @property
    def percentage(self) -> int:
        if self.answered == 0:
            return 0
        return round(self.score * 100 / self.answered)

    def result(self) -> str:
        total = self.total or 1
        percent = round(self.score * 100 / total)
        if percent >= 80:
            verdict = "Excellent work!"
        elif percent >= 60:
            verdict = "Good job! Keep practising."
        elif percent >= 40:
            verdict = "Not bad - revise the tricky topics."
        else:
            verdict = "Keep practising, you will improve."
        return (
            "Quiz Complete!\n"
            f"Subject: {self.subject}\n"
            f"Score: {self.score}/{total}\n"
            f"Percentage: {percent}%\n"
            f"{verdict}"
        )


def start_quiz(subject_text: str) -> tuple[QuizSession | None, str]:
    """Create a session for ``subject_text``.

    Returns ``(session, message)``; session is ``None`` for invalid subjects.
    """
    subject = normalise_subject(subject_text)
    if subject is None:
        available = ", ".join(list_subjects())
        return None, (
            "I don't have that subject yet. Available subjects:\n"
            f"{available}\n\nType the subject name to begin."
        )

    pool = list(QUIZ_BANK[subject])
    random.shuffle(pool)
    questions = pool[: min(QUESTIONS_PER_QUIZ, len(pool))]
    session = QuizSession(subject=subject, questions=questions)
    return session, (
        f"Starting {subject} quiz ({session.total} questions)!\n"
        "Answer with A, B, C or D.\n\n" + session.prompt()
    )
