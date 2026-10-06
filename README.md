# 🎓 Student AI Chatbot

A production-quality, fully functional **Student AI Chatbot** written in Python.
It runs in two modes and switches automatically:

| Mode | Trigger | Engine |
|------|---------|--------|
| 🌐 **ONLINE** | Internet available **and** `GEMINI_API_KEY` set | Google Gemini API |
| 📴 **OFFLINE** | No internet / no key / API failure | TF-IDF + Naive Bayes + offline knowledge base (215 Q&A) |

The bot **never crashes** when the internet, the API key or the model files are
missing - it degrades gracefully to offline mode.

---

## Features

- 🌐 Online mode with Google Gemini (`google-genai` SDK, REST fallback)
- 📴 Offline mode: `TF-IDF` → `Multinomial Naive Bayes` intent classification
- 📚 Offline knowledge base: 215 student Q&A entries with cosine-similarity search (threshold `0.30`)
- 🧠 **Quiz** (`/quiz`) - 5 subjects × 10 questions, live scoring and percentage
- 📅 **Study Planner** (`/planner`) - day-wise timetable with breaks and revision slots
- 🕘 **Chat History** (`/history`) - saved to `data/chat_log.json`, corruption-safe
- 🌏 **Hinglish support** - `python kya hai`, `quiz start karo`, `study plan bana do`
- 🛡️ Error handling everywhere: bad JSON, missing files, API errors, invalid input
- 🧪 Test suite: 93 tests (`pytest`)

## Project structure

```text
student-chatbot/
├── main.py                 # entry point + chat loop
├── config.py               # environment-driven configuration
├── requirements.txt
├── .env.example            # copy to .env
├── data/
│   ├── intents.json        # 165 training phrases, 12 intents
│   ├── offline_kb.json     # 215 offline Q&A entries
│   └── chat_log.json       # created automatically
├── models/
│   ├── intent_model.pkl    # created by training
│   └── vectorizer.pkl
├── nlp/
│   ├── preprocessing.py    # text normalisation
│   ├── train.py            # TF-IDF + Naive Bayes training
│   └── classifier.py       # model loading / prediction
├── services/
│   ├── internet.py         # socket connectivity check (8.8.8.8:53)
│   ├── gemini.py           # Gemini client + failure mapping
│   ├── knowledge_base.py   # cosine-similarity KB search
│   └── chat_history.py     # JSON chat log
├── features/
│   ├── quiz.py
│   ├── planner.py
│   └── commands.py
├── utils/logger.py
└── tests/                  # pytest suite
```

## Installation

```bash
python -m venv .venv
```

Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

Linux / macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Get a free key at <https://aistudio.google.com/apikey>.

> The app **works without** `.env` and without an API key - it simply stays in
> offline mode. The key is never hardcoded and `.env` is git-ignored.

## Train the model

```bash
python nlp/train.py
```

Output:

```text
Model trained successfully.
  samples=164 intents=12 features=466 train_accuracy=98.78%
```

Training runs **only** when you execute this command. If `models/*.pkl` are
missing, `main.py` auto-trains them once on first startup (safe bootstrap).

## Run

```bash
python main.py
```

Useful flags:

```bash
python main.py --offline   # force offline mode (skip connectivity check)
python main.py --check     # startup validation report, then exit
```

## Commands

```text
/help       Show commands
/quiz       Start a quiz
/planner    Create a study plan
/history    Show recent chat history
/mode       Show current mode
/clear      Clear screen
/exit       Exit chatbot
/quit       Exit chatbot
```

Plain `quit`, `exit`, `bye` also exit the application.

## How it works

```text
USER INPUT
  → quit/exit check
  → preprocessing
  → special command? (/help, /quiz, /planner, ...)
  → internet check
       ├─ ONLINE  → Gemini API → full AI answer
       └─ OFFLINE → TF-IDF + Naive Bayes → intent
                     → cosine-similarity KB search
                     → score ≥ 0.30 ? KB answer : offline fallback
  → display response + mode indicator
  → save chat history
  → loop
```

**Online mode** sends your message plus recent history to Gemini with a
*Student AI Assistant* system prompt (programming, maths, science, study
planning, debugging, Hinglish friendly).

**Offline mode** never touches the network. If Gemini fails at any point
(bad key, timeout, rate limit, service outage) you see:

```text
⚠️ Online AI unavailable. Switching to offline mode.
```

and the offline pipeline answers the same question.

## Quiz

```text
You: /quiz
Choose a subject:
   Computer, General Knowledge, Math, Python, Science
Subject: Python
Question 1/5
Which keyword defines a function in Python?
A. func
B. def
C. function
D. define
Your answer (A/B/C/D): B
✅ Correct!
...
Quiz Complete!
Score: 4/5
Percentage: 80%
```

Type `/skip` to skip a question, `/cancel` to end the quiz.

## Study planner

```text
You: /planner
Subjects: python, maths and science
Hours: 3
Days: 7
```

Produces a day-wise timetable:

```text
DAY 1
09:00 - 10:00  python
10:15 - 11:15  maths
11:30 - 12:00  Revision & Practice
```

## Tests

```bash
python -m pytest
```

Covers preprocessing, intent prediction, KB matching and the similarity
threshold, command parsing, exit handling, quiz scoring and chat history.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Intent model not found` | Run `python nlp/train.py` (or just start `main.py` - it auto-trains) |
| Always offline despite internet | Check `GEMINI_API_KEY` in `.env`; run `python main.py --check` |
| `⚠️ Online AI unavailable` | Key invalid/expired, rate limited or offline - the bot keeps working offline |
| `Knowledge base could not be loaded` | Validate `data/offline_kb.json` (must be valid JSON with a `knowledge_base` array) |
| `UnicodeEncodeError` on old terminals | The bot already falls back; or run `chcp 65001` in CMD |
| Logs | See `logs/chatbot.log` (no secrets are ever logged) |

## Security

- `GEMINI_API_KEY` is read from the environment only - never hardcoded
- `.env` and `models/*.pkl` are git-ignored
- API keys, stack traces and raw HTTP bodies are never printed to the user
