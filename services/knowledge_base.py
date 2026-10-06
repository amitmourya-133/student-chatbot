"""Offline knowledge base: hybrid retrieval over student Q&A.

Ranking score = 0.45 * TF-IDF cosine similarity
              + 0.30 * IDF-weighted keyword coverage
              + 0.15 * shared topic (query vs. entry)
              + 0.10 * intent match (only when the classifier gives a hint)

Generic filler phrases such as ``in simple words`` are removed from both the
index and the query so they never dominate the similarity.  Exact question
matches always rank first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
from nlp.preprocessing import preprocess

logger = logging.getLogger(__name__)

NO_MATCH_MESSAGE = (
    "I couldn't find a good offline answer for that.\n"
    "Please connect to the internet and try again."
)

# Filler words that carry no topic signal in either English or Hinglish.
# They are dropped from the index and from every query before matching so
# phrases like "in simple words" can never dominate the similarity score.
GENERIC_TERMS = frozenset(
    {
        # English function words / fillers
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "am", "in", "on", "at", "to", "of", "for", "and", "or", "but", "nor",
        "it", "its", "this", "that", "these", "those",
        "with", "as", "by", "from", "into", "about",
        "do", "does", "did", "done", "doing",
        "i", "me", "my", "mine", "myself", "you", "your", "yours",
        "we", "us", "our", "ours", "they", "them", "their",
        "he", "him", "his", "she", "her",
        "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
        "can", "could", "would", "should", "will", "shall", "may", "might",
        "must", "please", "tell", "give", "show", "let", "just", "very",
        "really", "actually", "some", "any", "all", "there", "here", "also",
        "only", "even", "much", "many", "more", "most", "other", "another",
        "get", "got", "need", "want", "like",
        "simple", "simply", "words", "word", "easy", "easily", "basic",
        "basics", "plain",
        # Hinglish function words / fillers
        "kya", "hai", "hain", "hoga", "ho", "hota", "hoti",
        "kaise", "kese", "kyu", "kyon", "kab", "kahan", "kaun",
        "batao", "bataiye", "bata", "mujhe", "mujh",
        "samjha", "samjhao", "samajh", "chahiye",
        "ka", "ke", "ki", "ko", "se", "me", "aur", "ya", "bhi", "to",
    }
)

# Subject nouns used to decide whether query and entry talk about the same
# topic.  Only real topics belong here - never verbs or filler words.
TOPIC_TERMS = frozenset(
    {
        "python", "javascript", "typescript", "html", "css", "dom",
        "sql", "database", "mysql", "sqlite", "mongodb", "postgresql",
        "firebase", "git", "github", "java", "react", "node", "npm",
        "angular", "vue", "php", "ruby", "rust", "kotlin", "swift",
        "c", "c++", "c#", "excel", "powerpoint", "windows", "linux",
        "android", "ios", "responsive", "web", "frontend", "backend",
        "api", "http", "network", "server",
        "algorithm", "algorithms", "recursion", "oop", "compiler",
        "interpreter", "variable", "variables", "loop", "loops",
        "array", "arrays", "function", "functions", "string", "strings",
        "class", "classes", "object", "objects", "os",
        "photosynthesis", "pythagoras", "theorem", "algebra", "geometry",
        "calculus", "probability", "statistics",
    }
)

W_TFIDF = 0.45
W_KEYWORD = 0.30
W_TOPIC = 0.15
W_INTENT = 0.10

EXACT_SCORE = 1.0
MAX_NON_EXACT_SCORE = 0.97


def _content_tokens(text: str) -> list[str]:
    """Unique content tokens of ``text`` with generic filler words removed."""
    return list(dict.fromkeys(token for token in text.split() if token not in GENERIC_TERMS))


@dataclass
class KBMatch:
    question: str
    answer: str
    intent: str
    score: float


class KnowledgeBase:
    """Cosine-similarity search across ``data/offline_kb.json``."""

    def __init__(
        self,
        path: Optional[Path] = None,
        threshold: Optional[float] = None,
    ) -> None:
        self.path = path or config.KNOWLEDGE_BASE_FILE
        self.threshold = (
            config.KB_SIMILARITY_THRESHOLD if threshold is None else threshold
        )
        self.entries: list[dict] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.loaded = False
        self._entry_clean: list[str] = []
        self._entry_tokens: list[frozenset] = []
        self._entry_topics: list[frozenset] = []

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load(self) -> bool:
        """Load the KB and build the TF-IDF matrix. Never raises."""
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except FileNotFoundError:
            logger.error("Knowledge base not found: %s", self.path)
            return False
        except json.JSONDecodeError:
            logger.exception("Knowledge base contains invalid JSON: %s", self.path)
            return False
        except Exception:
            logger.exception("Failed to load knowledge base")
            return False

        entries = raw.get("knowledge_base") if isinstance(raw, dict) else raw
        if not isinstance(entries, list) or not entries:
            logger.error("Knowledge base is empty: %s", self.path)
            return False

        clean_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict)
            and entry.get("question")
            and entry.get("answer")
        ]
        if not clean_entries:
            logger.error("Knowledge base has no usable entries")
            return False

        try:
            questions = [preprocess(entry["question"]) for entry in clean_entries]
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=1,
                sublinear_tf=True,
                stop_words=sorted(GENERIC_TERMS),
            )
            self.matrix = self.vectorizer.fit_transform(questions)
        except Exception:
            logger.exception("Failed to vectorise knowledge base")
            return False

        self._entry_clean = questions
        self._entry_tokens = [
            frozenset(_content_tokens(question)) for question in questions
        ]
        self._entry_topics = [
            tokens & TOPIC_TERMS for tokens in self._entry_tokens
        ]

        self.entries = clean_entries
        self.loaded = True
        logger.info("Loaded knowledge base: %d entries", len(self.entries))
        return True

    def ensure_ready(self) -> bool:
        if self.loaded:
            return True
        return self.load()

    # ------------------------------------------------------------------ #
    # search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_n: int = 3,
        intent_hint: Optional[str] = None,
    ) -> list[KBMatch]:
        """Return up to ``top_n`` matches sorted by similarity (best first)."""
        if not self.ensure_ready():
            return []

        cleaned = preprocess(query)
        if not cleaned:
            return []

        try:
            query_vec = self.vectorizer.transform([cleaned])
            tfidf_scores = cosine_similarity(query_vec, self.matrix)[0]
        except Exception:
            logger.exception("Knowledge base search failed")
            return []

        content = _content_tokens(cleaned)
        vocabulary = self.vectorizer.vocabulary_
        idf = self.vectorizer.idf_

        # IDF-weighted coverage of the query's content words by each entry.
        query_terms: dict[str, float] = {}
        for token in content:
            index = vocabulary.get(token)
            if index is not None:
                query_terms[token] = float(idf[index])
        total_weight = sum(query_terms.values())

        query_topics = frozenset(content) & TOPIC_TERMS

        scores = np.zeros(len(self.entries), dtype=float)
        hint_active = bool(intent_hint) and intent_hint != "fallback"
        for index, entry in enumerate(self.entries):
            entry_tokens = self._entry_tokens[index]

            keyword = 0.0
            if total_weight:
                covered = sum(
                    weight
                    for token, weight in query_terms.items()
                    if token in entry_tokens
                )
                keyword = covered / total_weight

            topic = 1.0 if query_topics & self._entry_topics[index] else 0.0
            intent = (
                1.0
                if hint_active and entry.get("intent") == intent_hint
                else 0.0
            )

            score = (
                W_TFIDF * float(tfidf_scores[index])
                + W_KEYWORD * keyword
                + W_TOPIC * topic
                + W_INTENT * intent
            )
            if cleaned == self._entry_clean[index]:
                score = EXACT_SCORE
            else:
                score = min(score, MAX_NON_EXACT_SCORE)
            scores[index] = score

        order = np.argsort(scores)[::-1][:top_n]
        matches: list[KBMatch] = []
        for index in order:
            score = float(scores[index])
            if score < self.threshold:
                continue
            entry = self.entries[int(index)]
            matches.append(
                KBMatch(
                    question=str(entry.get("question", "")),
                    answer=str(entry.get("answer", "")),
                    intent=str(entry.get("intent", "general")),
                    score=score,
                )
            )
        return matches

    def find_answer(
        self,
        query: str,
        intent_hint: Optional[str] = None,
    ) -> Optional[KBMatch]:
        """Best match above the threshold, or ``None``."""
        matches = self.search(query, top_n=1, intent_hint=intent_hint)
        return matches[0] if matches else None

    @property
    def size(self) -> int:
        return len(self.entries)
