"""Intent classification: rules first, machine learning as back-up.

1. Rule-based matching looks for known keywords and phrases. It is precise
   and easy to audit.
2. A TF-IDF + logistic regression model, trained on the example phrases in
   intents.json, catches messages the rules miss, including typos. It only
   answers when its confidence passes a threshold.
"""
import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from .preprocess import normalise

DATA_FILE = Path(__file__).parent / "data" / "intents.json"
ML_THRESHOLD = 0.35


@dataclass
class Match:
    tag: str
    group: str        # "topics" | "general" | "followups"
    source: str       # "rule" | "ml"
    confidence: float
    position: int = 0


def load_intents(path: Path = DATA_FILE) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class IntentClassifier:
    def __init__(self, intents: dict | None = None):
        self.intents = intents or load_intents()
        self.group_of = {}
        self._rules = []      # (compiled regex, tag)
        self._vocab = set()   # every word the bot has seen in its data
        texts, labels = [], []
        for group in ("topics", "general", "followups"):
            for intent in self.intents[group]:
                tag = intent["tag"]
                self.group_of[tag] = group
                for kw in intent.get("keywords", []):
                    kw_norm = normalise(kw)
                    plural = "s?" if " " not in kw_norm else ""
                    self._rules.append(
                        (re.compile(r"\b" + re.escape(kw_norm) + plural + r"\b"), tag))
                    self._vocab.update(kw_norm.split())
                    texts.append(kw_norm)
                    labels.append(tag)
                for pattern in intent["patterns"]:
                    texts.append(normalise(pattern))
                    labels.append(tag)
                    self._vocab.update(texts[-1].split())
        # Single-word symptom keywords that typos may be corrected to.
        self._spell_targets = sorted(
            {normalise(kw) for intent in self.intents["topics"]
             for kw in intent.get("keywords", [])
             if " " not in kw and len(kw) >= 5})
        self.model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                            sublinear_tf=True),
            LogisticRegression(C=20.0, max_iter=2000),
        )
        self.model.fit(texts, labels)

    # ---------------------------------------------------------- spelling --
    def correct_spelling(self, text: str) -> str:
        """Fix likely typos of symptom words ("hedache" -> "headache").

        Conservative on purpose: only unknown words of 4+ letters are
        changed, and only to a keyword with the same first letter.
        """
        out = []
        for token in text.split():
            if len(token) >= 4 and token.isalpha() and token not in self._vocab:
                close = difflib.get_close_matches(
                    token, [t for t in self._spell_targets if t[0] == token[0]],
                    n=1, cutoff=0.84)
                if close:
                    token = close[0]
            out.append(token)
        return " ".join(out)

    # ------------------------------------------------------------- rules --
    def rule_matches(self, text: str) -> list[Match]:
        """All keyword hits, longest first, dropping hits inside longer ones."""
        hits = []
        for regex, tag in self._rules:
            for m in regex.finditer(text):
                hits.append((m.end() - m.start(), m.start(), m.end(), tag))
        hits.sort(reverse=True)
        taken, out, seen = [], [], set()
        for _, start, end, tag in hits:
            if any(start < e and s < end for s, e in taken):
                continue
            taken.append((start, end))
            if tag not in seen:
                seen.add(tag)
                out.append(Match(tag, self.group_of[tag], "rule", 1.0, start))
        return sorted(out, key=lambda m: m.position)

    # ---------------------------------------------------------------- ml --
    def ml_match(self, text: str) -> Match | None:
        if not text:
            return None
        probs = self.model.predict_proba([text])[0]
        best = probs.argmax()
        tag = self.model.classes_[best]
        if probs[best] < ML_THRESHOLD:
            return None
        return Match(tag, self.group_of[tag], "ml", float(probs[best]))

    def classify(self, text: str) -> list[Match]:
        """Return matches for a normalised message (empty list if unknown)."""
        text = self.correct_spelling(text)
        matches = self.rule_matches(text)
        if matches:
            return matches
        ml = self.ml_match(text)
        return [ml] if ml else []
