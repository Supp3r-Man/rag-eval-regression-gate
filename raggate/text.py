"""Small text helpers shared by the retriever, the toy generators and the metrics."""
from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    "a an the of is are was be to in on for by and or what which how does do did many much "
    "can with from at it its that this as".split()
)


def stem(tok: str) -> str:
    """A deliberately light stemmer: retries->retry, requests->request, nothing clever."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _WORD.findall(text.lower())]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def contains_phrase(haystack: str, phrase: str) -> bool:
    """Whole-token containment, so the fact '5 MB' is not found inside '15 MB'."""
    h, p = normalize(haystack), normalize(phrase)
    if not p:
        return False
    return re.search(r"(?<![a-z0-9-])" + re.escape(p) + r"(?![a-z0-9-])", h) is not None


def is_number(tok: str) -> bool:
    return tok.isdigit()
