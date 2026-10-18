"""A small BM25 retriever (pure stdlib)."""
from __future__ import annotations

import math
from collections import Counter

from .golden import Passage
from .text import tokenize, STOPWORDS


class BM25:
    def __init__(self, passages: list[Passage], k1: float = 1.2, b: float = 0.75, use_idf: bool = True,
                 drop_stopwords: bool = True):
        self.passages = passages
        self.k1, self.b, self.use_idf = k1, b, use_idf
        self.drop_stopwords = drop_stopwords
        self.docs = [self._toks(p.title + " " + p.text) for p in passages]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(len(d) for d in self.docs) / max(1, len(self.docs))
        df: Counter = Counter()
        for d in self.tf:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def _toks(self, text: str) -> list[str]:
        toks = tokenize(text)
        return [t for t in toks if t not in STOPWORDS] if self.drop_stopwords else toks

    def search(self, query: str, depth: int = 10) -> list[tuple[str, float]]:
        q = self._toks(query)
        scored = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                f = tf.get(t, 0)
                if not f:
                    continue
                w = self.idf.get(t, 0.0) if self.use_idf else 1.0
                s += w * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append((self.passages[i].pid, s))
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:depth]

