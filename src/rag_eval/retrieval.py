from __future__ import annotations

import math
import re
from collections import Counter

from .models import Document, RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Tiny dependency-free BM25 retriever suitable for an evaluation fixture."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(f"{doc.title} {doc.text}") for doc in documents]
        self.term_freqs = [Counter(tokens) for tokens in self.tokens]
        self.avgdl = sum(len(tokens) for tokens in self.tokens) / len(self.tokens)
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.tokens:
            self.doc_freq.update(set(tokens))

    def _idf(self, token: str) -> float:
        n = len(self.documents)
        df = self.doc_freq.get(token, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 3) -> list[RetrievalResult]:
        if k <= 0:
            raise ValueError("k must be positive")
        q_tokens = tokenize(query)
        scored: list[tuple[float, int]] = []
        for idx, freqs in enumerate(self.term_freqs):
            dl = len(self.tokens[idx])
            score = 0.0
            for token in q_tokens:
                tf = freqs.get(token, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self._idf(token) * (tf * (self.k1 + 1)) / denom
            scored.append((score, idx))
        scored.sort(key=lambda item: (-item[0], self.documents[item[1]].id))
        return [
            RetrievalResult(
                doc_id=self.documents[idx].id,
                score=score,
                text=self.documents[idx].text,
            )
            for score, idx in scored[:k]
        ]
