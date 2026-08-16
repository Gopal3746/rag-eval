from __future__ import annotations

import re

from .models import GoldenExample, RetrievalResult


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def generate_extractively(example: GoldenExample, retrieved: list[RetrievalResult]) -> str:
    """Deterministic local answerer used for CI and smoke tests.

    It ranks retrieved sentences by overlap with the golden example's answer keywords.
    Production RAG output can be evaluated by replacing this function or using the library API.
    """
    if not retrieved:
        return "I do not have enough context to answer."
    keywords = {token.lower() for token in example.answer_keywords}
    candidates: list[tuple[int, str]] = []
    for result in retrieved:
        for sentence in _sentences(result.text):
            lowered = sentence.lower()
            score = sum(keyword in lowered for keyword in keywords)
            candidates.append((score, sentence))
    candidates.sort(key=lambda item: (-item[0], len(item[1])))
    best = candidates[0][1] if candidates else retrieved[0].text
    return best
