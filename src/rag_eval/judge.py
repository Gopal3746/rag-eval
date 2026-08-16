from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .models import GoldenExample, JudgeScore


class Judge(ABC):
    @abstractmethod
    def score(self, example: GoldenExample, answer: str, contexts: list[str]) -> JudgeScore:
        raise NotImplementedError


class HeuristicJudge(Judge):
    """Stable, secret-free judge for CI regression gating."""

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {"the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for", "with"}
        return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2 and t not in stop}

    def score(self, example: GoldenExample, answer: str, contexts: list[str]) -> JudgeScore:
        answer_terms = self._terms(answer)
        reference_terms = self._terms(example.answer)
        context_terms = self._terms(" ".join(contexts))
        if not answer_terms:
            return JudgeScore(faithfulness=1, relevance=1, rationale="Empty answer.")
        faith_ratio = len(answer_terms & context_terms) / len(answer_terms)
        rel_denom = max(1, len(reference_terms))
        rel_ratio = len(answer_terms & reference_terms) / rel_denom
        faith = max(1, min(5, round(1 + 4 * faith_ratio)))
        relevance = max(1, min(5, round(1 + 4 * rel_ratio)))
        return JudgeScore(
            faithfulness=faith,
            relevance=relevance,
            rationale=(
                f"deterministic lexical judge: context_support={faith_ratio:.3f}, "
                f"reference_overlap={rel_ratio:.3f}"
            ),
        )
