from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(doc_id in rel for doc_id in top) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    rel = set(relevant)
    if not rel:
        return 0.0
    return len(set(retrieved[:k]) & rel) / len(rel)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / rank
    return 0.0
