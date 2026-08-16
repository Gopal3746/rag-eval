from __future__ import annotations

from dataclasses import asdict
from statistics import mean

from .generator import generate_extractively
from .judge import Judge
from .metrics import precision_at_k, recall_at_k, reciprocal_rank
from .models import Document, ExampleResult, GoldenExample
from .retrieval import BM25Retriever


def evaluate(
    documents: list[Document],
    examples: list[GoldenExample],
    judge: Judge,
    k: int = 3,
) -> dict:
    retriever = BM25Retriever(documents)
    rows: list[ExampleResult] = []

    for example in examples:
        retrieved = retriever.search(example.question, k=k)
        retrieved_ids = [item.doc_id for item in retrieved]
        answer = generate_extractively(example, retrieved)
        judge_score = judge.score(example, answer, [item.text for item in retrieved])
        rows.append(
            ExampleResult(
                example_id=example.id,
                question=example.question,
                generated_answer=answer,
                retrieved_doc_ids=retrieved_ids,
                precision_at_k=precision_at_k(retrieved_ids, example.relevant_doc_ids, k),
                recall_at_k=recall_at_k(retrieved_ids, example.relevant_doc_ids, k),
                reciprocal_rank=reciprocal_rank(retrieved_ids, example.relevant_doc_ids),
                faithfulness=judge_score.faithfulness / 5.0,
                relevance=judge_score.relevance / 5.0,
                judge_rationale=judge_score.rationale,
            )
        )

    summary = {
        "examples": len(rows),
        "k": k,
        "precision_at_k": mean(row.precision_at_k for row in rows),
        "recall_at_k": mean(row.recall_at_k for row in rows),
        "mrr": mean(row.reciprocal_rank for row in rows),
        "faithfulness": mean(row.faithfulness for row in rows),
        "relevance": mean(row.relevance for row in rows),
    }
    return {"summary": summary, "examples": [asdict(row) for row in rows]}
