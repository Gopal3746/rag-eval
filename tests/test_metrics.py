from rag_eval.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_retrieval_metrics() -> None:
    retrieved = ["a", "b", "c"]
    relevant = ["b", "x"]
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert recall_at_k(retrieved, relevant, 3) == 1 / 2
    assert reciprocal_rank(retrieved, relevant) == 1 / 2
