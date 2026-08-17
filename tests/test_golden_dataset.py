from rag_eval.io import load_documents, load_golden


def test_golden_dataset_has_expected_size_and_valid_refs() -> None:
    docs = load_documents("data/corpus.jsonl")
    golden = load_golden("data/golden.jsonl")
    doc_ids = {doc.id for doc in docs}
    assert 20 <= len(golden) <= 30
    assert all(set(row.relevant_doc_ids) <= doc_ids for row in golden)
