from rag_eval.models import Document
from rag_eval.retrieval import BM25Retriever


def test_bm25_ranks_specific_document_first() -> None:
    docs = [
        Document(id="a", title="Database", text="PostgreSQL stores application records."),
        Document(id="b", title="Broker", text="Redis is the Celery broker for embedding jobs."),
    ]
    result = BM25Retriever(docs).search("Which broker does Celery use?", k=1)
    assert result[0].doc_id == "b"
