"""Tests for the TF-IDF retriever."""

from ragwatch.core.schema import Document
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever


def _make_corpus() -> list[Document]:
    return [
        Document(doc_id="d1", text="Paris is the capital of France."),
        Document(doc_id="d2", text="Berlin is the capital of Germany."),
        Document(doc_id="d3", text="Python is a programming language."),
    ]


def test_retrieve_returns_relevant_documents() -> None:
    retriever = TfidfRetriever()
    retriever.index(_make_corpus())

    results = retriever.retrieve("capital of France", top_k=2)

    assert len(results) >= 1
    assert results[0].document.doc_id == "d1"
    assert results[0].score > 0
    assert results[0].rank == 1
    assert results[0].retriever_name == "tfidf"


def test_retrieve_ranks_are_sequential() -> None:
    retriever = TfidfRetriever()
    retriever.index(_make_corpus())

    results = retriever.retrieve("capital", top_k=3)

    for i, result in enumerate(results, start=1):
        assert result.rank == i


def test_retrieve_scores_are_decreasing() -> None:
    retriever = TfidfRetriever()
    retriever.index(_make_corpus())

    results = retriever.retrieve("capital", top_k=3)

    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


def test_retrieve_empty_index() -> None:
    retriever = TfidfRetriever()
    results = retriever.retrieve("anything", top_k=5)
    assert results == []


def test_retrieve_respects_top_k() -> None:
    retriever = TfidfRetriever()
    retriever.index(_make_corpus())

    results = retriever.retrieve("capital", top_k=1)
    assert len(results) <= 1
