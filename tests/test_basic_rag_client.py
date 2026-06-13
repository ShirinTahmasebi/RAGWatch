"""Tests for the BasicRAGClient."""

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.core.schema import Document, RAGRun
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever


def _build_client() -> BasicRAGClient:
    documents = [
        Document(doc_id="d1", text="Paris is the capital of France."),
        Document(doc_id="d2", text="Berlin is the capital of Germany."),
        Document(doc_id="d3", text="Tokyo is the capital of Japan."),
    ]
    retriever = TfidfRetriever()
    retriever.index(documents)
    generator = HeuristicGenerator()
    return BasicRAGClient(retriever=retriever, generator=generator)


def test_run_returns_rag_run() -> None:
    client = _build_client()
    result = client.run("What is the capital of France?", top_k=2)

    assert isinstance(result, RAGRun)
    assert result.query == "What is the capital of France?"
    assert result.run_id  # non-empty
    assert result.latency_ms >= 0


def test_run_contains_retrieved_documents() -> None:
    client = _build_client()
    result = client.run("capital of France", top_k=2)

    assert len(result.retrieved_documents) >= 1
    assert result.retrieved_documents[0].document.doc_id == "d1"


def test_run_contains_generation() -> None:
    client = _build_client()
    result = client.run("capital of France", top_k=2)

    assert result.generation.answer
    assert result.generation.generator_name == "heuristic"


def test_run_metadata_includes_component_info() -> None:
    client = _build_client()
    result = client.run("capital of France", top_k=3)

    assert result.metadata["top_k"] == 3
    assert result.metadata["retriever"] == "TfidfRetriever"
    assert result.metadata["generator"] == "HeuristicGenerator"
    assert "num_retrieved_documents" in result.metadata
