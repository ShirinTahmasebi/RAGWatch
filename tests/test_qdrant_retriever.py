"""Tests for Qdrant retriever."""

import pytest

from ragwatch.core.schema import Document

try:
    import qdrant_client  # noqa: F401

    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


pytestmark = pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")


def _make_corpus() -> list[Document]:
    return [
        Document(doc_id="d1", text="Paris is the capital of France."),
        Document(doc_id="d2", text="Berlin is the capital of Germany."),
        Document(doc_id="d3", text="Python is a programming language."),
    ]


@pytest.fixture
def qdrant_retriever():
    from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel
    from ragwatch.retrievers.vector.qdrant_retriever import QdrantRetriever

    embedding_model = DeterministicEmbeddingModel(dimension=64)
    retriever = QdrantRetriever(
        embedding_model=embedding_model,
        collection_name="test_collection",
    )
    return retriever


class TestQdrantRetriever:
    def test_index_and_retrieve(self, qdrant_retriever) -> None:
        corpus = _make_corpus()
        qdrant_retriever.index(corpus)

        results = qdrant_retriever.retrieve("capital of France", top_k=2)
        assert len(results) >= 1
        assert results[0].retriever_name == "qdrant"
        assert results[0].rank == 1
        assert results[0].score is not None

    def test_retrieve_empty(self, qdrant_retriever) -> None:
        results = qdrant_retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_respects_top_k(self, qdrant_retriever) -> None:
        corpus = _make_corpus()
        qdrant_retriever.index(corpus)

        results = qdrant_retriever.retrieve("capital", top_k=1)
        assert len(results) == 1

    def test_ranks_are_sequential(self, qdrant_retriever) -> None:
        corpus = _make_corpus()
        qdrant_retriever.index(corpus)

        results = qdrant_retriever.retrieve("capital", top_k=3)
        for i, result in enumerate(results, start=1):
            assert result.rank == i
