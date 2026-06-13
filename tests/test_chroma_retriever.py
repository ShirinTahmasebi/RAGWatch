"""Tests for ChromaDB retriever."""

import pytest

from ragwatch.core.schema import Document

try:
    import chromadb  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


pytestmark = pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")


def _make_corpus() -> list[Document]:
    return [
        Document(doc_id="d1", text="Paris is the capital of France."),
        Document(doc_id="d2", text="Berlin is the capital of Germany."),
        Document(doc_id="d3", text="Python is a programming language."),
    ]


@pytest.fixture
def chroma_retriever():
    from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel
    from ragwatch.retrievers.vector.chroma_retriever import ChromaRetriever

    embedding_model = DeterministicEmbeddingModel(dimension=64)
    retriever = ChromaRetriever(
        embedding_model=embedding_model,
        collection_name="test_collection",
    )
    return retriever


class TestChromaRetriever:
    def test_index_and_retrieve(self, chroma_retriever) -> None:
        corpus = _make_corpus()
        chroma_retriever.index(corpus)

        results = chroma_retriever.retrieve("capital of France", top_k=2)
        assert len(results) >= 1
        assert results[0].retriever_name == "chroma"
        assert results[0].rank == 1
        assert results[0].score is not None

    def test_retrieve_empty(self, chroma_retriever) -> None:
        results = chroma_retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_respects_top_k(self, chroma_retriever) -> None:
        corpus = _make_corpus()
        chroma_retriever.index(corpus)

        results = chroma_retriever.retrieve("capital", top_k=1)
        assert len(results) == 1

    def test_ranks_are_sequential(self, chroma_retriever) -> None:
        corpus = _make_corpus()
        chroma_retriever.index(corpus)

        results = chroma_retriever.retrieve("capital", top_k=3)
        for i, result in enumerate(results, start=1):
            assert result.rank == i
