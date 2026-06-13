"""Tests for pgvector retriever.

These tests are skipped unless RAGWATCH_PGVECTOR_URL is set in the environment,
since they require a running PostgreSQL instance with pgvector.
"""

import os

import pytest

from ragwatch.core.schema import Document

PGVECTOR_URL = os.environ.get("RAGWATCH_PGVECTOR_URL", "")

pytestmark = pytest.mark.skipif(
    not PGVECTOR_URL, reason="RAGWATCH_PGVECTOR_URL not set"
)


def _make_corpus() -> list[Document]:
    return [
        Document(
            doc_id="pgtest_d1",
            text="Paris is the capital of France.",
            metadata={"source": "geography"},
        ),
        Document(
            doc_id="pgtest_d2",
            text="Berlin is the capital of Germany.",
            metadata={"source": "geography"},
        ),
        Document(
            doc_id="pgtest_d3",
            text="Python is a programming language.",
            metadata={"source": "technology"},
        ),
    ]


@pytest.fixture
def pgvector_retriever():
    from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel
    from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

    embedding_model = DeterministicEmbeddingModel(dimension=64)
    retriever = PgVectorRetriever(
        embedding_model=embedding_model,
        collection_name="ragwatch_pytest",
        connection_string=PGVECTOR_URL,
        dataset_name="pytest_dataset",
        reset_collection=True,
    )
    return retriever


class TestPgVectorRetriever:
    def test_index_and_retrieve(self, pgvector_retriever) -> None:
        corpus = _make_corpus()
        pgvector_retriever.index(corpus)

        results = pgvector_retriever.retrieve("capital of France", top_k=2)
        assert len(results) >= 1
        assert results[0].retriever_name == "pgvector"
        assert results[0].rank == 1

    def test_retrieve_empty(self) -> None:
        """A collection with no embeddings returns empty results."""
        from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel
        from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

        embedding_model = DeterministicEmbeddingModel(dimension=64)
        retriever = PgVectorRetriever(
            embedding_model=embedding_model,
            collection_name="ragwatch_pytest_never_indexed",
            connection_string=PGVECTOR_URL,
        )
        results = retriever.retrieve("anything", top_k=5)
        assert results == []

    def test_metadata_preserved(self, pgvector_retriever) -> None:
        corpus = _make_corpus()
        pgvector_retriever.index(corpus)

        results = pgvector_retriever.retrieve("capital", top_k=3)
        assert len(results) >= 1
        # Check retriever metadata
        assert results[0].metadata["collection_name"] == "ragwatch_pytest"
        assert results[0].metadata["vector_db_type"] == "pgvector"
        assert results[0].metadata["embedding_dim"] == 64

    def test_document_metadata_in_db(self, pgvector_retriever) -> None:
        corpus = _make_corpus()
        pgvector_retriever.index(corpus)

        results = pgvector_retriever.retrieve("France", top_k=3)
        # Check that document metadata was preserved through DB round-trip
        sources = [r.document.metadata.get("source") for r in results]
        assert "geography" in sources or "technology" in sources

    def test_tables_exist_after_index(self, pgvector_retriever) -> None:
        from ragwatch.storage.postgres import get_connection

        corpus = _make_corpus()
        pgvector_retriever.index(corpus)

        conn = get_connection(PGVECTOR_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM ragwatch_documents WHERE doc_id LIKE 'pgtest_%'"
                )
                doc_count = cur.fetchone()[0]
                assert doc_count == 3

                cur.execute(
                    "SELECT COUNT(*) FROM ragwatch_embeddings WHERE collection_name = 'ragwatch_pytest'"
                )
                emb_count = cur.fetchone()[0]
                assert emb_count == 3
        finally:
            conn.close()
