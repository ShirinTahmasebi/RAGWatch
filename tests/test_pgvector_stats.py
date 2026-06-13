"""Tests for pgvector database statistics queries.

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


@pytest.fixture(autouse=True)
def _seed_data():
    """Seed some test data for stats queries."""
    from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel
    from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

    embedding_model = DeterministicEmbeddingModel(dimension=64)
    retriever = PgVectorRetriever(
        embedding_model=embedding_model,
        collection_name="stats_test_collection",
        connection_string=PGVECTOR_URL,
        dataset_name="stats_test_dataset",
        reset_collection=True,
    )
    corpus = [
        Document(
            doc_id="stats_d1",
            text="Document one for stats testing.",
            metadata={"source": "test_source_a"},
        ),
        Document(
            doc_id="stats_d2",
            text="Document two for stats testing with more text.",
            metadata={"source": "test_source_b"},
        ),
    ]
    retriever.index(corpus)
    yield


class TestPgVectorStats:
    def test_document_count(self) -> None:
        from ragwatch.storage.postgres import get_connection

        conn = get_connection(PGVECTOR_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM ragwatch_documents WHERE doc_id LIKE 'stats_%'"
                )
                count = cur.fetchone()[0]
                assert count == 2
        finally:
            conn.close()

    def test_embedding_count(self) -> None:
        from ragwatch.storage.postgres import get_connection

        conn = get_connection(PGVECTOR_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM ragwatch_embeddings WHERE collection_name = 'stats_test_collection'"
                )
                count = cur.fetchone()[0]
                assert count == 2
        finally:
            conn.close()

    def test_collection_count(self) -> None:
        from ragwatch.storage.postgres import get_connection

        conn = get_connection(PGVECTOR_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(DISTINCT collection_name) FROM ragwatch_embeddings WHERE collection_name = 'stats_test_collection'"
                )
                count = cur.fetchone()[0]
                assert count == 1
        finally:
            conn.close()

    def test_avg_text_length(self) -> None:
        from ragwatch.storage.postgres import get_connection

        conn = get_connection(PGVECTOR_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT AVG(LENGTH(text)) FROM ragwatch_documents WHERE doc_id LIKE 'stats_%'"
                )
                avg_len = cur.fetchone()[0]
                assert avg_len is not None
                assert avg_len > 0
        finally:
            conn.close()

    def test_dataset_name_stored(self) -> None:
        from ragwatch.storage.postgres import get_connection

        conn = get_connection(PGVECTOR_URL)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT dataset_name FROM ragwatch_documents WHERE doc_id = 'stats_d1'"
                )
                ds = cur.fetchone()[0]
                assert ds == "stats_test_dataset"
        finally:
            conn.close()
