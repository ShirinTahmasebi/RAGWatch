"""Tests for database-based KPI metrics.

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
    """Seed test data for DB metric queries."""
    from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel
    from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

    embedding_model = DeterministicEmbeddingModel(dimension=64)
    retriever = PgVectorRetriever(
        embedding_model=embedding_model,
        collection_name="db_kpi_test",
        connection_string=PGVECTOR_URL,
        dataset_name="kpi_test_dataset",
        reset_collection=True,
    )
    corpus = [
        Document(
            doc_id="dbkpi_d1",
            text="First document for KPI testing.",
            metadata={"source": "source_alpha"},
        ),
        Document(
            doc_id="dbkpi_d2",
            text="Second document for KPI testing with longer text content.",
            metadata={"source": "source_beta"},
        ),
        Document(
            doc_id="dbkpi_d3",
            text="Third document.",
            metadata={"source": "source_alpha"},
        ),
    ]
    retriever.index(corpus)
    yield


class TestDBDocumentCount:
    def test_returns_positive_count(self) -> None:
        from ragwatch.metrics.db_stats import DBDocumentCountMetric

        result = DBDocumentCountMetric().compute(PGVECTOR_URL)
        assert result.value >= 3
        assert result.name == "db_document_count"
        assert result.category == "db_index_stats"


class TestDBEmbeddingCount:
    def test_returns_positive_count(self) -> None:
        from ragwatch.metrics.db_stats import DBEmbeddingCountMetric

        result = DBEmbeddingCountMetric().compute(PGVECTOR_URL)
        assert result.value >= 3


class TestDBCollectionCount:
    def test_returns_at_least_one(self) -> None:
        from ragwatch.metrics.db_stats import DBCollectionCountMetric

        result = DBCollectionCountMetric().compute(PGVECTOR_URL)
        assert result.value >= 1


class TestDBAverageDocumentLength:
    def test_returns_positive_float(self) -> None:
        from ragwatch.metrics.db_stats import DBAverageDocumentLengthMetric

        result = DBAverageDocumentLengthMetric().compute(PGVECTOR_URL)
        assert result.value is not None
        assert result.value > 0


class TestDBAverageEmbeddingDim:
    def test_returns_positive_float(self) -> None:
        from ragwatch.metrics.db_stats import DBAverageEmbeddingDimMetric

        result = DBAverageEmbeddingDimMetric().compute(PGVECTOR_URL)
        assert result.value is not None
        assert result.value > 0


class TestDBCollections:
    def test_returns_collection_names(self) -> None:
        from ragwatch.metrics.db_stats import DBCollectionsMetric

        result = DBCollectionsMetric().compute(PGVECTOR_URL)
        assert result.value is not None
        assert "db_kpi_test" in result.value
        assert "collections" in result.metadata


class TestDBDocumentsPerSource:
    def test_returns_json_string(self) -> None:
        import json

        from ragwatch.metrics.db_stats import DBDocumentsPerSourceMetric

        result = DBDocumentsPerSourceMetric().compute(PGVECTOR_URL)
        assert result.value is not None
        parsed = json.loads(result.value)
        assert isinstance(parsed, dict)
        assert len(parsed) >= 1


class TestDBKPIEngine:
    def test_compute_returns_all_metrics(self) -> None:
        from ragwatch.metrics.report import DBKPIEngine

        engine = DBKPIEngine()
        report = engine.compute(connection_string=PGVECTOR_URL)
        assert report.run_id is None
        assert report.query is None
        assert len(report.results) == 7

        names = [r.name for r in report.results]
        assert "db_document_count" in names
        assert "db_embedding_count" in names
        assert "db_collection_count" in names
        assert "db_avg_document_length" in names
        assert "db_avg_embedding_dim" in names
        assert "db_collections" in names
        assert "db_documents_per_source" in names
