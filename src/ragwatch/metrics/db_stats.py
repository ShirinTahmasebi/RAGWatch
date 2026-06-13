"""Database-based metrics computed from pgvector/Postgres tables."""

import json

from ragwatch.metrics.base import BaseDBMetric
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIResult


def _get_connection(connection_string: str):
    """Get a database connection with pgvector registered."""
    from ragwatch.storage.postgres import get_connection

    return get_connection(connection_string)


class DBDocumentCountMetric(BaseDBMetric):
    """Count of documents in ragwatch_documents."""

    kpi_id = KPIId.DB_DOCUMENT_COUNT

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ragwatch_documents")
                count = cur.fetchone()[0]
        finally:
            conn.close()
        return KPIResult(
            name=self.name,
            value=count,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class DBEmbeddingCountMetric(BaseDBMetric):
    """Count of embeddings in ragwatch_embeddings."""

    kpi_id = KPIId.DB_EMBEDDING_COUNT

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM ragwatch_embeddings")
                count = cur.fetchone()[0]
        finally:
            conn.close()
        return KPIResult(
            name=self.name,
            value=count,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class DBCollectionCountMetric(BaseDBMetric):
    """Count of distinct collections in ragwatch_embeddings."""

    kpi_id = KPIId.DB_COLLECTION_COUNT

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(DISTINCT collection_name) FROM ragwatch_embeddings"
                )
                count = cur.fetchone()[0]
        finally:
            conn.close()
        return KPIResult(
            name=self.name,
            value=count,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class DBAverageDocumentLengthMetric(BaseDBMetric):
    """Average character length of document text."""

    kpi_id = KPIId.DB_AVG_DOCUMENT_LENGTH

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT AVG(LENGTH(text)) FROM ragwatch_documents")
                avg_len = cur.fetchone()[0]
        finally:
            conn.close()
        value = float(avg_len) if avg_len is not None else None
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class DBAverageEmbeddingDimMetric(BaseDBMetric):
    """Average embedding dimension across all embeddings."""

    kpi_id = KPIId.DB_AVG_EMBEDDING_DIM

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT AVG(embedding_dim) FROM ragwatch_embeddings")
                avg_dim = cur.fetchone()[0]
        finally:
            conn.close()
        value = float(avg_dim) if avg_dim is not None else None
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class DBCollectionsMetric(BaseDBMetric):
    """List of collection names in the database."""

    kpi_id = KPIId.DB_COLLECTIONS

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT collection_name FROM ragwatch_embeddings ORDER BY collection_name"
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        names = [row[0] for row in rows]
        return KPIResult(
            name=self.name,
            value=", ".join(names) if names else None,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
            metadata={"collections": names},
        )


class DBDocumentsPerSourceMetric(BaseDBMetric):
    """Document counts grouped by source."""

    kpi_id = KPIId.DB_DOCUMENTS_PER_SOURCE

    def compute(self, connection_string: str) -> KPIResult:
        conn = _get_connection(connection_string)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(source, '(none)') AS src, COUNT(*)
                    FROM ragwatch_documents
                    GROUP BY source
                    ORDER BY COUNT(*) DESC
                """)
                rows = cur.fetchall()
        finally:
            conn.close()
        counts = {src: count for src, count in rows}
        return KPIResult(
            name=self.name,
            value=json.dumps(counts),
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
            metadata={"counts": counts},
        )


DEFAULT_DB_METRICS: list[BaseDBMetric] = [
    DBDocumentCountMetric(),
    DBEmbeddingCountMetric(),
    DBCollectionCountMetric(),
    DBAverageDocumentLengthMetric(),
    DBAverageEmbeddingDimMetric(),
    DBCollectionsMetric(),
    DBDocumentsPerSourceMetric(),
]
