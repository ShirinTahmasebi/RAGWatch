"""PostgreSQL connection and table initialization helpers."""

from __future__ import annotations

from pathlib import Path

_INIT_SQL = Path(__file__).resolve().parent.parent.parent.parent / "infra" / "postgres" / "init.sql"

# Inline DDL in case the init.sql is not available (e.g., when installed as a package)
_CREATE_TABLES_SQL = """\
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS ragwatch_documents (
    doc_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    dataset_name TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ragwatch_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_name TEXT NOT NULL,
    doc_id TEXT NOT NULL REFERENCES ragwatch_documents(doc_id) ON DELETE CASCADE,
    embedding_model TEXT,
    embedding_dim INT,
    embedding VECTOR,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_collection ON ragwatch_embeddings(collection_name);
CREATE INDEX IF NOT EXISTS idx_embeddings_doc_id ON ragwatch_embeddings(doc_id);
"""


def get_connection(connection_string: str):
    """Create a psycopg connection with pgvector registered."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn = psycopg.connect(connection_string)
    register_vector(conn)
    return conn


def ensure_tables(connection_string: str) -> None:
    """Create RAGWatch tables if they don't exist."""
    conn = get_connection(connection_string)
    try:
        with conn.cursor() as cur:
            cur.execute(_CREATE_TABLES_SQL)
        conn.commit()
    finally:
        conn.close()
