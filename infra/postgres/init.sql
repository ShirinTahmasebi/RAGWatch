-- RAGWatch PostgreSQL schema initialization
-- This runs automatically when the Docker container starts for the first time.

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Table: ragwatch_documents
-- Stores the corpus documents indexed by RAGWatch retrievers.
CREATE TABLE IF NOT EXISTS ragwatch_documents (
    doc_id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    dataset_name TEXT,
    source TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_dataset_name ON ragwatch_documents(dataset_name);
CREATE INDEX IF NOT EXISTS idx_documents_source ON ragwatch_documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON ragwatch_documents USING GIN (metadata);

-- Table: ragwatch_embeddings
-- Stores vector embeddings linked to documents and organized by collection.
CREATE TABLE IF NOT EXISTS ragwatch_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
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
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON ragwatch_embeddings(embedding_model);

-- TODO: Add HNSW or IVFFlat vector index once embedding dimensions are fixed.
-- Example (for 384-dim embeddings):
--   CREATE INDEX ON ragwatch_embeddings USING hnsw (embedding vector_cosine_ops)
--   WITH (m = 16, ef_construction = 64);
-- For now, exact search is used which is fine for research-scale datasets.
