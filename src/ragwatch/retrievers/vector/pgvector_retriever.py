"""pgvector-based vector retriever using persistent PostgreSQL tables."""

from __future__ import annotations

import json
from typing import Any

from ragwatch.config.env import get_env, load_env
from ragwatch.core.schema import Document, RetrievedDocument
from ragwatch.embeddings.base import BaseEmbeddingModel
from ragwatch.retrievers.vector.base import BaseVectorRetriever


class PgVectorRetriever(BaseVectorRetriever):
    """Retriever using PostgreSQL with pgvector for persistent vector similarity search.

    Documents are stored in `ragwatch_documents` and embeddings in `ragwatch_embeddings`.
    This allows SQL-based analysis and statistics on the stored data.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        collection_name: str = "ragwatch",
        connection_string: str | None = None,
        dataset_name: str | None = None,
        reset_collection: bool = False,
    ) -> None:
        """Initialize PgVectorRetriever.

        Args:
            embedding_model: The embedding model to use.
            collection_name: Logical collection name for organizing embeddings.
            connection_string: PostgreSQL connection string. If not provided,
                reads from RAGWATCH_PGVECTOR_URL environment variable.
            dataset_name: Optional dataset name stored with documents.
            reset_collection: If True, delete existing embeddings for this
                collection before indexing.
        """
        super().__init__(embedding_model, collection_name)

        try:
            import psycopg  # noqa: F401
            from pgvector.psycopg import register_vector  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "The 'psycopg' and 'pgvector' packages are required for PgVectorRetriever. "
                "Install them with: pip install ragwatch[vectordb]"
            ) from e

        if connection_string is None:
            load_env()
            connection_string = get_env("RAGWATCH_PGVECTOR_URL", required=True)

        self._connection_string: str = connection_string  # type: ignore[assignment]
        self._dataset_name = dataset_name
        self._reset_collection = reset_collection
        self._embedding_dim: int | None = None

    def _get_connection(self):
        """Create a psycopg connection with pgvector registered."""
        from ragwatch.storage.postgres import get_connection

        return get_connection(self._connection_string)

    def _ensure_tables(self) -> None:
        """Create RAGWatch tables if they don't exist."""
        from ragwatch.storage.postgres import ensure_tables

        ensure_tables(self._connection_string)

    def index(self, documents: list[Document]) -> None:
        """Index documents into PostgreSQL ragwatch_documents and ragwatch_embeddings."""
        if not documents:
            return

        self._ensure_tables()

        # Embed all documents
        texts = [doc.text for doc in documents]
        embeddings = self._embedding_model.embed_texts(texts)
        self._embedding_dim = len(embeddings[0])

        # Get embedding model name if available
        model_name = getattr(self._embedding_model, "model_name", None) or type(
            self._embedding_model
        ).__name__

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                # Reset collection if requested
                if self._reset_collection:
                    cur.execute(
                        "DELETE FROM ragwatch_embeddings WHERE collection_name = %s",
                        (self._collection_name,),
                    )

                # Upsert documents
                for doc in documents:
                    source = doc.metadata.get("source")
                    cur.execute(
                        """
                        INSERT INTO ragwatch_documents (doc_id, text, metadata, dataset_name, source)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (doc_id) DO UPDATE SET
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            dataset_name = EXCLUDED.dataset_name,
                            source = EXCLUDED.source
                        """,
                        (
                            doc.doc_id,
                            doc.text,
                            json.dumps(doc.metadata),
                            self._dataset_name,
                            source,
                        ),
                    )

                # Insert embeddings (delete old ones for this collection+doc first)
                for doc, embedding in zip(documents, embeddings):
                    cur.execute(
                        "DELETE FROM ragwatch_embeddings WHERE collection_name = %s AND doc_id = %s",
                        (self._collection_name, doc.doc_id),
                    )
                    cur.execute(
                        """
                        INSERT INTO ragwatch_embeddings
                            (collection_name, doc_id, embedding_model, embedding_dim, embedding, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            self._collection_name,
                            doc.doc_id,
                            model_name,
                            self._embedding_dim,
                            embedding,
                            json.dumps(doc.metadata),
                        ),
                    )

            conn.commit()
        finally:
            conn.close()

        # Keep local cache for building RetrievedDocument objects
        for doc in documents:
            self._documents[doc.doc_id] = doc

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve top-k documents by cosine similarity from the collection."""
        self._ensure_tables()

        if self._embedding_dim is None:
            # Try to infer dimension from existing embeddings
            conn = self._get_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT embedding_dim FROM ragwatch_embeddings WHERE collection_name = %s LIMIT 1",
                        (self._collection_name,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        return []
                    self._embedding_dim = row[0]
            finally:
                conn.close()

        query_embedding = self._embedding_model.embed_query(query)

        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT e.doc_id, d.text, d.metadata,
                           1 - (e.embedding <=> %s::vector) AS similarity
                    FROM ragwatch_embeddings e
                    JOIN ragwatch_documents d ON e.doc_id = d.doc_id
                    WHERE e.collection_name = %s
                    ORDER BY e.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, self._collection_name, query_embedding, top_k),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        retrieved: list[RetrievedDocument] = []
        for rank, (doc_id, text, metadata_raw, similarity) in enumerate(rows, start=1):
            meta: dict[str, Any] = {}
            if metadata_raw:
                if isinstance(metadata_raw, str):
                    meta = json.loads(metadata_raw)
                else:
                    meta = dict(metadata_raw)

            doc = Document(doc_id=doc_id, text=text, metadata=meta)
            self._documents[doc_id] = doc

            retrieved.append(
                RetrievedDocument(
                    document=doc,
                    score=float(similarity),
                    rank=rank,
                    retriever_name="pgvector",
                    metadata={
                        "collection_name": self._collection_name,
                        "vector_db_type": "pgvector",
                        "embedding_dim": self._embedding_dim,
                    },
                )
            )

        return retrieved
