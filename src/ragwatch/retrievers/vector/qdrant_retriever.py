"""Qdrant-based vector retriever."""

from ragwatch.core.schema import Document, RetrievedDocument
from ragwatch.embeddings.base import BaseEmbeddingModel
from ragwatch.retrievers.vector.base import BaseVectorRetriever


class QdrantRetriever(BaseVectorRetriever):
    """Retriever using Qdrant for vector storage and similarity search."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        collection_name: str = "ragwatch",
        location: str | None = None,
    ) -> None:
        """Initialize QdrantRetriever.

        Args:
            embedding_model: The embedding model to use.
            collection_name: Name of the Qdrant collection.
            location: Qdrant server URL, or None for in-memory mode.
        """
        super().__init__(embedding_model, collection_name)

        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError(
                "The 'qdrant-client' package is required for QdrantRetriever. "
                "Install it with: pip install ragwatch[vectordb]"
            ) from e

        if location:
            self._client = QdrantClient(url=location)
        else:
            self._client = QdrantClient(location=":memory:")

        self._vector_size: int | None = None

    def index(self, documents: list[Document]) -> None:
        """Index documents into Qdrant."""
        if not documents:
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        texts = [doc.text for doc in documents]
        embeddings = self._embedding_model.embed_texts(texts)

        # Create collection on first index call
        if self._vector_size is None:
            self._vector_size = len(embeddings[0])
            if self._client.collection_exists(self._collection_name):
                self._client.delete_collection(self._collection_name)
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )

        points = [
            PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    **doc.metadata,
                },
            )
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)

        for doc in documents:
            self._documents[doc.doc_id] = doc

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve top-k documents by cosine similarity."""
        if not self._documents or self._vector_size is None:
            return []

        query_embedding = self._embedding_model.embed_query(query)
        response = self._client.query_points(
            collection_name=self._collection_name,
            query=query_embedding,
            limit=top_k,
        )

        retrieved: list[RetrievedDocument] = []
        for rank, hit in enumerate(response.points, start=1):
            doc_id = hit.payload.get("doc_id", "")
            if doc_id not in self._documents:
                continue
            retrieved.append(
                RetrievedDocument(
                    document=self._documents[doc_id],
                    score=hit.score,
                    rank=rank,
                    retriever_name="qdrant",
                    metadata={"qdrant_id": hit.id},
                )
            )

        return retrieved
