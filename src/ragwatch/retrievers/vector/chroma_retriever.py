"""ChromaDB-based vector retriever."""

from ragwatch.core.schema import Document, RetrievedDocument
from ragwatch.embeddings.base import BaseEmbeddingModel
from ragwatch.retrievers.vector.base import BaseVectorRetriever


class ChromaRetriever(BaseVectorRetriever):
    """Retriever using ChromaDB for vector storage and similarity search."""

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        collection_name: str = "ragwatch",
    ) -> None:
        super().__init__(embedding_model, collection_name)

        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "The 'chromadb' package is required for ChromaRetriever. "
                "Install it with: pip install ragwatch[vectordb]"
            ) from e

        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, documents: list[Document]) -> None:
        """Index documents into ChromaDB."""
        if not documents:
            return

        texts = [doc.text for doc in documents]
        embeddings = self._embedding_model.embed_texts(texts)

        ids = [doc.doc_id for doc in documents]
        metadatas = [
            {k: str(v) for k, v in doc.metadata.items()} or None
            for doc in documents
        ]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        for doc in documents:
            self._documents[doc.doc_id] = doc

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve top-k documents by cosine similarity."""
        if not self._documents:
            return []

        query_embedding = self._embedding_model.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, len(self._documents)),
            include=["distances", "documents"],
        )

        retrieved: list[RetrievedDocument] = []
        ids = results["ids"][0] if results["ids"] else []
        distances = results["distances"][0] if results["distances"] else []

        for rank, (doc_id, distance) in enumerate(zip(ids, distances), start=1):
            if doc_id not in self._documents:
                continue
            # ChromaDB cosine distance: lower is better; similarity = 1 - distance
            score = 1.0 - distance
            retrieved.append(
                RetrievedDocument(
                    document=self._documents[doc_id],
                    score=score,
                    rank=rank,
                    retriever_name="chroma",
                    metadata={"distance": distance},
                )
            )

        return retrieved
