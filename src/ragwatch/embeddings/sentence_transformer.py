"""SentenceTransformer-based embedding model."""

from ragwatch.embeddings.base import BaseEmbeddingModel


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    """Embedding model backed by sentence-transformers.

    Default model: 'sentence-transformers/all-MiniLM-L6-v2'
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "The 'sentence-transformers' package is required for SentenceTransformerEmbeddingModel. "
                "Install it with: pip install ragwatch[vectordb]"
            ) from e

        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        embedding = self._model.encode([query], convert_to_numpy=True)
        return embedding[0].tolist()
