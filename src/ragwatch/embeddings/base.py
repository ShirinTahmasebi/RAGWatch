"""Base class for embedding models."""

from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    """Interface for text embedding models."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return their vector representations."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query and return its vector representation."""
        ...
