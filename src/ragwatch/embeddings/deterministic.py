"""Deterministic embedding model for testing."""

import hashlib
import struct

from ragwatch.embeddings.base import BaseEmbeddingModel


class DeterministicEmbeddingModel(BaseEmbeddingModel):
    """A deterministic embedding model that produces fixed-size vectors from text hashes.

    This does not provide semantic similarity — it is intended for unit tests only.
    The same input text always produces the same output vector.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._embed_one(query)

    def _embed_one(self, text: str) -> list[float]:
        # Use SHA-256 hash to generate deterministic bytes, then expand
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat hash bytes to fill desired dimension
        num_bytes_needed = self._dimension * 4  # 4 bytes per float32
        repeated = (h * (num_bytes_needed // len(h) + 1))[:num_bytes_needed]
        # Unpack as floats and normalize
        values = list(struct.unpack(f"<{self._dimension}f", repeated))
        # Normalize to unit vector
        norm = sum(v * v for v in values) ** 0.5
        if norm > 0:
            values = [v / norm for v in values]
        return values
