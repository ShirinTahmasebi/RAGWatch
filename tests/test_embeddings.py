"""Tests for embedding models."""

from ragwatch.embeddings.deterministic import DeterministicEmbeddingModel


class TestDeterministicEmbeddingModel:
    def test_produces_correct_dimension(self) -> None:
        model = DeterministicEmbeddingModel(dimension=32)
        result = model.embed_query("hello")
        assert len(result) == 32

    def test_is_deterministic(self) -> None:
        model = DeterministicEmbeddingModel(dimension=64)
        v1 = model.embed_query("test input")
        v2 = model.embed_query("test input")
        assert v1 == v2

    def test_different_texts_produce_different_vectors(self) -> None:
        model = DeterministicEmbeddingModel(dimension=64)
        v1 = model.embed_query("hello")
        v2 = model.embed_query("world")
        assert v1 != v2

    def test_embed_texts_returns_list_of_vectors(self) -> None:
        model = DeterministicEmbeddingModel(dimension=16)
        results = model.embed_texts(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(v) == 16 for v in results)

    def test_vectors_are_normalized(self) -> None:
        model = DeterministicEmbeddingModel(dimension=64)
        vec = model.embed_query("normalize me")
        norm = sum(v * v for v in vec) ** 0.5
        assert abs(norm - 1.0) < 1e-6
