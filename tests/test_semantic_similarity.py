"""Tests for semantic similarity utilities (deterministic, no dependencies)."""

import math

from ragwatch.semantic.similarity import (
    cosine_similarity,
    max_cosine_similarity,
    mean_cosine_similarity,
)


class TestCosineSimilarity:
    def test_identical_vectors_returns_one(self) -> None:
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 1.0

    def test_orthogonal_vectors_returns_zero(self) -> None:
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors_returns_minus_one(self) -> None:
        assert math.isclose(cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_none(self) -> None:
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) is None

    def test_empty_vector_returns_none(self) -> None:
        assert cosine_similarity([], [1.0]) is None

    def test_mismatched_lengths_returns_none(self) -> None:
        assert cosine_similarity([1.0, 2.0], [1.0]) is None


class TestMeanAndMaxCosineSimilarity:
    def test_mean_empty_context_returns_none(self) -> None:
        assert mean_cosine_similarity([1.0, 2.0], []) is None

    def test_mean_empty_query_returns_none(self) -> None:
        assert mean_cosine_similarity([], [[1.0, 2.0]]) is None

    def test_mean_of_two_contexts(self) -> None:
        # query == first context (sim 1.0), orthogonal to second (sim 0.0)
        value = mean_cosine_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        assert value is not None
        assert math.isclose(value, 0.5)

    def test_max_of_two_contexts(self) -> None:
        value = max_cosine_similarity([1.0, 0.0], [[0.0, 1.0], [1.0, 0.0]])
        assert value is not None
        assert math.isclose(value, 1.0)

    def test_mean_skips_undefined_zero_vector(self) -> None:
        # The zero context vector yields an undefined similarity and is skipped.
        value = mean_cosine_similarity([1.0, 0.0], [[1.0, 0.0], [0.0, 0.0]])
        assert value is not None
        assert math.isclose(value, 1.0)

    def test_max_all_undefined_returns_none(self) -> None:
        assert max_cosine_similarity([1.0, 0.0], [[0.0, 0.0]]) is None
