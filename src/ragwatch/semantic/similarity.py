"""Deterministic similarity utilities for semantic KPIs.

These helpers operate on plain numeric vectors and have no external
dependencies, so they are fully deterministic and testable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float | None:
    """Return the cosine similarity between two vectors.

    Returns ``None`` if either vector is empty, the lengths differ, or either
    vector has zero magnitude (so the similarity is undefined).
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return None

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return None

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def mean_cosine_similarity(
    query_vector: Sequence[float],
    context_vectors: list[Sequence[float]],
) -> float | None:
    """Return the mean cosine similarity of ``query_vector`` to each context vector.

    Returns ``None`` if the query vector or the context list is empty, or if no
    valid (defined) pairwise similarity can be computed.
    """
    sims = _pairwise_similarities(query_vector, context_vectors)
    if not sims:
        return None
    return sum(sims) / len(sims)


def max_cosine_similarity(
    query_vector: Sequence[float],
    context_vectors: list[Sequence[float]],
) -> float | None:
    """Return the maximum cosine similarity of ``query_vector`` to any context vector.

    Returns ``None`` if the query vector or the context list is empty, or if no
    valid (defined) pairwise similarity can be computed.
    """
    sims = _pairwise_similarities(query_vector, context_vectors)
    if not sims:
        return None
    return max(sims)


def _pairwise_similarities(
    query_vector: Sequence[float],
    context_vectors: list[Sequence[float]],
) -> list[float]:
    """Compute defined cosine similarities of the query against each context vector."""
    if not query_vector or not context_vectors:
        return []
    sims: list[float] = []
    for context_vector in context_vectors:
        sim = cosine_similarity(query_vector, context_vector)
        if sim is not None:
            sims.append(sim)
    return sims
