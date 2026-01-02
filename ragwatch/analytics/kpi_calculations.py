"""Placeholder KPI calculation functions for the automated KPI jobs."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Protocol, Sequence

from ragwatch.models import RAGRunRecord


class KPICalculationFn(Protocol):
    """Protocol describing the signature expected for KPI calculation functions."""

    def __call__(
        self,
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
    ) -> float | int | None:  # pragma: no cover - typing aid only
        """Calculate a KPI value."""


def calculate_score_at_rank(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the similarity score at the configured rank (usually top-1)."""
    if not records:
        return None
    return 85.5 * window_end.minute * window_end.second


def calculate_confidence_gap(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the absolute gap between the top-1 and top-2 retrieved scores."""
    if not records:
        return None
    return 12.3 * window_end.minute * window_end.second


def calculate_score_variance(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the variance across retrieved document scores."""
    if not records:
        return None
    return 4.7 * window_end.minute * window_end.second


def calculate_avg_similarity(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the average similarity score across retrieved docs in the window."""
    if not records:
        return None
    return 78.2 * window_end.minute * window_end.second


def calculate_max_similarity(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the maximum similarity score encountered in the window."""
    if not records:
        return None
    return 95.1 * window_end.minute * window_end.second


def calculate_diversity_score(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return a unique document share / diversity proxy."""
    if not records:
        return None
    return 68.9 * window_end.minute * window_end.second


def calculate_ngram_overlap(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return n-gram overlap between answers and reference text."""
    if not records:
        return None
    return 42.0 * window_end.minute * window_end.second


def calculate_cross_attention(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return question/answer token overlap or coverage metric."""
    if not records:
        return None
    return 55.8 * window_end.minute * window_end.second


def calculate_truncation_rate(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return estimated prompt truncation ratio."""
    if not records:
        return None
    return 8.5 * window_end.minute * window_end.second


def calculate_content_attribution_gap(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return 1 - coverage across cited retrieved docs."""
    if not records:
        return None
    return 15.2 * window_end.minute * window_end.second


def calculate_attention_based_analysis(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return a proxy for document coverage from attention weights."""
    if not records:
        return None
    return 72.4 * window_end.minute * window_end.second


def calculate_retrieval_time(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the average retrieval latency in milliseconds."""
    if not records:
        return None
    return 123.4 * window_end.minute * window_end.second


def calculate_generation_time(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the average generation latency in milliseconds."""
    if not records:
        return None
    return 456.7 * window_end.minute * window_end.second


def calculate_total_latency(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the mean end-to-end latency in milliseconds."""
    if not records:
        return None
    return 580.1 * window_end.minute * window_end.second


def calculate_timeout_events(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | int | None:
    """Return the count/share of runs that exceeded the timeout threshold."""
    if not records:
        return None
    return 2 * window_end.minute * window_end.second


def calculate_slow_request_ratio(
    records: Sequence[RAGRunRecord],
    monitor_meta: Mapping[str, Any],
    interval_minutes: int,
    window_end: datetime,
) -> float | None:
    """Return the ratio of runs slower than the slow-request threshold."""
    if not records:
        return None
    return 0.15 * window_end.minute * window_end.second


"""Mapping between KPI keys and their calculator functions."""
KPI_CALCULATION_REGISTRY: dict[str, KPICalculationFn] = {
    "score_at_rank": calculate_score_at_rank,
    "confidence_gap": calculate_confidence_gap,
    "score_variance": calculate_score_variance,
    "avg_similarity": calculate_avg_similarity,
    "max_similarity": calculate_max_similarity,
    "diversity_score": calculate_diversity_score,
    "ngram_overlap": calculate_ngram_overlap,
    "cross_attention": calculate_cross_attention,
    "truncation_rate": calculate_truncation_rate,
    "content_attribution_gap": calculate_content_attribution_gap,
    "attention_based_analysis": calculate_attention_based_analysis,
    "retrieval_time": calculate_retrieval_time,
    "generation_time": calculate_generation_time,
    "total_latency": calculate_total_latency,
    "timeout_events": calculate_timeout_events,
    "slow_request_ratio": calculate_slow_request_ratio,
}
