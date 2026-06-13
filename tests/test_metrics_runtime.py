"""Tests for runtime metrics."""

from ragwatch.core.schema import (
    Document,
    GenerationResult,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.metrics.runtime import (
    GenerationLatencyMsMetric,
    RetrievalLatencyMsMetric,
    TotalLatencyMsMetric,
)


def _make_run(
    latency_ms: float = 10.0,
    metadata: dict | None = None,
) -> RAGRun:
    return RAGRun(
        run_id="test-run",
        query="test query",
        retrieved_documents=[],
        generation=GenerationResult(answer="answer", generator_name="test"),
        latency_ms=latency_ms,
        metadata=metadata or {},
    )


class TestTotalLatency:
    def test_returns_latency_ms(self) -> None:
        run = _make_run(latency_ms=42.5)
        result = TotalLatencyMsMetric().compute(run)
        assert result.value == 42.5
        assert result.name == "total_latency_ms"


class TestRetrievalLatency:
    def test_returns_value_from_metadata(self) -> None:
        run = _make_run(metadata={"retrieval_latency_ms": 15.0})
        result = RetrievalLatencyMsMetric().compute(run)
        assert result.value == 15.0

    def test_returns_none_when_missing(self) -> None:
        run = _make_run(metadata={})
        result = RetrievalLatencyMsMetric().compute(run)
        assert result.value is None


class TestGenerationLatency:
    def test_returns_value_from_metadata(self) -> None:
        run = _make_run(metadata={"generation_latency_ms": 7.5})
        result = GenerationLatencyMsMetric().compute(run)
        assert result.value == 7.5

    def test_returns_none_when_missing(self) -> None:
        run = _make_run(metadata={})
        result = GenerationLatencyMsMetric().compute(run)
        assert result.value is None
