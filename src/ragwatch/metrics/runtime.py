"""Runtime metrics computed from RAGRun metadata."""

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.base import BaseMetric
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIResult


class TotalLatencyMsMetric(BaseMetric):
    """Total end-to-end latency of the RAG run."""

    kpi_id = KPIId.TOTAL_LATENCY_MS

    def compute(self, run: RAGRun) -> KPIResult:
        return KPIResult(
            name=self.name,
            value=run.latency_ms,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalLatencyMsMetric(BaseMetric):
    """Retrieval stage latency if available in metadata."""

    kpi_id = KPIId.RETRIEVAL_LATENCY_MS

    def compute(self, run: RAGRun) -> KPIResult:
        value = run.metadata.get("retrieval_latency_ms")
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class GenerationLatencyMsMetric(BaseMetric):
    """Generation stage latency if available in metadata."""

    kpi_id = KPIId.GENERATION_LATENCY_MS

    def compute(self, run: RAGRun) -> KPIResult:
        value = run.metadata.get("generation_latency_ms")
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


DEFAULT_RUNTIME_METRICS: list[BaseMetric] = [
    TotalLatencyMsMetric(),
    RetrievalLatencyMsMetric(),
    GenerationLatencyMsMetric(),
]
