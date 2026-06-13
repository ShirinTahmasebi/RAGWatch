"""Optional semantic (embedding-based) KPI metrics.

These metrics measure semantic relationships between the query, the retrieved
context, and the generated answer using cosine similarity over embeddings from
a configurable provider. They are intentionally separate from the default
deterministic :class:`KPIEngine` so the base project never requires semantic
dependencies or API keys.

This is a first relevance layer only. LLM-as-judge faithfulness is not
implemented here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.catalog import KPI_CATALOG, KPIId
from ragwatch.metrics.schema import KPIReport, KPIResult
from ragwatch.semantic.providers import BaseSemanticEmbeddingProvider
from ragwatch.semantic.similarity import (
    cosine_similarity,
    max_cosine_similarity,
    mean_cosine_similarity,
)


@dataclass
class SemanticRunEmbeddings:
    """Embeddings computed once per run and shared across semantic metrics."""

    query_vector: list[float] | None
    answer_vector: list[float] | None
    context_vectors: list[list[float]] = field(default_factory=list)


def embed_run(
    run: RAGRun, provider: BaseSemanticEmbeddingProvider
) -> SemanticRunEmbeddings:
    """Embed the query, answer, and retrieved document texts once for a run.

    Empty query/answer produce ``None`` vectors; empty retrieved documents
    produce an empty context list. All non-empty texts are embedded in a single
    provider call to avoid repeated round trips.
    """
    query = run.query or ""
    answer = run.generation.answer or ""
    context_texts = [rd.document.text for rd in run.retrieved_documents]

    # Build a single batch of the texts we actually need to embed.
    batch: list[str] = []
    query_idx: int | None = None
    answer_idx: int | None = None
    context_indices: list[int] = []

    if query.strip():
        query_idx = len(batch)
        batch.append(query)
    if answer.strip():
        answer_idx = len(batch)
        batch.append(answer)
    for text in context_texts:
        context_indices.append(len(batch))
        batch.append(text)

    vectors = provider.embed_texts(batch) if batch else []

    return SemanticRunEmbeddings(
        query_vector=vectors[query_idx] if query_idx is not None else None,
        answer_vector=vectors[answer_idx] if answer_idx is not None else None,
        context_vectors=[vectors[i] for i in context_indices],
    )


class BaseSemanticMetric(ABC):
    """Abstract base for semantic metrics computed from shared run embeddings."""

    kpi_id: KPIId

    @property
    def name(self) -> str:
        return str(self.kpi_id.value)

    @property
    def category(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].category)

    @property
    def stage(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].stage)

    @property
    def source(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].source)

    @property
    def description(self) -> str:
        return KPI_CATALOG[self.kpi_id].description

    def _result(self, value: float | None) -> KPIResult:
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )

    @abstractmethod
    def compute(self, embeddings: SemanticRunEmbeddings) -> KPIResult:
        """Compute this semantic metric from precomputed run embeddings."""
        ...


class QueryContextSimilarityMeanMetric(BaseSemanticMetric):
    """Mean cosine similarity between the query and each retrieved document."""

    kpi_id = KPIId.QUERY_CONTEXT_SIMILARITY_MEAN

    def compute(self, embeddings: SemanticRunEmbeddings) -> KPIResult:
        if embeddings.query_vector is None or not embeddings.context_vectors:
            return self._result(None)
        return self._result(
            mean_cosine_similarity(
                embeddings.query_vector, embeddings.context_vectors
            )
        )


class QueryContextSimilarityMaxMetric(BaseSemanticMetric):
    """Maximum cosine similarity between the query and any retrieved document."""

    kpi_id = KPIId.QUERY_CONTEXT_SIMILARITY_MAX

    def compute(self, embeddings: SemanticRunEmbeddings) -> KPIResult:
        if embeddings.query_vector is None or not embeddings.context_vectors:
            return self._result(None)
        return self._result(
            max_cosine_similarity(
                embeddings.query_vector, embeddings.context_vectors
            )
        )


class AnswerContextSimilarityMeanMetric(BaseSemanticMetric):
    """Mean cosine similarity between the answer and each retrieved document."""

    kpi_id = KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN

    def compute(self, embeddings: SemanticRunEmbeddings) -> KPIResult:
        if embeddings.answer_vector is None or not embeddings.context_vectors:
            return self._result(None)
        return self._result(
            mean_cosine_similarity(
                embeddings.answer_vector, embeddings.context_vectors
            )
        )


class AnswerContextSimilarityMaxMetric(BaseSemanticMetric):
    """Maximum cosine similarity between the answer and any retrieved document."""

    kpi_id = KPIId.ANSWER_CONTEXT_SIMILARITY_MAX

    def compute(self, embeddings: SemanticRunEmbeddings) -> KPIResult:
        if embeddings.answer_vector is None or not embeddings.context_vectors:
            return self._result(None)
        return self._result(
            max_cosine_similarity(
                embeddings.answer_vector, embeddings.context_vectors
            )
        )


class AnswerQuerySimilarityMetric(BaseSemanticMetric):
    """Cosine similarity between the generated answer and the query."""

    kpi_id = KPIId.ANSWER_QUERY_SIMILARITY

    def compute(self, embeddings: SemanticRunEmbeddings) -> KPIResult:
        if embeddings.answer_vector is None or embeddings.query_vector is None:
            return self._result(None)
        return self._result(
            cosine_similarity(embeddings.answer_vector, embeddings.query_vector)
        )


DEFAULT_SEMANTIC_METRICS: list[BaseSemanticMetric] = [
    QueryContextSimilarityMeanMetric(),
    QueryContextSimilarityMaxMetric(),
    AnswerContextSimilarityMeanMetric(),
    AnswerContextSimilarityMaxMetric(),
    AnswerQuerySimilarityMetric(),
]


class SemanticKPIEngine:
    """Compute semantic KPIs from a RAGRun using an embedding provider.

    The engine embeds each run once (query, answer, retrieved texts) and shares
    those embeddings across all configured metrics.
    """

    def __init__(
        self,
        provider: BaseSemanticEmbeddingProvider,
        metrics: list[BaseSemanticMetric] | None = None,
    ) -> None:
        self._provider = provider
        self._metrics = (
            metrics if metrics is not None else DEFAULT_SEMANTIC_METRICS
        )

    @property
    def metrics(self) -> list[BaseSemanticMetric]:
        """Return the list of configured semantic metrics."""
        return self._metrics

    def compute(self, run: RAGRun) -> KPIReport:
        """Compute all semantic metrics for a RAGRun and return a KPIReport."""
        embeddings = embed_run(run, self._provider)
        results = [metric.compute(embeddings) for metric in self._metrics]
        return KPIReport(
            run_id=run.run_id,
            query=run.query,
            results=results,
            metadata={"source": "semantic_embedding_model"},
        )
