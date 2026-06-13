"""Tests for semantic KPI metrics and engine using a fake provider."""

import math

from ragwatch.core.schema import (
    Document,
    GenerationResult,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.report import KPIEngine
from ragwatch.metrics.semantic import (
    AnswerQuerySimilarityMetric,
    QueryContextSimilarityMaxMetric,
    QueryContextSimilarityMeanMetric,
    SemanticKPIEngine,
    embed_run,
)
from ragwatch.semantic.providers import BaseSemanticEmbeddingProvider

EXPECTED_SEMANTIC_KPI_IDS = [
    KPIId.QUERY_CONTEXT_SIMILARITY_MEAN.value,
    KPIId.QUERY_CONTEXT_SIMILARITY_MAX.value,
    KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN.value,
    KPIId.ANSWER_CONTEXT_SIMILARITY_MAX.value,
    KPIId.ANSWER_QUERY_SIMILARITY.value,
]


class FakeSemanticEmbeddingProvider(BaseSemanticEmbeddingProvider):
    """Deterministic fake provider mapping known texts to fixed vectors.

    Unknown texts map to a zero vector so similarities are undefined (None).
    """

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = mapping
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [self._mapping.get(t, [0.0, 0.0]) for t in texts]


def _make_run(
    query: str = "q",
    answer: str = "a",
    doc_texts: list[str] | None = None,
) -> RAGRun:
    doc_texts = doc_texts if doc_texts is not None else ["d1", "d2"]
    docs = [
        RetrievedDocument(
            document=Document(doc_id=f"doc{i}", text=text, metadata={}),
            score=1.0,
            rank=i + 1,
            retriever_name="fake",
        )
        for i, text in enumerate(doc_texts)
    ]
    return RAGRun(
        run_id="run-1",
        query=query,
        retrieved_documents=docs,
        generation=GenerationResult(answer=answer, generator_name="fake"),
        latency_ms=1.0,
    )


class TestSemanticMetricValues:
    def test_query_context_similarity_values(self) -> None:
        provider = FakeSemanticEmbeddingProvider(
            {
                "q": [1.0, 0.0],
                "a": [1.0, 0.0],
                "d1": [1.0, 0.0],  # identical to query -> 1.0
                "d2": [0.0, 1.0],  # orthogonal -> 0.0
            }
        )
        run = _make_run(doc_texts=["d1", "d2"])
        embeddings = embed_run(run, provider)

        mean = QueryContextSimilarityMeanMetric().compute(embeddings)
        max_ = QueryContextSimilarityMaxMetric().compute(embeddings)
        assert mean.value is not None and math.isclose(mean.value, 0.5)
        assert max_.value is not None and math.isclose(max_.value, 1.0)

    def test_answer_query_similarity(self) -> None:
        provider = FakeSemanticEmbeddingProvider(
            {"q": [1.0, 0.0], "a": [1.0, 0.0], "d1": [0.0, 1.0]}
        )
        run = _make_run(doc_texts=["d1"])
        embeddings = embed_run(run, provider)
        result = AnswerQuerySimilarityMetric().compute(embeddings)
        assert result.value is not None and math.isclose(result.value, 1.0)

    def test_embed_run_calls_provider_once(self) -> None:
        provider = FakeSemanticEmbeddingProvider({"q": [1.0], "a": [1.0]})
        run = _make_run(doc_texts=["d1", "d2"])
        embed_run(run, provider)
        assert provider.call_count == 1


class TestSemanticMetricEdgeCases:
    def test_empty_retrieved_docs_returns_none(self) -> None:
        provider = FakeSemanticEmbeddingProvider({"q": [1.0, 0.0], "a": [1.0, 0.0]})
        run = _make_run(doc_texts=[])
        embeddings = embed_run(run, provider)
        assert QueryContextSimilarityMeanMetric().compute(embeddings).value is None
        assert QueryContextSimilarityMaxMetric().compute(embeddings).value is None

    def test_empty_answer_returns_none(self) -> None:
        provider = FakeSemanticEmbeddingProvider({"q": [1.0, 0.0], "d1": [1.0, 0.0]})
        run = _make_run(answer="", doc_texts=["d1"])
        embeddings = embed_run(run, provider)
        assert embeddings.answer_vector is None
        assert AnswerQuerySimilarityMetric().compute(embeddings).value is None


class TestSemanticKPIEngine:
    def test_compute_returns_expected_kpi_ids(self) -> None:
        provider = FakeSemanticEmbeddingProvider(
            {"q": [1.0, 0.0], "a": [1.0, 0.0], "d1": [1.0, 0.0]}
        )
        engine = SemanticKPIEngine(provider)
        report = engine.compute(_make_run(doc_texts=["d1"]))
        assert [r.name for r in report.results] == EXPECTED_SEMANTIC_KPI_IDS
        assert report.metadata["source"] == "semantic_embedding_model"


class TestDefaultEngineUnaffected:
    def test_default_kpi_engine_works_without_semantic_deps(self) -> None:
        # The default deterministic engine must not require any semantic code.
        report = KPIEngine().compute(_make_run(doc_texts=["d1", "d2"]))
        names = {r.name for r in report.results}
        assert KPIId.TOTAL_LATENCY_MS.value in names
        # No semantic KPIs should appear in the default report.
        assert KPIId.ANSWER_QUERY_SIMILARITY.value not in names
