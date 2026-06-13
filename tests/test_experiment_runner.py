"""Tests for ExperimentRunner."""

from ragwatch.core.interfaces import BaseRAGClient
from ragwatch.core.schema import (
    Document,
    GenerationResult,
    QAExample,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import ExperimentConfig
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIReport, KPIResult
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.semantic.providers import BaseSemanticEmbeddingProvider


class _MockClient(BaseRAGClient):
    """A simple mock client that returns a fixed RAGRun."""

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id="d1", text="mock text", metadata={"source": "mock"}
                ),
                score=0.9,
                rank=1,
                retriever_name="mock",
            )
        ]
        return RAGRun(
            run_id="mock-run-id",
            query=query,
            retrieved_documents=docs,
            generation=GenerationResult(answer="mock answer", generator_name="mock"),
            latency_ms=1.0,
            metadata={
                "retrieval_latency_ms": 0.5,
                "generation_latency_ms": 0.5,
                "total_latency_ms": 1.0,
            },
        )


class _FailingClient(BaseRAGClient):
    """A client that always raises an exception."""

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        raise RuntimeError("Simulated failure")


class _MixedClient(BaseRAGClient):
    """A client that fails on a specific query."""

    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on
        self._good = _MockClient()

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        if query == self._fail_on:
            raise RuntimeError("boom")
        return self._good.run(query, top_k=top_k)


class _FakeDBEngine:
    """Stand-in for DBKPIEngine that returns a fixed report."""

    def compute(self, connection_string: str | None = None) -> KPIReport:
        return KPIReport(
            run_id=None,
            query=None,
            results=[
                KPIResult(
                    name="db_document_count",
                    value=42,
                    category="db_index_stats",
                    stage="database",
                    source="postgres",
                    description="docs",
                )
            ],
            metadata={"source": "database"},
        )


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test_experiment",
        dataset_name="test_dataset",
        retriever_name="mock",
        generator_name="mock",
        top_k=3,
    )


def _make_examples(n: int = 3) -> list[QAExample]:
    return [
        QAExample(
            example_id=f"ex{i}",
            question=f"Question {i}?",
            answers=[f"answer_{i}"],
        )
        for i in range(n)
    ]


class TestExperimentRunner:
    def test_runs_over_multiple_examples(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(5), config=_make_config())

        assert len(result.examples) == 5
        assert result.num_examples == 5
        assert result.num_successful == 5
        assert result.num_failed == 0

    def test_one_result_per_example(self) -> None:
        examples = _make_examples(4)
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=examples, config=_make_config())

        assert len(result.examples) == len(examples)
        ids = [e.example_id for e in result.examples]
        assert ids == ["ex0", "ex1", "ex2", "ex3"]

    def test_successful_examples_have_run_and_report(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(2), config=_make_config())

        for ex in result.examples:
            assert ex.succeeded
            assert ex.run is not None
            assert ex.kpi_report is not None
            assert ex.error is None

    def test_stores_example_metadata_on_run(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(2), config=_make_config())

        first = result.examples[0]
        assert first.run is not None
        assert first.run.metadata["example_id"] == "ex0"
        assert first.run.metadata["gold_answers"] == ["answer_0"]
        assert first.run.metadata["question"] == "Question 0?"

    def test_respects_limit(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(
            qa_examples=_make_examples(10), config=_make_config(), limit=3
        )

        assert result.num_examples == 3

    def test_respects_config_max_examples(self) -> None:
        config = _make_config()
        config.max_examples = 2
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(10), config=config)

        assert result.num_examples == 2

    def test_failing_client_does_not_crash_experiment(self) -> None:
        runner = ExperimentRunner(client=_FailingClient())
        result = runner.run(qa_examples=_make_examples(3), config=_make_config())

        assert result.num_examples == 3
        assert result.num_successful == 0
        assert result.num_failed == 3
        for ex in result.examples:
            assert not ex.succeeded
            assert ex.run is None
            assert ex.kpi_report is None
            assert ex.error is not None
            assert "Simulated failure" in ex.error

    def test_mixed_success_and_failure(self) -> None:
        runner = ExperimentRunner(client=_MixedClient(fail_on="Question 1?"))
        result = runner.run(qa_examples=_make_examples(3), config=_make_config())

        assert result.num_examples == 3
        assert result.num_successful == 2
        assert result.num_failed == 1
        failed = [e for e in result.examples if not e.succeeded]
        assert len(failed) == 1
        assert failed[0].example_id == "ex1"

    def test_counts_helpers(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(4), config=_make_config())

        assert result.num_examples == 4
        assert result.num_successful == 4
        assert result.num_failed == 0
        assert result.metadata["num_examples"] == 4
        assert result.metadata["num_successful"] == 4
        assert result.metadata["num_failed"] == 0

    def test_db_kpis_not_computed_by_default(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(2), config=_make_config())

        assert result.db_kpi_report is None

    def test_db_kpis_computed_when_requested(self) -> None:
        runner = ExperimentRunner(
            client=_MockClient(), db_kpi_engine=_FakeDBEngine()
        )
        result = runner.run(
            qa_examples=_make_examples(2),
            config=_make_config(),
            compute_db_kpis=True,
        )

        assert result.db_kpi_report is not None
        assert result.db_kpi_report.results[0].name == "db_document_count"
        assert result.db_kpi_report.results[0].value == 42


class _FakeSemanticProvider(BaseSemanticEmbeddingProvider):
    """Deterministic provider: every text maps to the same unit vector.

    With identical vectors all cosine similarities equal 1.0, which keeps the
    expected semantic KPI values simple and stable.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[1.0, 0.0] for _ in texts]


_SEMANTIC_KPI_NAMES = {
    str(KPIId.QUERY_CONTEXT_SIMILARITY_MEAN),
    str(KPIId.QUERY_CONTEXT_SIMILARITY_MAX),
    str(KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN),
    str(KPIId.ANSWER_CONTEXT_SIMILARITY_MAX),
    str(KPIId.ANSWER_QUERY_SIMILARITY),
}


class TestExperimentRunnerSemanticKPIs:
    def test_semantic_disabled_by_default_has_no_semantic_kpis(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(2), config=_make_config())

        for ex in result.examples:
            assert ex.kpi_report is not None
            names = {r.name for r in ex.kpi_report.results}
            assert names.isdisjoint(_SEMANTIC_KPI_NAMES)

    def test_enabling_without_engine_raises(self) -> None:
        runner = ExperimentRunner(client=_MockClient())
        try:
            runner.run(
                qa_examples=_make_examples(1),
                config=_make_config(),
                compute_semantic_kpis=True,
            )
        except ValueError as exc:
            assert "semantic_kpi_engine" in str(exc)
        else:  # pragma: no cover - explicit failure if no error raised
            raise AssertionError("Expected ValueError when no semantic engine")

    def test_merges_semantic_kpis_when_enabled(self) -> None:
        engine = SemanticKPIEngine(provider=_FakeSemanticProvider())
        runner = ExperimentRunner(
            client=_MockClient(), semantic_kpi_engine=engine
        )
        result = runner.run(
            qa_examples=_make_examples(2),
            config=_make_config(),
            compute_semantic_kpis=True,
        )

        for ex in result.examples:
            assert ex.kpi_report is not None
            names = {r.name for r in ex.kpi_report.results}
            # Default deterministic KPIs are still present alongside semantic.
            assert str(KPIId.TOTAL_LATENCY_MS) in names
            assert _SEMANTIC_KPI_NAMES.issubset(names)
            values = {
                r.name: r.value for r in ex.kpi_report.results
            }
            assert values[str(KPIId.QUERY_CONTEXT_SIMILARITY_MEAN)] == 1.0
            assert values[str(KPIId.ANSWER_QUERY_SIMILARITY)] == 1.0

    def test_semantic_not_computed_for_failed_runs(self) -> None:
        provider = _FakeSemanticProvider()
        engine = SemanticKPIEngine(provider=provider)
        runner = ExperimentRunner(
            client=_FailingClient(), semantic_kpi_engine=engine
        )
        result = runner.run(
            qa_examples=_make_examples(3),
            config=_make_config(),
            compute_semantic_kpis=True,
        )

        assert result.num_successful == 0
        assert result.num_failed == 3
        # The semantic provider is never called for failed RAG runs.
        assert provider.call_count == 0
