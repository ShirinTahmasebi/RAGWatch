"""End-to-end integration tests for semantic KPIs in batch experiments.

These tests use a deterministic fake semantic provider and never require
Hugging Face, OpenAI, Azure, Postgres, Chroma, Qdrant, Streamlit, Plotly, or
API keys.
"""

import csv
from pathlib import Path

from ragwatch.core.interfaces import BaseRAGClient
from ragwatch.core.schema import (
    Document,
    GenerationResult,
    QAExample,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.experiments.exporters import export_experiment, export_kpis_csv
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import ExperimentConfig
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.semantic.providers import BaseSemanticEmbeddingProvider


class _MockClient(BaseRAGClient):
    """Returns a fixed RAGRun for any query."""

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id="d1", text="some context", metadata={"source": "src"}
                ),
                score=0.9,
                rank=1,
                retriever_name="mock",
            )
        ]
        return RAGRun(
            run_id="run-1",
            query=query,
            retrieved_documents=docs,
            generation=GenerationResult(answer="an answer", generator_name="mock"),
            latency_ms=1.0,
            metadata={
                "retrieval_latency_ms": 0.5,
                "generation_latency_ms": 0.5,
                "total_latency_ms": 1.0,
            },
        )


class _FakeSemanticProvider(BaseSemanticEmbeddingProvider):
    """Deterministic provider mapping every text to the same unit vector."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


_SEMANTIC_KPI_NAMES = [
    str(KPIId.QUERY_CONTEXT_SIMILARITY_MEAN),
    str(KPIId.QUERY_CONTEXT_SIMILARITY_MAX),
    str(KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN),
    str(KPIId.ANSWER_CONTEXT_SIMILARITY_MAX),
    str(KPIId.ANSWER_QUERY_SIMILARITY),
]


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="exp",
        dataset_name="ds",
        retriever_name="mock",
        generator_name="mock",
        top_k=3,
    )


def _make_examples(n: int = 2) -> list[QAExample]:
    return [
        QAExample(example_id=f"ex{i}", question=f"Q{i}?", answers=[f"a{i}"])
        for i in range(n)
    ]


def _run_with_semantic() -> "ExperimentRunner":
    engine = SemanticKPIEngine(provider=_FakeSemanticProvider())
    return ExperimentRunner(client=_MockClient(), semantic_kpi_engine=engine)


class TestSemanticKpisCsvExport:
    def test_kpis_csv_includes_semantic_columns_when_enabled(
        self, tmp_path: Path
    ) -> None:
        runner = _run_with_semantic()
        result = runner.run(
            qa_examples=_make_examples(2),
            config=_make_config(),
            compute_semantic_kpis=True,
        )
        out = tmp_path / "kpis.csv"
        export_kpis_csv(result, out)

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows
        for name in _SEMANTIC_KPI_NAMES:
            assert name in rows[0]
        assert rows[0][str(KPIId.ANSWER_QUERY_SIMILARITY)] == "1.0"

    def test_kpis_csv_has_no_semantic_columns_when_disabled(
        self, tmp_path: Path
    ) -> None:
        runner = ExperimentRunner(client=_MockClient())
        result = runner.run(qa_examples=_make_examples(2), config=_make_config())
        out = tmp_path / "kpis.csv"
        export_kpis_csv(result, out)

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows
        for name in _SEMANTIC_KPI_NAMES:
            assert name not in rows[0]

    def test_full_experiment_export_tree_with_semantic(
        self, tmp_path: Path
    ) -> None:
        runner = _run_with_semantic()
        result = runner.run(
            qa_examples=_make_examples(2),
            config=_make_config(),
            compute_semantic_kpis=True,
        )
        export_experiment(result, tmp_path)

        for name in ("runs.jsonl", "kpis.jsonl", "kpis.csv", "summary.json"):
            assert (tmp_path / name).exists()

        with (tmp_path / "kpis.csv").open() as f:
            header = next(csv.reader(f))
        assert str(KPIId.QUERY_CONTEXT_SIMILARITY_MEAN) in header
