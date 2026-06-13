"""Tests for the retriever comparison runner and exporters."""

import csv
import json
from pathlib import Path

from ragwatch.core.interfaces import BaseRAGClient
from ragwatch.core.schema import (
    Document,
    GenerationResult,
    QAExample,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.experiments.comparison import (
    ComparisonConfig,
    ComparisonRunner,
    RetrieverExperimentSpec,
    export_combined_kpis_csv,
    export_comparison,
    export_comparison_summary_csv,
    export_comparison_summary_json,
)
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIReport, KPIResult
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.semantic.providers import BaseSemanticEmbeddingProvider


class _MockClient(BaseRAGClient):
    """Returns a fixed RAGRun, tagged with a retriever name."""

    def __init__(self, retriever_name: str = "mock") -> None:
        self._retriever_name = retriever_name

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id="d1", text="doc text", metadata={"source": "src"}
                ),
                score=0.8,
                rank=1,
                retriever_name=self._retriever_name,
            )
        ]
        return RAGRun(
            run_id=f"{self._retriever_name}-run",
            query=query,
            retrieved_documents=docs,
            generation=GenerationResult(answer="mock answer", generator_name="mock"),
            latency_ms=2.0,
            metadata={
                "retrieval_latency_ms": 1.5,
                "generation_latency_ms": 0.5,
                "total_latency_ms": 2.0,
            },
        )


class _FailingClient(BaseRAGClient):
    """Raises on every run (per-example failures)."""

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        raise RuntimeError("boom")


def _make_examples(n: int = 3) -> list[QAExample]:
    return [
        QAExample(example_id=f"ex{i}", question=f"Q{i}?", answers=[f"a{i}"])
        for i in range(n)
    ]


def _make_config(top_k_values: list[int] | None = None) -> ComparisonConfig:
    return ComparisonConfig(
        comparison_name="test_comparison",
        dataset_name="test_ds",
        top_k_values=top_k_values if top_k_values is not None else [3, 5],
        max_examples=10,
    )


def _make_specs() -> list[RetrieverExperimentSpec]:
    return [
        RetrieverExperimentSpec(
            name="mock_a",
            client=_MockClient("mock_a"),
            retriever_name="mock_a",
            generator_name="heuristic",
        ),
        RetrieverExperimentSpec(
            name="mock_b",
            client=_MockClient("mock_b"),
            retriever_name="mock_b",
            generator_name="heuristic",
        ),
    ]


class TestComparisonRunner:
    def test_runs_multiple_specs_and_top_k(self) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(3), _make_specs(), _make_config([3, 5]))

        # 2 specs x 2 top_k = 4 experiment settings
        assert result.num_experiments == 4

    def test_experiment_keys_are_stable_and_readable(self) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), _make_specs(), _make_config([3, 5]))

        assert set(result.experiment_keys) == {
            "mock_a_top3",
            "mock_a_top5",
            "mock_b_top3",
            "mock_b_top5",
        }

    def test_one_experiment_result_per_setting(self) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(3), _make_specs(), _make_config([3]))

        assert len(result.experiment_results) == 2
        for exp in result.experiment_results.values():
            assert exp.num_examples == 3
            assert exp.num_successful == 3

    def test_top_k_propagates_to_experiment_config(self) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(1), _make_specs(), _make_config([3, 5]))

        assert result.experiment_results["mock_a_top3"].config.top_k == 3
        assert result.experiment_results["mock_a_top5"].config.top_k == 5

    def test_per_example_failure_does_not_crash_comparison(self) -> None:
        specs = [
            RetrieverExperimentSpec(
                name="good",
                client=_MockClient("good"),
                retriever_name="good",
                generator_name="heuristic",
            ),
            RetrieverExperimentSpec(
                name="bad",
                client=_FailingClient(),
                retriever_name="bad",
                generator_name="heuristic",
            ),
        ]
        runner = ComparisonRunner()
        result = runner.run(_make_examples(3), specs, _make_config([3]))

        assert result.num_experiments == 2
        assert result.experiment_results["good_top3"].num_successful == 3
        bad = result.experiment_results["bad_top3"]
        assert bad.num_successful == 0
        assert bad.num_failed == 3


class TestComparisonExporters:
    def test_export_combined_kpis_csv(self, tmp_path: Path) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), _make_specs(), _make_config([3, 5]))
        out = tmp_path / "combined_kpis.csv"
        export_combined_kpis_csv(result, out)

        assert out.exists()
        with out.open() as f:
            rows = list(csv.DictReader(f))
        # 4 settings x 2 examples = 8 rows
        assert len(rows) == 8
        first = rows[0]
        for col in (
            "comparison_name",
            "experiment_key",
            "dataset_name",
            "retriever_name",
            "generator_name",
            "top_k",
            "example_id",
            "query",
            "run_id",
            "num_retrieved_documents",
            "total_latency_ms",
        ):
            assert col in first

    def test_export_comparison_summary_csv(self, tmp_path: Path) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(3), _make_specs(), _make_config([3, 5]))
        out = tmp_path / "comparison_summary.csv"
        export_comparison_summary_csv(result, out)

        assert out.exists()
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4  # one row per setting
        row = rows[0]
        assert row["num_examples"] == "3"
        assert row["num_successful"] == "3"
        assert "avg_total_latency_ms" in row
        assert "avg_retrieval_score_mean" in row

    def test_export_comparison_summary_json(self, tmp_path: Path) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), _make_specs(), _make_config([3]))
        out = tmp_path / "comparison_summary.json"
        export_comparison_summary_json(result, out)

        payload = json.loads(out.read_text())
        assert payload["comparison_name"] == "test_comparison"
        assert payload["top_k_values"] == [3]
        assert payload["num_experiments"] == 2
        assert "created_at" in payload
        assert len(payload["experiments"]) == 2
        exp = payload["experiments"][0]
        assert "experiment_key" in exp
        assert "aggregates" in exp

    def test_export_comparison_creates_full_tree(self, tmp_path: Path) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), _make_specs(), _make_config([3, 5]))
        export_comparison(result, tmp_path)

        assert (tmp_path / "combined_kpis.csv").exists()
        assert (tmp_path / "comparison_summary.csv").exists()
        assert (tmp_path / "comparison_summary.json").exists()

        experiments_dir = tmp_path / "experiments"
        for key in result.experiment_keys:
            exp_dir = experiments_dir / key
            for name in (
                "runs.jsonl",
                "kpis.jsonl",
                "kpis.csv",
                "db_kpis.json",
                "summary.json",
            ):
                assert (exp_dir / name).exists()


class _StubDBEngine:
    """Stand-in DB KPI engine that returns a fixed report without Postgres."""

    def __init__(self) -> None:
        self.compute_calls = 0

    def compute(self, connection_string: str | None = None) -> KPIReport:
        self.compute_calls += 1
        return KPIReport(
            run_id=None,
            query=None,
            results=[
                KPIResult(
                    name="db_document_count",
                    value=7,
                    category="db_health",
                    stage="database",
                    source="postgres",
                    description="stub",
                )
            ],
            metadata={"source": "database"},
        )


class TestComparisonDBKPIs:
    def test_spec_is_backward_compatible_without_db_fields(self) -> None:
        spec = RetrieverExperimentSpec(
            name="mock",
            client=_MockClient("mock"),
            retriever_name="mock",
            generator_name="heuristic",
        )
        assert spec.db_kpi_engine is None
        assert spec.compute_db_kpis is False

    def test_default_specs_produce_no_db_kpi_report(self) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), _make_specs(), _make_config([3]))
        for exp in result.experiment_results.values():
            assert exp.db_kpi_report is None

    def test_compute_db_kpis_passes_engine_and_populates_report(self) -> None:
        stub = _StubDBEngine()
        specs = [
            RetrieverExperimentSpec(
                name="pg",
                client=_MockClient("pg"),
                retriever_name="pgvector",
                generator_name="heuristic",
                db_kpi_engine=stub,  # type: ignore[arg-type]
                compute_db_kpis=True,
            )
        ]
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), specs, _make_config([3, 5]))

        # One DB snapshot per setting (2 top_k values).
        assert stub.compute_calls == 2
        for exp in result.experiment_results.values():
            assert exp.db_kpi_report is not None
            assert exp.db_kpi_report.results[0].name == "db_document_count"


class _FakeSemanticProvider(BaseSemanticEmbeddingProvider):
    """Deterministic provider mapping every text to the same unit vector."""

    def __init__(self) -> None:
        self.call_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        return [[1.0, 0.0] for _ in texts]


_SEMANTIC_KPI_NAMES = {
    str(KPIId.QUERY_CONTEXT_SIMILARITY_MEAN),
    str(KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN),
    str(KPIId.ANSWER_QUERY_SIMILARITY),
}


class TestComparisonSemanticKPIs:
    def test_spec_is_backward_compatible_without_semantic_fields(self) -> None:
        spec = RetrieverExperimentSpec(
            name="mock",
            client=_MockClient("mock"),
            retriever_name="mock",
            generator_name="heuristic",
        )
        assert spec.semantic_kpi_engine is None
        assert spec.compute_semantic_kpis is False

    def test_semantic_disabled_by_default_in_combined_csv(
        self, tmp_path: Path
    ) -> None:
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), _make_specs(), _make_config([3]))
        out = tmp_path / "combined_kpis.csv"
        export_combined_kpis_csv(result, out)

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows
        for name in _SEMANTIC_KPI_NAMES:
            assert name not in rows[0]

    def test_semantic_passed_through_per_spec(self, tmp_path: Path) -> None:
        provider = _FakeSemanticProvider()
        engine = SemanticKPIEngine(provider=provider)
        specs = [
            RetrieverExperimentSpec(
                name="sem",
                client=_MockClient("sem"),
                retriever_name="sem",
                generator_name="heuristic",
                semantic_kpi_engine=engine,
                compute_semantic_kpis=True,
            ),
            RetrieverExperimentSpec(
                name="plain",
                client=_MockClient("plain"),
                retriever_name="plain",
                generator_name="heuristic",
            ),
        ]
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), specs, _make_config([3]))

        sem_exp = result.experiment_results["sem_top3"]
        for ex in sem_exp.examples:
            assert ex.kpi_report is not None
            names = {r.name for r in ex.kpi_report.results}
            assert _SEMANTIC_KPI_NAMES.issubset(names)

        plain_exp = result.experiment_results["plain_top3"]
        for ex in plain_exp.examples:
            assert ex.kpi_report is not None
            names = {r.name for r in ex.kpi_report.results}
            assert names.isdisjoint(_SEMANTIC_KPI_NAMES)

    def test_semantic_kpis_appear_in_combined_csv_when_enabled(
        self, tmp_path: Path
    ) -> None:
        engine = SemanticKPIEngine(provider=_FakeSemanticProvider())
        specs = [
            RetrieverExperimentSpec(
                name="sem",
                client=_MockClient("sem"),
                retriever_name="sem",
                generator_name="heuristic",
                semantic_kpi_engine=engine,
                compute_semantic_kpis=True,
            )
        ]
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), specs, _make_config([3]))
        out = tmp_path / "combined_kpis.csv"
        export_combined_kpis_csv(result, out)

        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert rows
        for name in _SEMANTIC_KPI_NAMES:
            assert name in rows[0]
        assert rows[0][str(KPIId.ANSWER_QUERY_SIMILARITY)] == "1.0"

    def test_semantic_averages_in_summary_when_enabled(
        self, tmp_path: Path
    ) -> None:
        engine = SemanticKPIEngine(provider=_FakeSemanticProvider())
        specs = [
            RetrieverExperimentSpec(
                name="sem",
                client=_MockClient("sem"),
                retriever_name="sem",
                generator_name="heuristic",
                semantic_kpi_engine=engine,
                compute_semantic_kpis=True,
            )
        ]
        runner = ComparisonRunner()
        result = runner.run(_make_examples(3), specs, _make_config([3]))
        out = tmp_path / "comparison_summary.csv"
        export_comparison_summary_csv(result, out)

        with out.open() as f:
            rows = list(csv.DictReader(f))
        row = rows[0]
        assert row[f"avg_{KPIId.QUERY_CONTEXT_SIMILARITY_MEAN}"] == "1.0"
        assert row[f"avg_{KPIId.ANSWER_QUERY_SIMILARITY}"] == "1.0"


class TestComparisonDBKPIsExport:
    def test_pgvector_like_spec_writes_non_null_db_kpis_json(
        self, tmp_path: Path
    ) -> None:
        specs = [
            RetrieverExperimentSpec(
                name="pg",
                client=_MockClient("pg"),
                retriever_name="pgvector",
                generator_name="heuristic",
                db_kpi_engine=_StubDBEngine(),  # type: ignore[arg-type]
                compute_db_kpis=True,
            ),
            RetrieverExperimentSpec(
                name="tfidf",
                client=_MockClient("tfidf"),
                retriever_name="tfidf",
                generator_name="heuristic",
            ),
        ]
        runner = ComparisonRunner()
        result = runner.run(_make_examples(2), specs, _make_config([3]))
        export_comparison(result, tmp_path)

        # pgvector report is non-null: serialized report has a results list.
        pg_payload = json.loads(
            (tmp_path / "experiments" / "pg_top3" / "db_kpis.json").read_text()
        )
        assert "results" in pg_payload

        # Non-pgvector retrievers keep a null db_kpis.json.
        tfidf_payload = json.loads(
            (tmp_path / "experiments" / "tfidf_top3" / "db_kpis.json").read_text()
        )
        assert tfidf_payload == {"db_kpi_report": None}
