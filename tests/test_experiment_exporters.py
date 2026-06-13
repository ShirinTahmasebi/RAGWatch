"""Tests for experiment exporters."""

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
from ragwatch.experiments.exporters import (
    export_db_kpis_json,
    export_experiment,
    export_experiment_summary_json,
    export_kpis_csv,
    export_kpis_jsonl,
    export_runs_jsonl,
)
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import (
    ExperimentConfig,
    ExperimentResult,
)
from ragwatch.metrics.schema import KPIReport, KPIResult


class _MockClient(BaseRAGClient):
    def run(self, query: str, top_k: int = 5) -> RAGRun:
        docs = [
            RetrievedDocument(
                document=Document(
                    doc_id="d1", text="doc text here", metadata={"source": "src_a"}
                ),
                score=0.85,
                rank=1,
                retriever_name="mock",
            ),
            RetrievedDocument(
                document=Document(
                    doc_id="d2", text="another doc", metadata={"source": "src_b"}
                ),
                score=0.60,
                rank=2,
                retriever_name="mock",
            ),
        ]
        return RAGRun(
            run_id="test-run-id",
            query=query,
            retrieved_documents=docs,
            generation=GenerationResult(answer="test answer", generator_name="mock"),
            latency_ms=5.0,
            metadata={
                "retrieval_latency_ms": 3.0,
                "generation_latency_ms": 2.0,
                "total_latency_ms": 5.0,
            },
        )


class _FailOnClient(BaseRAGClient):
    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on
        self._good = _MockClient()

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        if query == self._fail_on:
            raise RuntimeError("boom")
        return self._good.run(query, top_k=top_k)


def _make_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test_exp",
        dataset_name="test_ds",
        retriever_name="mock",
        generator_name="mock",
        top_k=2,
    )


def _make_result(n: int = 3, client: BaseRAGClient | None = None) -> ExperimentResult:
    examples = [
        QAExample(example_id=f"ex{i}", question=f"Q{i}?", answers=[f"a{i}"])
        for i in range(n)
    ]
    runner = ExperimentRunner(client=client or _MockClient())
    return runner.run(qa_examples=examples, config=_make_config())


class TestExportRunsJsonl:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "runs.jsonl"
        export_runs_jsonl(_make_result(3), out)
        assert out.exists()

    def test_one_line_per_example(self, tmp_path: Path) -> None:
        out = tmp_path / "runs.jsonl"
        export_runs_jsonl(_make_result(3), out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_line_structure_for_success(self, tmp_path: Path) -> None:
        out = tmp_path / "runs.jsonl"
        export_runs_jsonl(_make_result(1), out)
        record = json.loads(out.read_text().strip())
        for key in (
            "experiment_name",
            "dataset_name",
            "retriever_name",
            "generator_name",
            "top_k",
            "example_id",
            "question",
            "gold_answers",
            "run_id",
            "answer",
            "retrieved_doc_ids",
            "retrieved_scores",
            "retrieved_ranks",
            "retrieved_sources",
            "latency_ms",
            "metadata",
        ):
            assert key in record
        assert record["retrieved_doc_ids"] == ["d1", "d2"]
        assert record["retrieved_ranks"] == [1, 2]
        assert record["retrieved_sources"] == ["src_a", "src_b"]

    def test_failed_example_record(self, tmp_path: Path) -> None:
        result = _make_result(2, client=_FailOnClient(fail_on="Q1?"))
        out = tmp_path / "runs.jsonl"
        export_runs_jsonl(result, out)
        lines = out.read_text().strip().split("\n")
        records = [json.loads(line) for line in lines]
        failed = [r for r in records if r["example_id"] == "ex1"][0]
        assert "error" in failed
        assert "answer" not in failed

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "dir" / "runs.jsonl"
        export_runs_jsonl(_make_result(1), out)
        assert out.exists()


class TestExportKpisJsonl:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "kpis.jsonl"
        export_kpis_jsonl(_make_result(2), out)
        assert out.exists()

    def test_one_line_per_successful_example(self, tmp_path: Path) -> None:
        result = _make_result(3, client=_FailOnClient(fail_on="Q1?"))
        out = tmp_path / "kpis.jsonl"
        export_kpis_jsonl(result, out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2  # one failed example excluded

    def test_contains_experiment_and_kpi_columns(self, tmp_path: Path) -> None:
        out = tmp_path / "kpis.jsonl"
        export_kpis_jsonl(_make_result(1), out)
        record = json.loads(out.read_text().strip())
        assert record["experiment_name"] == "test_exp"
        assert record["dataset_name"] == "test_ds"
        assert record["example_id"] == "ex0"
        assert "num_retrieved_documents" in record
        assert "total_latency_ms" in record


class TestExportKpisCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        out = tmp_path / "kpis.csv"
        export_kpis_csv(_make_result(3), out)
        assert out.exists()

    def test_has_header_and_data_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "kpis.csv"
        export_kpis_csv(_make_result(3), out)
        with out.open() as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4  # 1 header + 3 data rows

    def test_has_expected_columns(self, tmp_path: Path) -> None:
        out = tmp_path / "kpis.csv"
        export_kpis_csv(_make_result(1), out)
        with out.open() as f:
            row = next(csv.DictReader(f))
        assert row["experiment_name"] == "test_exp"
        assert row["dataset_name"] == "test_ds"
        assert row["retriever_name"] == "mock"
        assert row["generator_name"] == "mock"
        assert row["top_k"] == "2"
        assert row["example_id"] == "ex0"
        assert "run_id" in row
        assert "query" in row
        assert "num_retrieved_documents" in row
        assert "retrieval_score_mean" in row
        assert "total_latency_ms" in row

    def test_only_successful_rows(self, tmp_path: Path) -> None:
        result = _make_result(3, client=_FailOnClient(fail_on="Q0?"))
        out = tmp_path / "kpis.csv"
        export_kpis_csv(result, out)
        with out.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2

    def test_empty_result_creates_empty_file(self, tmp_path: Path) -> None:
        config = ExperimentConfig(
            experiment_name="empty",
            dataset_name="empty",
            retriever_name="none",
            generator_name="none",
        )
        result = ExperimentResult(config=config)
        out = tmp_path / "empty.csv"
        export_kpis_csv(result, out)
        assert out.exists()
        assert out.read_text() == ""


def _db_report() -> KPIReport:
    return KPIReport(
        run_id=None,
        query=None,
        results=[
            KPIResult(
                name="db_document_count",
                value=10,
                category="db_index_stats",
                stage="database",
                source="postgres",
                description="docs",
            )
        ],
        metadata={"source": "database"},
    )


class TestExportDbKpisJson:
    def test_writes_null_when_absent(self, tmp_path: Path) -> None:
        out = tmp_path / "db_kpis.json"
        export_db_kpis_json(_make_result(1), out)
        payload = json.loads(out.read_text())
        assert payload == {"db_kpi_report": None}

    def test_writes_report_when_present(self, tmp_path: Path) -> None:
        result = _make_result(1)
        result.db_kpi_report = _db_report()
        out = tmp_path / "db_kpis.json"
        export_db_kpis_json(result, out)
        payload = json.loads(out.read_text())
        assert payload["results"][0]["name"] == "db_document_count"
        assert payload["results"][0]["value"] == 10


class TestExportSummaryJson:
    def test_contains_counts_and_aggregates(self, tmp_path: Path) -> None:
        result = _make_result(3)
        out = tmp_path / "summary.json"
        export_experiment_summary_json(result, out)
        summary = json.loads(out.read_text())
        assert summary["num_examples"] == 3
        assert summary["num_successful"] == 3
        assert summary["num_failed"] == 0
        assert "created_at" in summary
        assert summary["config"]["experiment_name"] == "test_exp"
        assert "avg_total_latency_ms" in summary["aggregate_metrics"]


class TestExportExperiment:
    def test_creates_all_files(self, tmp_path: Path) -> None:
        result = _make_result(3)
        export_experiment(result, tmp_path)
        for name in (
            "runs.jsonl",
            "kpis.jsonl",
            "kpis.csv",
            "db_kpis.json",
            "summary.json",
        ):
            assert (tmp_path / name).exists()

    def test_db_kpis_file_with_report(self, tmp_path: Path) -> None:
        result = _make_result(2)
        result.db_kpi_report = _db_report()
        export_experiment(result, tmp_path)
        payload = json.loads((tmp_path / "db_kpis.json").read_text())
        assert payload["results"][0]["name"] == "db_document_count"
