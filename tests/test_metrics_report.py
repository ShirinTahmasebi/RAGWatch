"""Tests for KPIEngine and report utilities."""

import pytest

from ragwatch.core.schema import (
    Document,
    GenerationResult,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.metrics.report import KPIEngine, to_dict, to_flat_dict
from ragwatch.metrics.retrieval import NumRetrievedDocumentsMetric


def _make_run(num_docs: int = 3) -> RAGRun:
    docs = [
        RetrievedDocument(
            document=Document(doc_id=f"d{i}", text=f"Document {i} text content"),
            score=0.9 - i * 0.1,
            rank=i + 1,
            retriever_name="test",
        )
        for i in range(num_docs)
    ]
    return RAGRun(
        run_id="run-123",
        query="What is the capital of France?",
        retrieved_documents=docs,
        generation=GenerationResult(
            answer="Paris is the capital of France.",
            generator_name="heuristic",
        ),
        latency_ms=42.5,
        metadata={"retrieval_latency_ms": 20.0, "generation_latency_ms": 22.5},
    )


class TestKPIEngine:
    def test_default_metrics_returns_all_expected_names(self) -> None:
        engine = KPIEngine()
        run = _make_run()
        report = engine.compute(run)

        names = [r.name for r in report.results]
        expected = [
            "num_retrieved_documents",
            "retrieval_score_min",
            "retrieval_score_max",
            "retrieval_score_mean",
            "retrieval_score_std",
            "retrieval_score_range",
            "retrieval_score_gap_top1_top2",
            "context_length_chars",
            "unique_retrieved_sources",
            "retrieval_redundancy",
            "answer_length_chars",
            "answer_length_words",
            "answer_to_context_length_ratio",
            "total_latency_ms",
            "retrieval_latency_ms",
            "generation_latency_ms",
        ]
        assert names == expected

    def test_custom_metrics(self) -> None:
        engine = KPIEngine(metrics=[NumRetrievedDocumentsMetric()])
        run = _make_run()
        report = engine.compute(run)
        assert len(report.results) == 1
        assert report.results[0].name == "num_retrieved_documents"

    def test_report_has_run_id_and_query(self) -> None:
        engine = KPIEngine()
        run = _make_run()
        report = engine.compute(run)
        assert report.run_id == "run-123"
        assert report.query == "What is the capital of France?"

    def test_runtime_latency_from_metadata(self) -> None:
        engine = KPIEngine()
        run = _make_run()
        report = engine.compute(run)
        flat = to_flat_dict(report)
        assert flat["total_latency_ms"] == 42.5
        assert flat["retrieval_latency_ms"] == 20.0
        assert flat["generation_latency_ms"] == 22.5

    def test_runtime_latency_missing_metadata(self) -> None:
        run = _make_run()
        run.metadata = {}
        engine = KPIEngine()
        report = engine.compute(run)
        flat = to_flat_dict(report)
        assert flat["retrieval_latency_ms"] is None
        assert flat["generation_latency_ms"] is None


class TestToDict:
    def test_structure(self) -> None:
        engine = KPIEngine(metrics=[NumRetrievedDocumentsMetric()])
        run = _make_run()
        report = engine.compute(run)
        d = to_dict(report)

        assert d["run_id"] == "run-123"
        assert d["query"] == "What is the capital of France?"
        assert len(d["results"]) == 1
        assert d["results"][0]["name"] == "num_retrieved_documents"
        assert d["results"][0]["value"] == 3
        assert d["results"][0]["category"] == "retrieval_quality"


class TestToFlatDict:
    def test_flat_mapping(self) -> None:
        engine = KPIEngine()
        run = _make_run()
        report = engine.compute(run)
        flat = to_flat_dict(report)

        assert flat["run_id"] == "run-123"
        assert flat["query"] == "What is the capital of France?"
        assert flat["num_retrieved_documents"] == 3
        assert "retrieval_score_mean" in flat
        assert "total_latency_ms" in flat

    def test_no_nested_dicts(self) -> None:
        engine = KPIEngine()
        run = _make_run()
        report = engine.compute(run)
        flat = to_flat_dict(report)

        for key, value in flat.items():
            assert not isinstance(value, dict), f"Key {key} has dict value"
            assert not isinstance(value, list), f"Key {key} has list value"
