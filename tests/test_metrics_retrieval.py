"""Tests for retrieval metrics."""

import math

from ragwatch.core.schema import (
    Document,
    GenerationResult,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.metrics.retrieval import (
    ContextLengthCharsMetric,
    NumRetrievedDocumentsMetric,
    RetrievalRedundancyMetric,
    RetrievalScoreGapTop1Top2Metric,
    RetrievalScoreMaxMetric,
    RetrievalScoreMeanMetric,
    RetrievalScoreMinMetric,
    RetrievalScoreRangeMetric,
    RetrievalScoreStdMetric,
    UniqueRetrievedSourcesMetric,
)


def _make_run(
    docs: list[RetrievedDocument] | None = None,
) -> RAGRun:
    if docs is None:
        docs = []
    return RAGRun(
        run_id="test-run",
        query="test query",
        retrieved_documents=docs,
        generation=GenerationResult(answer="test answer", generator_name="test"),
        latency_ms=10.0,
    )


def _make_doc(
    doc_id: str,
    text: str,
    score: float,
    rank: int,
    metadata: dict | None = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(doc_id=doc_id, text=text, metadata=metadata or {}),
        score=score,
        rank=rank,
        retriever_name="test",
    )


# --- NumRetrievedDocumentsMetric ---


class TestNumRetrievedDocuments:
    def test_no_documents(self) -> None:
        run = _make_run([])
        result = NumRetrievedDocumentsMetric().compute(run)
        assert result.value == 0

    def test_multiple_documents(self) -> None:
        docs = [
            _make_doc("d1", "hello world", 0.9, 1),
            _make_doc("d2", "foo bar", 0.7, 2),
            _make_doc("d3", "baz qux", 0.5, 3),
        ]
        run = _make_run(docs)
        result = NumRetrievedDocumentsMetric().compute(run)
        assert result.value == 3


# --- Score metrics ---


class TestRetrievalScoreMetrics:
    def test_no_documents_returns_none(self) -> None:
        run = _make_run([])
        assert RetrievalScoreMinMetric().compute(run).value is None
        assert RetrievalScoreMaxMetric().compute(run).value is None
        assert RetrievalScoreMeanMetric().compute(run).value is None
        assert RetrievalScoreStdMetric().compute(run).value is None
        assert RetrievalScoreRangeMetric().compute(run).value is None

    def test_single_document(self) -> None:
        docs = [_make_doc("d1", "hello", 0.8, 1)]
        run = _make_run(docs)
        assert RetrievalScoreMinMetric().compute(run).value == 0.8
        assert RetrievalScoreMaxMetric().compute(run).value == 0.8
        assert RetrievalScoreMeanMetric().compute(run).value == 0.8
        assert RetrievalScoreStdMetric().compute(run).value == 0.0
        assert RetrievalScoreRangeMetric().compute(run).value == 0.0

    def test_multiple_documents(self) -> None:
        docs = [
            _make_doc("d1", "a", 0.9, 1),
            _make_doc("d2", "b", 0.7, 2),
            _make_doc("d3", "c", 0.5, 3),
        ]
        run = _make_run(docs)
        assert RetrievalScoreMinMetric().compute(run).value == 0.5
        assert RetrievalScoreMaxMetric().compute(run).value == 0.9
        mean = RetrievalScoreMeanMetric().compute(run).value
        assert abs(mean - 0.7) < 1e-9
        assert RetrievalScoreRangeMetric().compute(run).value == pytest.approx(0.4)

        # Population std
        expected_std = math.sqrt(((0.2**2) + 0 + (0.2**2)) / 3)
        assert RetrievalScoreStdMetric().compute(run).value == pytest.approx(
            expected_std, abs=1e-9
        )


# --- Gap metric ---


class TestScoreGap:
    def test_no_documents_returns_none(self) -> None:
        run = _make_run([])
        assert RetrievalScoreGapTop1Top2Metric().compute(run).value is None

    def test_single_document_returns_none(self) -> None:
        docs = [_make_doc("d1", "hello", 0.9, 1)]
        run = _make_run(docs)
        assert RetrievalScoreGapTop1Top2Metric().compute(run).value is None

    def test_two_documents(self) -> None:
        docs = [
            _make_doc("d1", "a", 0.9, 1),
            _make_doc("d2", "b", 0.6, 2),
        ]
        run = _make_run(docs)
        result = RetrievalScoreGapTop1Top2Metric().compute(run)
        assert result.value == pytest.approx(0.3)

    def test_uses_rank_not_list_order(self) -> None:
        # Documents in wrong order by list position but correct by rank
        docs = [
            _make_doc("d2", "b", 0.6, 2),
            _make_doc("d1", "a", 0.9, 1),
        ]
        run = _make_run(docs)
        result = RetrievalScoreGapTop1Top2Metric().compute(run)
        assert result.value == pytest.approx(0.3)


# --- Context length ---


class TestContextLength:
    def test_no_documents(self) -> None:
        run = _make_run([])
        assert ContextLengthCharsMetric().compute(run).value == 0

    def test_sums_all_document_lengths(self) -> None:
        docs = [
            _make_doc("d1", "hello", 0.9, 1),  # 5 chars
            _make_doc("d2", "world!!", 0.7, 2),  # 7 chars
        ]
        run = _make_run(docs)
        assert ContextLengthCharsMetric().compute(run).value == 12


# --- Unique sources ---


class TestUniqueSources:
    def test_no_documents(self) -> None:
        run = _make_run([])
        assert UniqueRetrievedSourcesMetric().compute(run).value == 0

    def test_uses_source_metadata(self) -> None:
        docs = [
            _make_doc("d1", "a", 0.9, 1, metadata={"source": "wiki"}),
            _make_doc("d2", "b", 0.8, 2, metadata={"source": "wiki"}),
            _make_doc("d3", "c", 0.7, 3, metadata={"source": "arxiv"}),
        ]
        run = _make_run(docs)
        assert UniqueRetrievedSourcesMetric().compute(run).value == 2

    def test_falls_back_to_doc_id(self) -> None:
        docs = [
            _make_doc("d1", "a", 0.9, 1),
            _make_doc("d2", "b", 0.8, 2),
        ]
        run = _make_run(docs)
        assert UniqueRetrievedSourcesMetric().compute(run).value == 2


# --- Redundancy ---


class TestRedundancy:
    def test_single_document_returns_zero(self) -> None:
        docs = [_make_doc("d1", "hello world foo bar", 0.9, 1)]
        run = _make_run(docs)
        result = RetrievalRedundancyMetric().compute(run)
        assert result.value == 0.0
        assert result.metadata["num_pairs"] == 0

    def test_identical_documents_high_redundancy(self) -> None:
        text = "the quick brown fox jumps over the lazy dog"
        docs = [
            _make_doc("d1", text, 0.9, 1),
            _make_doc("d2", text, 0.8, 2),
        ]
        run = _make_run(docs)
        result = RetrievalRedundancyMetric().compute(run)
        assert result.value == 1.0
        assert result.metadata["num_pairs"] == 1

    def test_unrelated_documents_low_redundancy(self) -> None:
        docs = [
            _make_doc("d1", "the quick brown fox jumps over the lazy dog", 0.9, 1),
            _make_doc(
                "d2",
                "quantum computing uses superposition and entanglement principles",
                0.8,
                2,
            ),
        ]
        run = _make_run(docs)
        result = RetrievalRedundancyMetric().compute(run)
        assert result.value < 0.1

    def test_metadata_includes_info(self) -> None:
        docs = [
            _make_doc("d1", "foo bar baz", 0.9, 1),
            _make_doc("d2", "baz qux quux", 0.8, 2),
            _make_doc("d3", "alpha beta gamma", 0.7, 3),
        ]
        run = _make_run(docs)
        result = RetrievalRedundancyMetric().compute(run)
        assert result.metadata["num_documents"] == 3
        assert result.metadata["ngram_size"] == 3
        assert result.metadata["num_pairs"] == 3


import pytest
