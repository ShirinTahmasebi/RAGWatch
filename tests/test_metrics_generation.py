"""Tests for generation metrics."""

from ragwatch.core.schema import (
    Document,
    GenerationResult,
    RAGRun,
    RetrievedDocument,
)
from ragwatch.metrics.generation import (
    AnswerLengthCharsMetric,
    AnswerLengthWordsMetric,
    AnswerToContextLengthRatioMetric,
)


def _make_run(
    answer: str,
    docs: list[RetrievedDocument] | None = None,
) -> RAGRun:
    if docs is None:
        docs = []
    return RAGRun(
        run_id="test-run",
        query="test query",
        retrieved_documents=docs,
        generation=GenerationResult(answer=answer, generator_name="test"),
        latency_ms=10.0,
    )


def _make_doc(doc_id: str, text: str) -> RetrievedDocument:
    return RetrievedDocument(
        document=Document(doc_id=doc_id, text=text),
        score=0.9,
        rank=1,
        retriever_name="test",
    )


class TestAnswerLengthChars:
    def test_empty_answer(self) -> None:
        run = _make_run("")
        assert AnswerLengthCharsMetric().compute(run).value == 0

    def test_normal_answer(self) -> None:
        run = _make_run("Paris is the capital")
        assert AnswerLengthCharsMetric().compute(run).value == 20


class TestAnswerLengthWords:
    def test_empty_answer(self) -> None:
        run = _make_run("")
        assert AnswerLengthWordsMetric().compute(run).value == 0

    def test_normal_answer(self) -> None:
        run = _make_run("Paris is the capital of France")
        assert AnswerLengthWordsMetric().compute(run).value == 6


class TestAnswerToContextRatio:
    def test_no_context_returns_none(self) -> None:
        run = _make_run("some answer", docs=[])
        result = AnswerToContextLengthRatioMetric().compute(run)
        assert result.value is None

    def test_normal_ratio(self) -> None:
        # answer = 10 chars, context = 20 chars -> ratio = 0.5
        docs = [_make_doc("d1", "a" * 20)]
        run = _make_run("b" * 10, docs=docs)
        result = AnswerToContextLengthRatioMetric().compute(run)
        assert result.value == 0.5

    def test_multiple_documents(self) -> None:
        # answer = 5 chars, context = 10 + 10 = 20 chars -> ratio = 0.25
        docs = [
            _make_doc("d1", "a" * 10),
            _make_doc("d2", "b" * 10),
        ]
        run = _make_run("c" * 5, docs=docs)
        result = AnswerToContextLengthRatioMetric().compute(run)
        assert result.value == 0.25
