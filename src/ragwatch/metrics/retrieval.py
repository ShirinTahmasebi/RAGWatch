"""Deterministic retrieval metrics computed from RAGRun."""

import math
from typing import Any

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.base import BaseMetric
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIResult


class NumRetrievedDocumentsMetric(BaseMetric):
    """Count of retrieved documents."""

    kpi_id = KPIId.NUM_RETRIEVED_DOCUMENTS

    def compute(self, run: RAGRun) -> KPIResult:
        return KPIResult(
            name=self.name,
            value=len(run.retrieved_documents),
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalScoreMinMetric(BaseMetric):
    """Minimum retrieval score across retrieved documents."""

    kpi_id = KPIId.RETRIEVAL_SCORE_MIN

    def compute(self, run: RAGRun) -> KPIResult:
        scores = [rd.score for rd in run.retrieved_documents]
        value = min(scores) if scores else None
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalScoreMaxMetric(BaseMetric):
    """Maximum retrieval score across retrieved documents."""

    kpi_id = KPIId.RETRIEVAL_SCORE_MAX

    def compute(self, run: RAGRun) -> KPIResult:
        scores = [rd.score for rd in run.retrieved_documents]
        value = max(scores) if scores else None
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalScoreMeanMetric(BaseMetric):
    """Mean retrieval score across retrieved documents."""

    kpi_id = KPIId.RETRIEVAL_SCORE_MEAN

    def compute(self, run: RAGRun) -> KPIResult:
        scores = [rd.score for rd in run.retrieved_documents]
        value = sum(scores) / len(scores) if scores else None
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalScoreStdMetric(BaseMetric):
    """Population standard deviation of retrieval scores."""

    kpi_id = KPIId.RETRIEVAL_SCORE_STD

    def compute(self, run: RAGRun) -> KPIResult:
        scores = [rd.score for rd in run.retrieved_documents]
        if not scores:
            value = None
        else:
            mean = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / len(scores)
            value = math.sqrt(variance)
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalScoreRangeMetric(BaseMetric):
    """Range (max - min) of retrieval scores."""

    kpi_id = KPIId.RETRIEVAL_SCORE_RANGE

    def compute(self, run: RAGRun) -> KPIResult:
        scores = [rd.score for rd in run.retrieved_documents]
        value = max(scores) - min(scores) if scores else None
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class RetrievalScoreGapTop1Top2Metric(BaseMetric):
    """Gap between the top-1 and top-2 retrieval scores."""

    kpi_id = KPIId.RETRIEVAL_SCORE_GAP_TOP1_TOP2

    def compute(self, run: RAGRun) -> KPIResult:
        docs = run.retrieved_documents
        if len(docs) < 2:
            value = None
        else:
            sorted_docs = sorted(docs, key=lambda d: d.rank)
            value = sorted_docs[0].score - sorted_docs[1].score
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class ContextLengthCharsMetric(BaseMetric):
    """Total character length of all retrieved document texts."""

    kpi_id = KPIId.CONTEXT_LENGTH_CHARS

    def compute(self, run: RAGRun) -> KPIResult:
        total = sum(len(rd.document.text) for rd in run.retrieved_documents)
        return KPIResult(
            name=self.name,
            value=total,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class UniqueRetrievedSourcesMetric(BaseMetric):
    """Count of unique sources among retrieved documents."""

    kpi_id = KPIId.UNIQUE_RETRIEVED_SOURCES

    def compute(self, run: RAGRun) -> KPIResult:
        sources: set[str] = set()
        for rd in run.retrieved_documents:
            source_val = rd.document.metadata.get("source")
            if source_val is not None:
                sources.add(str(source_val))
            else:
                sources.add(rd.document.doc_id)
        return KPIResult(
            name=self.name,
            value=len(sources),
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


def _word_shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    """Extract normalized word n-gram shingles from text."""
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def _jaccard_similarity(a: set[Any], b: set[Any]) -> float:
    """Compute Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0


class RetrievalRedundancyMetric(BaseMetric):
    """Lexical redundancy among retrieved documents via shingle Jaccard similarity."""

    kpi_id = KPIId.RETRIEVAL_REDUNDANCY

    def __init__(self, ngram_size: int = 3) -> None:
        self._ngram_size = ngram_size

    def compute(self, run: RAGRun) -> KPIResult:
        docs = run.retrieved_documents
        n = len(docs)
        if n < 2:
            return KPIResult(
                name=self.name,
                value=0.0,
                category=self.category,
                stage=self.stage,
                source=self.source,
                description=self.description,
                metadata={
                    "num_documents": n,
                    "ngram_size": self._ngram_size,
                    "num_pairs": 0,
                },
            )

        shingle_sets = [
            _word_shingles(rd.document.text, self._ngram_size) for rd in docs
        ]

        total_sim = 0.0
        num_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += _jaccard_similarity(shingle_sets[i], shingle_sets[j])
                num_pairs += 1

        mean_sim = total_sim / num_pairs if num_pairs > 0 else 0.0

        return KPIResult(
            name=self.name,
            value=mean_sim,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
            metadata={
                "num_documents": n,
                "ngram_size": self._ngram_size,
                "num_pairs": num_pairs,
            },
        )


DEFAULT_RETRIEVAL_METRICS: list[BaseMetric] = [
    NumRetrievedDocumentsMetric(),
    RetrievalScoreMinMetric(),
    RetrievalScoreMaxMetric(),
    RetrievalScoreMeanMetric(),
    RetrievalScoreStdMetric(),
    RetrievalScoreRangeMetric(),
    RetrievalScoreGapTop1Top2Metric(),
    ContextLengthCharsMetric(),
    UniqueRetrievedSourcesMetric(),
    RetrievalRedundancyMetric(),
]
