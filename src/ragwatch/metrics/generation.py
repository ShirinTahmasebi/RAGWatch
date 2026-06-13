"""Deterministic generation metrics computed from RAGRun."""

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.base import BaseMetric
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.schema import KPIResult


class AnswerLengthCharsMetric(BaseMetric):
    """Character length of the generated answer."""

    kpi_id = KPIId.ANSWER_LENGTH_CHARS

    def compute(self, run: RAGRun) -> KPIResult:
        return KPIResult(
            name=self.name,
            value=len(run.generation.answer),
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class AnswerLengthWordsMetric(BaseMetric):
    """Word count of the generated answer."""

    kpi_id = KPIId.ANSWER_LENGTH_WORDS

    def compute(self, run: RAGRun) -> KPIResult:
        words = run.generation.answer.split()
        return KPIResult(
            name=self.name,
            value=len(words),
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


class AnswerToContextLengthRatioMetric(BaseMetric):
    """Ratio of answer length to total retrieved context length."""

    kpi_id = KPIId.ANSWER_TO_CONTEXT_LENGTH_RATIO

    def compute(self, run: RAGRun) -> KPIResult:
        context_len = sum(len(rd.document.text) for rd in run.retrieved_documents)
        if context_len == 0:
            value = None
        else:
            value = len(run.generation.answer) / context_len
        return KPIResult(
            name=self.name,
            value=value,
            category=self.category,
            stage=self.stage,
            source=self.source,
            description=self.description,
        )


DEFAULT_GENERATION_METRICS: list[BaseMetric] = [
    AnswerLengthCharsMetric(),
    AnswerLengthWordsMetric(),
    AnswerToContextLengthRatioMetric(),
]
