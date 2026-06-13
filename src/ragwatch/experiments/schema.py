"""Data models for experiment configuration and results."""

from dataclasses import dataclass, field
from typing import Any

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.schema import KPIReport


@dataclass
class ExperimentConfig:
    """Configuration for a batch experiment run."""

    experiment_name: str
    dataset_name: str
    retriever_name: str
    generator_name: str
    top_k: int = 5
    max_examples: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentExampleResult:
    """Result for a single QA example in a batch experiment.

    For failed examples, ``run`` and ``kpi_report`` are ``None`` and
    ``error`` contains the error message.
    """

    example_id: str
    question: str
    gold_answers: list[str] = field(default_factory=list)
    run: RAGRun | None = None
    kpi_report: KPIReport | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """Whether this example produced a RAGRun without error."""
        return self.run is not None and self.error is None


@dataclass
class ExperimentResult:
    """Results from a completed batch experiment."""

    config: ExperimentConfig
    examples: list[ExperimentExampleResult] = field(default_factory=list)
    db_kpi_report: KPIReport | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_examples(self) -> int:
        """Total number of examples processed."""
        return len(self.examples)

    @property
    def num_successful(self) -> int:
        """Number of examples that produced a RAGRun and KPIReport."""
        return sum(1 for e in self.examples if e.succeeded)

    @property
    def num_failed(self) -> int:
        """Number of examples that failed."""
        return sum(1 for e in self.examples if not e.succeeded)
