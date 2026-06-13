"""Batch experiment runner for RAG pipelines."""

from __future__ import annotations

import traceback

from ragwatch.core.interfaces import BaseRAGClient
from ragwatch.core.schema import QAExample
from ragwatch.experiments.schema import (
    ExperimentConfig,
    ExperimentExampleResult,
    ExperimentResult,
)
from ragwatch.metrics.report import DBKPIEngine, KPIEngine
from ragwatch.metrics.schema import KPIReport


class ExperimentRunner:
    """Run a RAG client over multiple QA examples and compute KPIs."""

    def __init__(
        self,
        client: BaseRAGClient,
        kpi_engine: KPIEngine | None = None,
        db_kpi_engine: DBKPIEngine | None = None,
    ) -> None:
        self._client = client
        self._kpi_engine = kpi_engine if kpi_engine is not None else KPIEngine()
        self._db_kpi_engine = db_kpi_engine

    def run(
        self,
        qa_examples: list[QAExample],
        config: ExperimentConfig,
        limit: int | None = None,
        compute_db_kpis: bool = False,
    ) -> ExperimentResult:
        """Run the experiment over the given QA examples.

        Args:
            qa_examples: List of QA examples to run.
            config: Experiment configuration.
            limit: Optional limit on the number of examples to process.
                   Overrides ``config.max_examples`` if provided.
            compute_db_kpis: If True, compute a single DB KPI snapshot after
                   the runs using the configured (or a default) DBKPIEngine.

        Returns:
            ExperimentResult with one ExperimentExampleResult per example.
        """
        max_n = limit if limit is not None else config.max_examples
        examples = qa_examples[:max_n] if max_n is not None else qa_examples

        example_results: list[ExperimentExampleResult] = []
        for example in examples:
            example_results.append(self._run_one(example, config))

        db_kpi_report: KPIReport | None = None
        if compute_db_kpis:
            engine = self._db_kpi_engine or DBKPIEngine()
            db_kpi_report = engine.compute()

        return ExperimentResult(
            config=config,
            examples=example_results,
            db_kpi_report=db_kpi_report,
            metadata={
                "num_examples": len(example_results),
                "num_successful": sum(1 for e in example_results if e.succeeded),
                "num_failed": sum(1 for e in example_results if not e.succeeded),
            },
        )

    def _run_one(
        self, example: QAExample, config: ExperimentConfig
    ) -> ExperimentExampleResult:
        """Run a single QA example, capturing any error without raising."""
        try:
            run = self._client.run(query=example.question, top_k=config.top_k)
            # Enrich metadata with example info
            run.metadata["example_id"] = example.example_id
            run.metadata["gold_answers"] = example.answers
            run.metadata["question"] = example.question

            report = self._kpi_engine.compute(run)
            return ExperimentExampleResult(
                example_id=example.example_id,
                question=example.question,
                gold_answers=example.answers,
                run=run,
                kpi_report=report,
                metadata=dict(example.metadata),
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            return ExperimentExampleResult(
                example_id=example.example_id,
                question=example.question,
                gold_answers=example.answers,
                run=None,
                kpi_report=None,
                error=str(exc),
                metadata={
                    "traceback": traceback.format_exc(),
                    **dict(example.metadata),
                },
            )
