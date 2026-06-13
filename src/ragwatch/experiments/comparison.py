"""Retriever comparison experiments.

Runs the same QA dataset across multiple retriever backends and top-k settings,
reusing the existing ExperimentRunner, and exports combined comparison files.
"""

from __future__ import annotations

import csv
import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragwatch.core.interfaces import BaseRAGClient
from ragwatch.core.schema import QAExample
from ragwatch.experiments.exporters import export_experiment
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import ExperimentConfig, ExperimentResult
from ragwatch.metrics.catalog import KPIId
from ragwatch.metrics.report import KPIEngine, to_flat_dict

# Summary metrics averaged per experiment setting. These reference the central
# KPI catalog so KPI names are not hardcoded as raw strings.
SUMMARY_METRIC_IDS: list[KPIId] = [
    KPIId.TOTAL_LATENCY_MS,
    KPIId.RETRIEVAL_LATENCY_MS,
    KPIId.GENERATION_LATENCY_MS,
    KPIId.RETRIEVAL_SCORE_MEAN,
    KPIId.RETRIEVAL_REDUNDANCY,
    KPIId.ANSWER_LENGTH_WORDS,
]


@dataclass
class ComparisonConfig:
    """Configuration for a retriever comparison experiment."""

    comparison_name: str
    dataset_name: str
    top_k_values: list[int]
    max_examples: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverExperimentSpec:
    """A single retriever backend to include in a comparison."""

    name: str
    client: BaseRAGClient
    retriever_name: str
    generator_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Results from a completed retriever comparison."""

    config: ComparisonConfig
    experiment_results: dict[str, ExperimentResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def experiment_keys(self) -> list[str]:
        """Stable, readable keys for each experiment setting."""
        return list(self.experiment_results.keys())

    @property
    def num_experiments(self) -> int:
        """Number of experiment settings (retriever x top_k)."""
        return len(self.experiment_results)


class ComparisonRunner:
    """Run several retriever specs across multiple top-k values."""

    def __init__(self, kpi_engine: KPIEngine | None = None) -> None:
        self._kpi_engine = kpi_engine if kpi_engine is not None else KPIEngine()

    def run(
        self,
        qa_examples: list[QAExample],
        specs: list[RetrieverExperimentSpec],
        config: ComparisonConfig,
    ) -> ComparisonResult:
        """Run each spec across each top-k value and collect ExperimentResults.

        A failure in one experiment setting does not abort the whole comparison.
        """
        experiment_results: dict[str, ExperimentResult] = {}

        for spec in specs:
            for top_k in config.top_k_values:
                key = f"{spec.name}_top{top_k}"
                experiment_results[key] = self._run_one_setting(
                    qa_examples, spec, top_k, config, key
                )

        return ComparisonResult(
            config=config,
            experiment_results=experiment_results,
            metadata={
                "num_experiments": len(experiment_results),
                "experiment_keys": list(experiment_results.keys()),
            },
        )

    def _run_one_setting(
        self,
        qa_examples: list[QAExample],
        spec: RetrieverExperimentSpec,
        top_k: int,
        config: ComparisonConfig,
        key: str,
    ) -> ExperimentResult:
        """Run one (retriever, top_k) experiment, capturing setup failures."""
        experiment_config = ExperimentConfig(
            experiment_name=key,
            dataset_name=config.dataset_name,
            retriever_name=spec.retriever_name,
            generator_name=spec.generator_name,
            top_k=top_k,
            max_examples=config.max_examples,
            metadata={
                "comparison_name": config.comparison_name,
                "experiment_key": key,
                "spec_name": spec.name,
                **dict(spec.metadata),
            },
        )

        try:
            runner = ExperimentRunner(
                client=spec.client, kpi_engine=self._kpi_engine
            )
            return runner.run(
                qa_examples=qa_examples,
                config=experiment_config,
                limit=config.max_examples,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            return ExperimentResult(
                config=experiment_config,
                examples=[],
                db_kpi_report=None,
                metadata={
                    "num_examples": 0,
                    "num_successful": 0,
                    "num_failed": 0,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _summary_aggregates(result: ExperimentResult) -> dict[str, float | None]:
    """Average the configured summary KPIs over successful runs.

    Returns ``avg_<kpi>`` keys; missing/non-numeric metrics map to ``None``.
    """
    wanted = [str(kpi_id) for kpi_id in SUMMARY_METRIC_IDS]
    sums: dict[str, float] = {name: 0.0 for name in wanted}
    counts: dict[str, int] = {name: 0 for name in wanted}

    for ex in result.examples:
        if not ex.succeeded or ex.kpi_report is None:
            continue
        for kpi in ex.kpi_report.results:
            if kpi.name in sums and isinstance(kpi.value, (int, float)) and not isinstance(
                kpi.value, bool
            ):
                sums[kpi.name] += float(kpi.value)
                counts[kpi.name] += 1

    return {
        f"avg_{name}": (sums[name] / counts[name] if counts[name] > 0 else None)
        for name in wanted
    }


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def export_combined_kpis_csv(result: ComparisonResult, path: str | Path) -> None:
    """Write one row per successful run across all experiments."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison_name = result.config.comparison_name
    base_columns = [
        "comparison_name",
        "experiment_key",
        "dataset_name",
        "retriever_name",
        "generator_name",
        "top_k",
        "example_id",
        "query",
        "run_id",
    ]

    # Determine KPI columns from the first successful run found.
    kpi_columns: list[str] = []
    for exp in result.experiment_results.values():
        for ex in exp.examples:
            if ex.succeeded and ex.kpi_report is not None:
                kpi_columns = [r.name for r in ex.kpi_report.results]
                break
        if kpi_columns:
            break

    if not kpi_columns:
        path.write_text("", encoding="utf-8")
        return

    columns = base_columns + kpi_columns

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for key, exp in result.experiment_results.items():
            config = exp.config
            for ex in exp.examples:
                if not ex.succeeded or ex.kpi_report is None:
                    continue
                flat = to_flat_dict(ex.kpi_report)
                row: dict[str, Any] = {
                    "comparison_name": comparison_name,
                    "experiment_key": key,
                    "dataset_name": config.dataset_name,
                    "retriever_name": config.retriever_name,
                    "generator_name": config.generator_name,
                    "top_k": config.top_k,
                    "example_id": ex.example_id,
                    "query": ex.kpi_report.query,
                    "run_id": ex.kpi_report.run_id,
                }
                for col in kpi_columns:
                    row[col] = flat.get(col)
                writer.writerow(row)


def export_comparison_summary_csv(result: ComparisonResult, path: str | Path) -> None:
    """Write one row per experiment setting with counts and averaged metrics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    comparison_name = result.config.comparison_name
    avg_columns = [f"avg_{kpi_id}" for kpi_id in SUMMARY_METRIC_IDS]
    columns = [
        "comparison_name",
        "experiment_key",
        "dataset_name",
        "retriever_name",
        "generator_name",
        "top_k",
        "num_examples",
        "num_successful",
        "num_failed",
    ] + avg_columns

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for key, exp in result.experiment_results.items():
            config = exp.config
            row: dict[str, Any] = {
                "comparison_name": comparison_name,
                "experiment_key": key,
                "dataset_name": config.dataset_name,
                "retriever_name": config.retriever_name,
                "generator_name": config.generator_name,
                "top_k": config.top_k,
                "num_examples": exp.num_examples,
                "num_successful": exp.num_successful,
                "num_failed": exp.num_failed,
            }
            row.update(_summary_aggregates(exp))
            writer.writerow(row)


def export_comparison_summary_json(result: ComparisonResult, path: str | Path) -> None:
    """Write a JSON summary of the comparison and per-experiment aggregates."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = result.config
    experiments = []
    for key, exp in result.experiment_results.items():
        experiments.append(
            {
                "experiment_key": key,
                "retriever_name": exp.config.retriever_name,
                "generator_name": exp.config.generator_name,
                "top_k": exp.config.top_k,
                "num_examples": exp.num_examples,
                "num_successful": exp.num_successful,
                "num_failed": exp.num_failed,
                "aggregates": _summary_aggregates(exp),
            }
        )

    summary: dict[str, Any] = {
        "comparison_name": config.comparison_name,
        "dataset_name": config.dataset_name,
        "top_k_values": config.top_k_values,
        "num_experiments": result.num_experiments,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiments": experiments,
    }
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def export_comparison(result: ComparisonResult, output_dir: str | Path) -> None:
    """Export combined comparison files plus one folder per experiment setting."""
    output_dir = Path(output_dir)
    experiments_dir = output_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    export_combined_kpis_csv(result, output_dir / "combined_kpis.csv")
    export_comparison_summary_csv(result, output_dir / "comparison_summary.csv")
    export_comparison_summary_json(result, output_dir / "comparison_summary.json")

    for key, exp in result.experiment_results.items():
        export_experiment(exp, experiments_dir / key)
