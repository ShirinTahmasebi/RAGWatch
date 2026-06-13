"""Export experiment results to JSONL, CSV, and JSON files."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ragwatch.experiments.schema import (
    ExperimentExampleResult,
    ExperimentResult,
)
from ragwatch.metrics.report import to_dict, to_flat_dict


def _retrieved_source(rd: Any) -> Any:
    """Best-effort source for a retrieved document (from its metadata)."""
    return rd.document.metadata.get("source")


def _run_record(result: ExperimentResult, ex: ExperimentExampleResult) -> dict[str, Any]:
    """Build a single runs.jsonl record for an example result."""
    config = result.config
    if not ex.succeeded or ex.run is None:
        return {
            "experiment_name": config.experiment_name,
            "example_id": ex.example_id,
            "question": ex.question,
            "error": ex.error,
        }

    run = ex.run
    return {
        "experiment_name": config.experiment_name,
        "dataset_name": config.dataset_name,
        "retriever_name": config.retriever_name,
        "generator_name": config.generator_name,
        "top_k": config.top_k,
        "example_id": ex.example_id,
        "question": ex.question,
        "gold_answers": ex.gold_answers,
        "run_id": run.run_id,
        "answer": run.generation.answer,
        "retrieved_doc_ids": [rd.document.doc_id for rd in run.retrieved_documents],
        "retrieved_scores": [rd.score for rd in run.retrieved_documents],
        "retrieved_ranks": [rd.rank for rd in run.retrieved_documents],
        "retrieved_sources": [_retrieved_source(rd) for rd in run.retrieved_documents],
        "latency_ms": run.latency_ms,
        "metadata": run.metadata,
    }


def export_runs_jsonl(result: ExperimentResult, path: str | Path) -> None:
    """Export per-example RAG runs to a JSONL file (one line per example).

    Full retrieved document text is intentionally excluded to keep files small.
    Failed examples are written with their error message.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for ex in result.examples:
            record = _run_record(result, ex)
            f.write(json.dumps(record, default=str) + "\n")


def export_kpis_jsonl(result: ExperimentResult, path: str | Path) -> None:
    """Export KPI reports to a JSONL file (one line per successful example)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = result.config
    with path.open("w", encoding="utf-8") as f:
        for ex in result.examples:
            if not ex.succeeded or ex.kpi_report is None:
                continue
            record: dict[str, Any] = {
                "experiment_name": config.experiment_name,
                "dataset_name": config.dataset_name,
                "retriever_name": config.retriever_name,
                "generator_name": config.generator_name,
                "top_k": config.top_k,
                "example_id": ex.example_id,
            }
            record.update(to_flat_dict(ex.kpi_report))
            f.write(json.dumps(record, default=str) + "\n")


def export_kpis_csv(result: ExperimentResult, path: str | Path) -> None:
    """Export KPI reports to a CSV file (one row per successful example).

    Columns: experiment config + example_id + query + run_id + one per KPI name.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    successful = [
        ex for ex in result.examples if ex.succeeded and ex.kpi_report is not None
    ]
    if not successful:
        path.write_text("", encoding="utf-8")
        return

    config = result.config
    config_columns = [
        "experiment_name",
        "dataset_name",
        "retriever_name",
        "generator_name",
        "top_k",
        "example_id",
        "query",
        "run_id",
    ]

    # KPI columns come from the report results (which reference the catalog).
    kpi_columns = [r.name for r in successful[0].kpi_report.results]  # type: ignore[union-attr]
    columns = config_columns + kpi_columns

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()

        for ex in successful:
            report = ex.kpi_report
            flat = to_flat_dict(report)  # type: ignore[arg-type]
            row: dict[str, Any] = {
                "experiment_name": config.experiment_name,
                "dataset_name": config.dataset_name,
                "retriever_name": config.retriever_name,
                "generator_name": config.generator_name,
                "top_k": config.top_k,
                "example_id": ex.example_id,
                "query": report.query,  # type: ignore[union-attr]
                "run_id": report.run_id,  # type: ignore[union-attr]
            }
            for col in kpi_columns:
                row[col] = flat.get(col)
            writer.writerow(row)


def export_db_kpis_json(result: ExperimentResult, path: str | Path) -> None:
    """Export the DB KPI snapshot to JSON.

    If no DB KPI report is present, writes ``{"db_kpi_report": null}``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if result.db_kpi_report is None:
        payload: dict[str, Any] = {"db_kpi_report": None}
    else:
        payload = to_dict(result.db_kpi_report)

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _aggregate_metrics(result: ExperimentResult) -> dict[str, float]:
    """Compute simple averages of selected numeric KPIs over successful runs."""
    wanted = (
        "total_latency_ms",
        "retrieval_score_mean",
        "answer_length_words",
    )
    sums: dict[str, float] = {k: 0.0 for k in wanted}
    counts: dict[str, int] = {k: 0 for k in wanted}

    for ex in result.examples:
        if not ex.succeeded or ex.kpi_report is None:
            continue
        for kpi in ex.kpi_report.results:
            if kpi.name in wanted and isinstance(kpi.value, (int, float)):
                sums[kpi.name] += float(kpi.value)
                counts[kpi.name] += 1

    return {
        f"avg_{name}": sums[name] / counts[name]
        for name in wanted
        if counts[name] > 0
    }


def export_experiment_summary_json(result: ExperimentResult, path: str | Path) -> None:
    """Export an experiment summary (config, counts, timestamp, aggregates)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config = result.config
    summary: dict[str, Any] = {
        "config": {
            "experiment_name": config.experiment_name,
            "dataset_name": config.dataset_name,
            "retriever_name": config.retriever_name,
            "generator_name": config.generator_name,
            "top_k": config.top_k,
            "max_examples": config.max_examples,
            "metadata": config.metadata,
        },
        "num_examples": result.num_examples,
        "num_successful": result.num_successful,
        "num_failed": result.num_failed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "aggregate_metrics": _aggregate_metrics(result),
        "has_db_kpi_report": result.db_kpi_report is not None,
    }
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


def export_experiment(result: ExperimentResult, output_dir: str | Path) -> None:
    """Export all standard experiment files to ``output_dir``.

    Creates: runs.jsonl, kpis.jsonl, kpis.csv, db_kpis.json, summary.json.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_runs_jsonl(result, output_dir / "runs.jsonl")
    export_kpis_jsonl(result, output_dir / "kpis.jsonl")
    export_kpis_csv(result, output_dir / "kpis.csv")
    export_db_kpis_json(result, output_dir / "db_kpis.json")
    export_experiment_summary_json(result, output_dir / "summary.json")

