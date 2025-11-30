"""Command-line entry point for ragwatch_client."""
from __future__ import annotations

from typing import Any, Callable, Optional

import typer

from ragwatch_client.datasets.base import DatasetClient

from .datasets import DATASETS
from .cli_params import (
    DATASET_ARGUMENT,
    DATASET_NAME_OPTION,
    INTERVAL_OPTION,
    LOG_DIR_OPTION,
    MAX_ITERATIONS_OPTION,
    PIPELINE_NAME_OPTION,
)
from .runner import DatasetRunConfig, run_eval, run_stream

app = typer.Typer(help="Dataset-agnostic client for running RAGWatch demos.")


def _load_dataset(dataset_id: str) -> DatasetClient:
    try:
        dataset_cls = DATASETS[dataset_id]
    except KeyError as exc:  # pragma: no cover - CLI validation
        available = ", ".join(sorted(DATASETS.keys())) or "<none>"
        raise typer.BadParameter(
            f"Unknown dataset '{dataset_id}'. Available options: {available}."
        ) from exc
    return dataset_cls()


def _execute_cli_command(
    runner: Callable[[DatasetClient, DatasetRunConfig], Any],
    *,
    dataset_id: str,
    log_dir: Optional[str],
    dataset_name: Optional[str],
    pipeline_name: Optional[str],
    interval: Optional[float] = None,
    max_iterations: Optional[int] = None,
):
    dataset_client: DatasetClient = _load_dataset(dataset_id)
    cfg = DatasetRunConfig(
        log_dir=log_dir,
        dataset_name=dataset_name or dataset_client.default_dataset_name,
        pipeline_name=pipeline_name or dataset_client.default_pipeline_name,
        interval_seconds=interval if interval is not None else 1.0,
        max_iterations=max_iterations,
    )
    runner(dataset_client, cfg)


@app.command()
def eval(
    dataset: str = DATASET_ARGUMENT,
    log_dir: Optional[str] = LOG_DIR_OPTION,
    dataset_name: Optional[str] = DATASET_NAME_OPTION,
    pipeline_name: Optional[str] = PIPELINE_NAME_OPTION,
):
    """Run a single-pass evaluation for a dataset."""

    _execute_cli_command(
        run_eval,
        dataset_id=dataset,
        log_dir=log_dir,
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
    )


@app.command()
def stream(
    dataset: str = DATASET_ARGUMENT,
    log_dir: Optional[str] = LOG_DIR_OPTION,
    dataset_name: Optional[str] = DATASET_NAME_OPTION,
    pipeline_name: Optional[str] = PIPELINE_NAME_OPTION,
    interval: float = INTERVAL_OPTION,
    max_iterations: Optional[int] = MAX_ITERATIONS_OPTION,
):
    """Continuously run questions for a dataset at a fixed cadence."""

    _execute_cli_command(
        run_stream,
        dataset_id=dataset,
        log_dir=log_dir,
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
        interval=interval,
        max_iterations=max_iterations,
    )


@app.command()
def datasets():
    """List all datasets bundled with the client."""

    for dataset_id, dataset_cls in sorted(DATASETS.items()):
        description = getattr(dataset_cls, "description", "")
        typer.echo(f"{dataset_id}\t{description}")


if __name__ == "__main__":
    app()
