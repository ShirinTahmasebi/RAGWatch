"""Command-line entry point for ragwatch_client."""
from __future__ import annotations

from typing import Any, Callable, Optional

import typer

from ragwatch.utils import EnvKeys, env_str

from ragwatch_client.factory import DATASETS, DatasetClient
from .cli_params import (
    DATASET_ARGUMENT,
    INTERVAL_OPTION,
    LOG_DIR_OPTION,
    MAX_ITERATIONS_OPTION,
)
from .runner import DatasetRunConfig, run_eval, run_stream

app = typer.Typer(help="Dataset-agnostic client for running RAGWatch demos.")


def _load_dataset(dataset_name: str) -> DatasetClient:
    try:
        dataset_builder = DATASETS[dataset_name]
    except KeyError as exc:  # pragma: no cover - CLI validation
        available = ", ".join(sorted(DATASETS.keys())) or "<none>"
        raise typer.BadParameter(
            f"Unknown dataset '{dataset_name}'. Available options: {available}."
        ) from exc
    return dataset_builder()


def _execute_cli_command(
    runner: Callable[[DatasetClient, DatasetRunConfig], Any],
    *,
    dataset_key: str,
    log_dir: Optional[str],
    interval: Optional[float] = None,
    max_iterations: Optional[int] = None,
):
    dataset_client: DatasetClient = _load_dataset(dataset_key)
    cfg = DatasetRunConfig(
        log_dir=log_dir,
        dataset_name=dataset_client.dataset_name,
        version=env_str(EnvKeys.VERSION),
        interval_seconds=interval if interval is not None else 1.0,
        max_iterations=max_iterations,
    )
    runner(dataset_client, cfg)


@app.command()
def eval(
    dataset: str = DATASET_ARGUMENT,
    log_dir: Optional[str] = LOG_DIR_OPTION,
):
    """Run a single-pass evaluation for a dataset."""

    _execute_cli_command(
        run_eval,
        dataset_key=dataset,
        log_dir=log_dir,
    )


@app.command()
def stream(
    dataset: str = DATASET_ARGUMENT,
    log_dir: Optional[str] = LOG_DIR_OPTION,
    interval: float = INTERVAL_OPTION,
    max_iterations: Optional[int] = MAX_ITERATIONS_OPTION,
):
    """Continuously run questions for a dataset at a fixed cadence."""

    _execute_cli_command(
        run_stream,
        dataset_key=dataset,
        log_dir=log_dir,
        interval=interval,
        max_iterations=max_iterations,
    )


@app.command()
def datasets():
    """List all datasets bundled with the client."""

    for dataset_name, builder in sorted(DATASETS.items()):
        client = builder()
        description = getattr(client, "description", "")
        typer.echo(f"{dataset_name}\t{description}")


if __name__ == "__main__":
    app()
