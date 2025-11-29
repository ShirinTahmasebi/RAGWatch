"""Command-line entry point for ragwatch_client."""
from __future__ import annotations

from typing import Optional

import typer

from . import dataset_registry
from .runner import DatasetRunConfig, run_eval, run_stream

app = typer.Typer(help="Dataset-agnostic client for running RAGWatch demos.")


def _register_dataset_commands():
    for dataset in dataset_registry.all():
        _register_single_dataset(dataset)


def _register_single_dataset(dataset):
    dataset_app = typer.Typer(help=dataset.description)

    @dataset_app.command("eval")
    def eval_command(
        log_dir: Optional[str] = typer.Option(
            None,
            help="Directory for logs (overrides dataset defaults).",
        ),
        dataset_name: str = typer.Option(
            dataset.default_dataset_name,
            help="Dataset name stored in logs.",
        ),
        pipeline_name: str = typer.Option(
            dataset.default_pipeline_name,
            help="Pipeline identifier stored in logs.",
        ),
    ):
        """Run a single-pass evaluation for the dataset."""

        cfg = DatasetRunConfig(
            log_dir=log_dir,
            dataset_name=dataset_name,
            pipeline_name=pipeline_name,
        )
        run_eval(dataset, cfg)

    @dataset_app.command("stream")
    def stream_command(
        log_dir: Optional[str] = typer.Option(
            None,
            help="Directory for logs (overrides dataset defaults).",
        ),
        dataset_name: str = typer.Option(
            dataset.default_dataset_name,
            help="Dataset name stored in logs.",
        ),
        pipeline_name: str = typer.Option(
            dataset.default_pipeline_name,
            help="Pipeline identifier stored in logs.",
        ),
        interval: float = typer.Option(1.0, help="Delay in seconds between questions."),
        max_iterations: Optional[int] = typer.Option(
            None,
            help="Optional cap on iterations (defaults to infinite loop).",
        ),
    ):
        """Continuously run questions for the dataset at a fixed cadence."""

        cfg = DatasetRunConfig(
            log_dir=log_dir,
            dataset_name=dataset_name,
            pipeline_name=pipeline_name,
            interval_seconds=interval,
            max_iterations=max_iterations,
        )
        run_stream(dataset, cfg)

    app.add_typer(dataset_app, name=dataset.slug)


@app.command()
def datasets():
    """List all datasets registered with the client."""

    for dataset in dataset_registry.all():
        typer.echo(f"{dataset.slug}\t{dataset.description}")


_register_dataset_commands()

if __name__ == "__main__":
    app()
