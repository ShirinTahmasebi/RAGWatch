"""Command-line entry point for the HotpotQA prototype."""
from __future__ import annotations

from typing import Optional

import typer

from .run_eval import run_hotpotqa_eval, run_hotpotqa_stream

app = typer.Typer(help="Utilities for experimenting with the HotpotQA pipeline.")


@app.command()
def eval(
    log_dir: Optional[str] = typer.Option(
        None,
        help="Directory for HotpotQA logs (defaults to RAGWATCH_HOTPOTQA_LOG_DIR)",
    ),
    dataset_name: str = typer.Option("hotpotqa", help="Dataset name stored in logs."),
    pipeline_name: str = typer.Option("v1", help="Pipeline identifier stored in logs."),
):
    """Run the evaluation harness and log outputs via RAGWatch."""

    run_hotpotqa_eval(
        log_dir=log_dir,
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
    )


@app.command()
def stream(
    log_dir: Optional[str] = typer.Option(
        None,
        help="Directory for HotpotQA logs (defaults to RAGWATCH_HOTPOTQA_LOG_DIR)",
    ),
    dataset_name: str = typer.Option("hotpotqa", help="Dataset name stored in logs."),
    pipeline_name: str = typer.Option("v1", help="Pipeline identifier stored in logs."),
    interval: float = typer.Option(1.0, help="Delay (in seconds) between questions."),
    max_iterations: Optional[int] = typer.Option(
        None, help="Optional cap on the number of streamed questions."
    ),
):
    """Continuously sample questions and log answers at a fixed cadence."""

    run_hotpotqa_stream(
        log_dir=log_dir,
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
        interval_seconds=interval,
        max_iterations=max_iterations,
    )


if __name__ == "__main__":
    app()
