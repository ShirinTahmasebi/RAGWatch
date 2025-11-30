"""Shared Typer argument and option definitions for CLI commands."""
from __future__ import annotations

import typer

DATASET_ARGUMENT = typer.Argument(
    ...,
    help="Dataset id (e.g., hotpotqa).",
)
LOG_DIR_OPTION = typer.Option(
    None,
    "--log-dir",
    help="Directory for logs (overrides dataset defaults).",
)
DATASET_NAME_OPTION = typer.Option(
    None,
    "--dataset-name",
    help="Dataset name stored in logs (defaults to dataset metadata).",
)
PIPELINE_NAME_OPTION = typer.Option(
    None,
    "--pipeline-name",
    help="Pipeline identifier stored in logs (defaults to dataset metadata).",
)
INTERVAL_OPTION = typer.Option(
    1.0,
    "--interval",
    help="Delay in seconds between questions.",
)
MAX_ITERATIONS_OPTION = typer.Option(
    None,
    "--max-iterations",
    help="Optional cap on iterations (defaults to infinite loop).",
)
