"""Shared Typer argument and option definitions for CLI commands."""
from __future__ import annotations

import typer

DATASET_ARGUMENT = typer.Argument(
    ...,
    help="Dataset name (e.g., hotpotqa).",
)
LOG_DIR_OPTION = typer.Option(
    None,
    "--log-dir",
    help="Directory for logs (overrides dataset defaults).",
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
