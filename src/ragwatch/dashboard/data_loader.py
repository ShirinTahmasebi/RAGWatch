"""File-loading utilities for the RAGWatch research dashboard.

These functions read the comparison output files produced by
``ragwatch.experiments.comparison.export_comparison``. They have no Streamlit
dependency so they can be unit tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

COMBINED_KPIS_FILE = "combined_kpis.csv"
COMPARISON_SUMMARY_CSV_FILE = "comparison_summary.csv"
COMPARISON_SUMMARY_JSON_FILE = "comparison_summary.json"
EXPERIMENTS_DIRNAME = "experiments"
RUNS_JSONL_FILE = "runs.jsonl"


def _require_file(path: Path) -> Path:
    """Return ``path`` if it exists, otherwise raise a clear FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found: {path}")
    return path


def load_combined_kpis(comparison_dir: str | Path) -> pd.DataFrame:
    """Load ``combined_kpis.csv`` from a comparison output directory."""
    path = _require_file(Path(comparison_dir) / COMBINED_KPIS_FILE)
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_comparison_summary(comparison_dir: str | Path) -> pd.DataFrame:
    """Load ``comparison_summary.csv`` from a comparison output directory."""
    path = _require_file(Path(comparison_dir) / COMPARISON_SUMMARY_CSV_FILE)
    if path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def load_comparison_summary_json(comparison_dir: str | Path) -> dict[str, Any]:
    """Load ``comparison_summary.json`` from a comparison output directory."""
    path = _require_file(Path(comparison_dir) / COMPARISON_SUMMARY_JSON_FILE)
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def list_experiment_dirs(comparison_dir: str | Path) -> list[Path]:
    """Return the per-experiment subdirectories under ``experiments/``.

    Returns an empty list if the experiments directory does not exist.
    """
    experiments_dir = Path(comparison_dir) / EXPERIMENTS_DIRNAME
    if not experiments_dir.exists():
        return []
    return sorted(p for p in experiments_dir.iterdir() if p.is_dir())


def load_runs_jsonl(experiment_dir: str | Path) -> pd.DataFrame:
    """Load ``runs.jsonl`` from a single experiment directory.

    Each line is a JSON object; the result is a DataFrame with one row per run.
    """
    path = _require_file(Path(experiment_dir) / RUNS_JSONL_FILE)
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
