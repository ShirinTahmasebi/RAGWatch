"""Dataset-agnostic client utilities for running RAGWatch demos."""
from __future__ import annotations

from .datasets import DATASETS
from .runner import DatasetRunConfig, run_eval, run_stream

__all__ = [
    "DATASETS",
    "DatasetRunConfig",
    "run_eval",
    "run_stream",
]
