"""Dataset-agnostic client utilities for running RAGWatch demos."""
from __future__ import annotations

from .datasets import registry as dataset_registry
from .runner import DatasetRunConfig, run_eval, run_stream

__all__ = [
    "dataset_registry",
    "DatasetRunConfig",
    "run_eval",
    "run_stream",
]
