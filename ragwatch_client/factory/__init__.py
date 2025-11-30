"""Public factory API for composing runnable RAG datasets."""
from __future__ import annotations

from .builder import RAGDatasetBuilder
from .clients import RAGDatasetClient
from .interfaces import DatasetClient, RAGRuntimeResources
from .registry import DATASETS, build_hotpotqa_dataset

__all__ = [
    "DATASETS",
    "DatasetClient",
    "RAGDatasetBuilder",
    "RAGDatasetClient",
    "RAGRuntimeResources",
    "build_hotpotqa_dataset",
]
