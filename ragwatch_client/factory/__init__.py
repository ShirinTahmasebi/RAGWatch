"""Composable factories that wire datasets, vectorstores, and pipelines."""
from __future__ import annotations

from typing import Callable, Dict

from .base import (
    DatasetClient,
    DatasetResources,
    VectorstoreDatasetAdapter,
    build_vectorstore_dataset,
)
from ragwatch_client.datasets.hotpotqa import (
    DATASET_NAME as HOTPOTQA_NAME,
    build_data_source as build_hotpotqa_data_source,
)
from ragwatch_client.pipelines import BasicRAGPipelineBuilder
from ragwatch_client.vectorstores import FaissVectorStoreAdapter


def build_hotpotqa_dataset() -> DatasetClient:
    data_source = build_hotpotqa_data_source()
    vectorstore = FaissVectorStoreAdapter()
    pipeline = BasicRAGPipelineBuilder()
    return build_vectorstore_dataset(
        data_source=data_source,
        vectorstore=vectorstore,
        pipeline=pipeline,
    )


DATASETS: Dict[str, Callable[[], DatasetClient]] = {
    HOTPOTQA_NAME: build_hotpotqa_dataset,
}

__all__ = [
    "DATASETS",
    "DatasetClient",
    "DatasetResources",
    "VectorstoreDatasetAdapter",
    "build_vectorstore_dataset",
    "build_hotpotqa_dataset",
]
