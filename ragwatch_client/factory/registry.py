"""Dataset registry wiring built on top of the abstract RAG factory."""
from __future__ import annotations

from typing import Callable, Dict

from ragwatch_client.datasets.hotpotqa import (
    DATASET_NAME as HOTPOTQA_NAME,
    HotpotQADataSource,
)
from ragwatch_client.pipelines import BasicRAGPipelineBuilder
from ragwatch_client.vectorstores import FaissVectorStoreAdapter

from .builder import RAGDatasetBuilder
from .interfaces import DatasetClient


def build_hotpotqa_dataset() -> DatasetClient:
    """Construct the runnable HotpotQA dataset client using the RAG builder."""

    return (
        RAGDatasetBuilder()
        .with_corpus(HotpotQADataSource())
        .with_vectorstore(FaissVectorStoreAdapter())
        .with_pipeline(BasicRAGPipelineBuilder())
    ).build()


DATASETS: Dict[str, Callable[[], DatasetClient]] = {
    HOTPOTQA_NAME: build_hotpotqa_dataset,
}

__all__ = ["DATASETS", "build_hotpotqa_dataset"]
