"""Fluent builder helpers for dataset client assembly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ragwatch_client.datasets.base import CorpusDataset

from .clients import RAGDatasetClient
from .interfaces import DatasetClient

if TYPE_CHECKING:  # pragma: no cover
    from ragwatch_client.pipelines.base import RAGPipelineBuilder
    from ragwatch_client.vectorstores.base import VectorStoreAdapter


@dataclass(slots=True)
class RAGDatasetBuilder:
    """Fluent builder for composing RAG dataset clients."""

    corpus_dataset: Optional[CorpusDataset] = None
    vectorstore_adapter: Optional["VectorStoreAdapter"] = None
    pipeline_builder: Optional["RAGPipelineBuilder"] = None
    top_k: Optional[int] = None

    def with_corpus(self, corpus_dataset: CorpusDataset) -> "RAGDatasetBuilder":
        self.corpus_dataset = corpus_dataset
        return self

    def with_vectorstore(self, vectorstore_adapter: "VectorStoreAdapter") -> "RAGDatasetBuilder":
        self.vectorstore_adapter = vectorstore_adapter
        return self

    def with_pipeline(self, pipeline_builder: "RAGPipelineBuilder") -> "RAGDatasetBuilder":
        self.pipeline_builder = pipeline_builder
        return self

    def with_top_k(self, top_k: int | None) -> "RAGDatasetBuilder":
        self.top_k = top_k
        return self

    def build(self) -> DatasetClient:
        if not self.corpus_dataset:
            raise ValueError("Corpus dataset must be provided before building a RAG dataset client.")
        if not self.vectorstore_adapter:
            raise ValueError("Vectorstore adapter must be provided before building a RAG dataset client.")
        if not self.pipeline_builder:
            raise ValueError("Pipeline builder must be provided before building a RAG dataset client.")

        return RAGDatasetClient(
            corpus_dataset=self.corpus_dataset,
            vectorstore_adapter=self.vectorstore_adapter,
            pipeline_builder=self.pipeline_builder,
            top_k_override=self.top_k,
        )


__all__ = ["RAGDatasetBuilder"]