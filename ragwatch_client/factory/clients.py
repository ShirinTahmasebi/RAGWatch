"""Concrete dataset client implementations built from composable parts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ragwatch_client.datasets.base import CorpusDataset

from .interfaces import DatasetClient, RAGRuntimeResources

if TYPE_CHECKING:  # pragma: no cover
    from ragwatch_client.pipelines.base import RAGPipelineBuilder
    from ragwatch_client.vectorstores.base import VectorStoreAdapter


@dataclass(slots=True)
class RAGDatasetClient(DatasetClient):
    """Compose a corpus dataset, vectorstore adapter, and pipeline builder."""

    corpus_dataset: CorpusDataset
    vectorstore_adapter: "VectorStoreAdapter"
    pipeline_builder: "RAGPipelineBuilder"
    top_k_override: Optional[int] = None

    @property
    def dataset_name(self) -> str:
        return self.corpus_dataset.dataset_name

    @property
    def description(self) -> str:
        return getattr(self.corpus_dataset, "description", "")

    @property
    def log_env_var(self) -> str | None:
        return getattr(self.corpus_dataset, "log_env_var", None)

    def load_questions(self) -> List[Dict[str, Any]]:
        return self.corpus_dataset.load_questions()

    def prepare_resources(self) -> RAGRuntimeResources:
        docs = self.corpus_dataset.build_document_corpus()
        vectorstore, source = self.vectorstore_adapter.load_or_build(self.dataset_name, docs)
        top_k = self.top_k_override or self.vectorstore_adapter.default_top_k
        retriever = self.vectorstore_adapter.get_retriever(vectorstore, top_k=top_k)
        rag_chain = self.pipeline_builder.build_chain(retriever)
        metadata = self._build_metadata(len(docs), source, top_k)
        return RAGRuntimeResources(vectorstore=vectorstore, rag_chain=rag_chain, metadata=metadata)

    # ----- Internal helpers -------------------------------------------------
    def _build_metadata(self, doc_count: int, source: str, top_k: int) -> Dict[str, Any]:
        metadata = self.vectorstore_adapter.build_metadata(doc_count=doc_count, source=source, top_k=top_k)
        metadata.setdefault("doc_count", doc_count)
        metadata.setdefault("top_k", top_k)
        metadata.setdefault("vectorstore_source", source)
        return metadata


__all__ = ["RAGDatasetClient"]
