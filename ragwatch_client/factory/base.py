"""Common interfaces and wiring utilities for assembling runnable datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, runtime_checkable

from ragwatch_client.datasets.base import CorpusDataset

if TYPE_CHECKING:  # pragma: no cover
    from ragwatch_client.pipelines.base import RAGPipelineBuilder
    from ragwatch_client.vectorstores.base import VectorStoreAdapter


@dataclass(slots=True)
class DatasetResources:
    """Artifacts returned by dataset clients when preparing a run."""

    vectorstore: Any
    rag_chain: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetClient(Protocol):
    """Minimal interface dataset clients must provide."""

    @property
    def description(self) -> str:
        """Human-readable summary surfaced in CLI listings."""

    @property
    def dataset_name(self) -> str:
        """Unique dataset name used for CLI selection and log defaults."""

    @property
    def log_env_var(self) -> str | None:
        """Optional environment variable controlling log location."""

    def load_questions(self) -> List[Dict[str, Any]]:
        """Return a list of QA dicts consumed by the generic runner."""

    def prepare_resources(self) -> DatasetResources:
        """Eagerly build or load the vector store and downstream RAG chain."""


@dataclass(slots=True)
class VectorstoreDatasetAdapter(DatasetClient):
    """Compose a corpus dataset, vectorstore adapter, and pipeline builder."""

    data_source: CorpusDataset
    vectorstore_adapter: "VectorStoreAdapter"
    pipeline_builder: "RAGPipelineBuilder"
    top_k_override: Optional[int] = None

    @property
    def dataset_name(self) -> str:
        return self.data_source.dataset_name

    @property
    def description(self) -> str:
        return getattr(self.data_source, "description", "")

    @property
    def log_env_var(self) -> str | None:
        return getattr(self.data_source, "log_env_var", None)

    def load_questions(self) -> List[Dict[str, Any]]:
        return self.data_source.load_questions()

    def prepare_resources(self) -> DatasetResources:
        docs = self.data_source.build_document_corpus()
        vectorstore, source = self.vectorstore_adapter.load_or_build(self.dataset_name, docs)
        top_k = self.top_k_override or self.vectorstore_adapter.default_top_k
        retriever = self.vectorstore_adapter.get_retriever(vectorstore, top_k=top_k)
        rag_chain = self.pipeline_builder.build_chain(retriever)
        metadata = self._build_metadata(len(docs), source, top_k)
        return DatasetResources(vectorstore=vectorstore, rag_chain=rag_chain, metadata=metadata)

    # ----- Internal helpers -------------------------------------------------
    def _build_metadata(self, doc_count: int, source: str, top_k: int) -> Dict[str, Any]:
        metadata = self.vectorstore_adapter.build_metadata(doc_count=doc_count, source=source, top_k=top_k)
        metadata.setdefault("doc_count", doc_count)
        metadata.setdefault("top_k", top_k)
        metadata.setdefault("vectorstore_source", source)
        return metadata


def build_vectorstore_dataset(
    *,
    data_source: CorpusDataset,
    vectorstore: "VectorStoreAdapter",
    pipeline: "RAGPipelineBuilder",
    top_k: Optional[int] = None,
) -> DatasetClient:
    """Factory helper that wires a corpus dataset into a vectorstore-backed dataset."""

    return VectorstoreDatasetAdapter(
        data_source=data_source,
        vectorstore_adapter=vectorstore,
        pipeline_builder=pipeline,
        top_k_override=top_k,
    )


__all__ = [
    "DatasetClient",
    "DatasetResources",
    "VectorstoreDatasetAdapter",
    "build_vectorstore_dataset",
]
