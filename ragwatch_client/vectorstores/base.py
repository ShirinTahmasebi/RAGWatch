"""Abstract interfaces for pluggable vectorstore adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class VectorStoreAdapter(ABC):
    """Defines how datasets build/load vectorstores and derive retrievers."""

    default_top_k: int = 5

    def load_or_build(self, dataset_name: str, docs: List[Dict[str, Any]]) -> Tuple[Any, str]:
        """Load the vectorstore or build it from docs when missing."""

        try:
            return self.load_if_exists(dataset_name), "loaded"
        except FileNotFoundError:
            vectorstore = self.build(dataset_name, docs)
            if vectorstore is None:
                raise RuntimeError("VectorStoreAdapter.build() must return a vectorstore instance")
            return vectorstore, "built"

    @abstractmethod
    def load_if_exists(self, dataset_name: str) -> Any:
        """Load a persisted vectorstore for the dataset if present."""

    @abstractmethod
    def build(self, dataset_name: str, docs: List[Dict[str, Any]]) -> Any:
        """Build and persist a vectorstore from docs."""

    def get_retriever(self, vectorstore: Any, *, top_k: int | None = None) -> Any:
        """Return a retriever object for the supplied vectorstore."""

        k = top_k or self.default_top_k
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def build_metadata(self, *, doc_count: int, source: str, top_k: int) -> Dict[str, Any]:
        """Metadata describing how the vectorstore was prepared."""

        return {
            "doc_count": doc_count,
            "vectorstore_source": source,
            "top_k": top_k,
        }


__all__ = ["VectorStoreAdapter"]
