"""Interfaces shared by factory-built dataset clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass(slots=True)
class RAGRuntimeResources:
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

    def prepare_resources(self) -> RAGRuntimeResources:
        """Eagerly build or load the vector store and downstream RAG chain."""


__all__ = ["DatasetClient", "RAGRuntimeResources"]
