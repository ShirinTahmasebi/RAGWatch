"""Common interfaces for dataset-specific client modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass(slots=True)
class DatasetResources:
    """Artifacts returned by dataset modules when preparing a run."""

    vectorstore: Any
    rag_chain: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DatasetClient(Protocol):
    """Minimal interface dataset implementations must provide."""

    @property
    def id(self) -> str:
        """Stable dataset identifier used in CLI commands."""

    @property
    def description(self) -> str:
        """Human-readable summary surfaced in CLI listings."""

    @property
    def default_dataset_name(self) -> str:
        """Name stored alongside logs when user does not override it."""

    @property
    def default_pipeline_name(self) -> str:
        """Pipeline label stored in logs when the user does not override it."""

    def load_questions(self) -> List[Dict[str, Any]]:
        """Return a list of QA dicts consumed by the generic runner."""

    def prepare_resources(self) -> DatasetResources:
        """Eagerly build or load the vector store and downstream RAG chain."""

