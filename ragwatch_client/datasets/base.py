"""Common interfaces for dataset-specific client modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


@dataclass(slots=True)
class DatasetResources:
    """Artifacts returned by dataset modules when preparing a run."""

    vectorstore: Any
    rag_chain: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetClient(Protocol):
    """Minimal interface dataset implementations must provide."""

    slug: str
    description: str
    default_dataset_name: str
    default_pipeline_name: str

    def load_questions(self) -> List[Dict[str, Any]]:
        """Return a list of QA dicts consumed by the generic runner."""

    def prepare_resources(self) -> DatasetResources:
        """Eagerly build or load the vector store and downstream RAG chain."""


class DatasetRegistry:
    """Runtime registry that maps dataset slugs to implementations."""

    def __init__(self):
        self._datasets: Dict[str, DatasetClient] = {}

    def register(self, dataset: DatasetClient) -> None:
        if dataset.slug in self._datasets:
            raise ValueError(f"Dataset slug already registered: {dataset.slug}")
        self._datasets[dataset.slug] = dataset

    def get(self, slug: str) -> DatasetClient:
        try:
            return self._datasets[slug]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown dataset slug: {slug}") from exc

    def all(self) -> List[DatasetClient]:
        return list(self._datasets.values())
