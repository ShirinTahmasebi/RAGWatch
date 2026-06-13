"""Base class for QA dataset adapters."""

from abc import ABC, abstractmethod

from ragwatch.core.schema import RAGDataset


class BaseQAAdapter(ABC):
    """Interface for dataset adapters that produce a RAGDataset."""

    @abstractmethod
    def load(
        self, split: str = "validation", max_examples: int | None = None
    ) -> RAGDataset:
        """Load a dataset split and return a RAGDataset."""
        ...
