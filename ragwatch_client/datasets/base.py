"""Dataset-only contracts without any vectorstore or pipeline wiring."""
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


@runtime_checkable
class CorpusDataset(Protocol):
    """Dataset implementations responsible for questions + corpus preparation."""

    dataset_name: str
    description: str
    log_env_var: str | None

    def load_questions(self) -> List[Dict[str, Any]]:
        """Return the ordered list of QA dicts for this dataset."""

    def build_document_corpus(self) -> List[Dict[str, Any]]:
        """Return the raw documents that should be embedded/indexed."""


__all__ = ["CorpusDataset"]

