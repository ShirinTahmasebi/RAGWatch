"""Abstract interfaces for RAG pipeline builders."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RAGPipelineBuilder(ABC):
    """Builds executable RAG chains from a retriever."""

    @abstractmethod
    def build_chain(self, retriever: Any) -> Any:
        """Return a callable LangChain (or similar) pipeline."""


__all__ = ["RAGPipelineBuilder"]
