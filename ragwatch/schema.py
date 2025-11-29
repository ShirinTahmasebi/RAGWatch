"""Core schema definitions for RAGWatch logging."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import datetime as dt

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    """Metadata representing a document returned by retrieval."""

    doc_id: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RunRecord(BaseModel):
    """End-to-end record of a single RAG pipeline execution."""

    run_id: str
    session_id: str
    timestamp: dt.datetime

    dataset_name: str
    pipeline_name: str

    question: str
    answer: str

    retrieved_docs: List[RetrievedDoc]

    latency_ms: Dict[str, float]
    token_usage: Dict[str, int]

    extra: Dict[str, Any] = Field(default_factory=dict)
