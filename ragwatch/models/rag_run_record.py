"""Pydantic model describing a full RAG pipeline execution."""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from .retrieved_doc import RetrievedDoc


class RAGRunRecord(BaseModel):
    """End-to-end record for a single monitored RAG run."""

    run_id: str
    session_id: str
    timestamp: dt.datetime

    dataset_name: str
    version: str

    question: str
    answer: str

    retrieved_docs: List[RetrievedDoc]

    latency_ms: Dict[str, float]
    token_usage: Dict[str, int]

    extra: Dict[str, Any] = Field(default_factory=dict)
