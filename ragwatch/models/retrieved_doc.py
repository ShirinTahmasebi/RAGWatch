"""Pydantic model representing a single retrieved document."""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    """Metadata wrapper for documents returned by retrieval."""

    doc_id: str
    score: float
    source: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
