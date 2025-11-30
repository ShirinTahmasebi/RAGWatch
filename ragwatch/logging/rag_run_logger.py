"""High-level logging interface for RAGWatch."""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Optional

from ..models import RAGRunRecord, RetrievedDoc
from ..utils import JSONLWriter


class RAGWatchLogger:
    """Convenience wrapper that records RAG runs to a JSONL log."""

    def __init__(self, log_dir: str, dataset_name: str, version: str):
        self.dataset_name = dataset_name
        self.version = version
        log_file = f"{log_dir}/{dataset_name}/{dataset_name}_{version}.jsonl"
        self.writer = JSONLWriter(log_file)

    def new_run_id(self) -> str:
        return str(uuid.uuid4())

    def log_run(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        latency_ms: Dict[str, float],
        token_usage: Dict[str, int],
        session_id: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> RAGRunRecord:
        record = RAGRunRecord(
            run_id=self.new_run_id(),
            session_id=session_id,
            timestamp=dt.datetime.utcnow(),
            dataset_name=self.dataset_name,
            version=self.version,
            question=question,
            answer=answer,
            retrieved_docs=[RetrievedDoc(**doc) for doc in retrieved_docs],
            latency_ms=latency_ms,
            token_usage=token_usage,
            extra=extra or {},
        )
        self.writer.write(record)
        return record
