"""High-level logging interface for RAGWatch."""
from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List, Optional

from .schema import RetrievedDoc, RunRecord
from .writers import JSONLWriter


class RAGWatchLogger:
    """Convenience wrapper that records RAG runs to a JSONL log."""

    def __init__(self, log_dir: str, dataset_name: str, pipeline_name: str):
        self.dataset_name = dataset_name
        self.pipeline_name = pipeline_name
        log_file = f"{log_dir}/{dataset_name}_{pipeline_name}.jsonl"
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
    ) -> RunRecord:
        record = RunRecord(
            run_id=self.new_run_id(),
            session_id=session_id,
            timestamp=dt.datetime.utcnow(),
            dataset_name=self.dataset_name,
            pipeline_name=self.pipeline_name,
            question=question,
            answer=answer,
            retrieved_docs=[RetrievedDoc(**doc) for doc in retrieved_docs],
            latency_ms=latency_ms,
            token_usage=token_usage,
            extra=extra or {},
        )
        self.writer.write(record)
        return record
