"""Writers responsible for persisting RAGWatch run records."""
from __future__ import annotations

from pathlib import Path

from ..models import RAGRunRecord


class JSONLWriter:
    """Append `RAGRunRecord` entries to a newline-delimited JSON log file."""

    def __init__(self, log_path: str):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: RAGRunRecord) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(record.model_dump_json() + "\n")
