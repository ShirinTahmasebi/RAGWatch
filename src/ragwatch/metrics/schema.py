"""Data models for KPI results and reports."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class KPIResult:
    """A single KPI measurement from a RAG run."""

    name: str
    value: float | int | str | bool | None
    category: str
    stage: str
    source: str
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class KPIReport:
    """A collection of KPI results for a single RAG run or database report."""

    run_id: str | None
    query: str | None
    results: list[KPIResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
