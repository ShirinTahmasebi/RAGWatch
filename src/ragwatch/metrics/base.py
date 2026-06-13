"""Base metric interface for KPI calculations."""

from abc import ABC, abstractmethod

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.catalog import KPI_CATALOG, KPIId
from ragwatch.metrics.schema import KPIResult


class BaseMetric(ABC):
    """Abstract base class for deterministic RAG metrics computed from a RAGRun."""

    kpi_id: KPIId

    @property
    def name(self) -> str:
        return str(self.kpi_id.value)

    @property
    def category(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].category)

    @property
    def stage(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].stage)

    @property
    def source(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].source)

    @property
    def description(self) -> str:
        return KPI_CATALOG[self.kpi_id].description

    @abstractmethod
    def compute(self, run: RAGRun) -> KPIResult:
        """Compute this metric from a RAGRun and return a KPIResult."""
        ...


class BaseDBMetric(ABC):
    """Abstract base class for metrics computed from database tables."""

    kpi_id: KPIId

    @property
    def name(self) -> str:
        return str(self.kpi_id.value)

    @property
    def category(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].category)

    @property
    def stage(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].stage)

    @property
    def source(self) -> str:
        return str(KPI_CATALOG[self.kpi_id].source)

    @property
    def description(self) -> str:
        return KPI_CATALOG[self.kpi_id].description

    @abstractmethod
    def compute(self, connection_string: str) -> KPIResult:
        """Compute this metric from database tables and return a KPIResult."""
        ...
