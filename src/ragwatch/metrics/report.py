"""KPI engine for computing and reporting metrics from RAG runs and databases."""

from typing import Any

from ragwatch.core.schema import RAGRun
from ragwatch.metrics.base import BaseDBMetric, BaseMetric
from ragwatch.metrics.generation import DEFAULT_GENERATION_METRICS
from ragwatch.metrics.retrieval import DEFAULT_RETRIEVAL_METRICS
from ragwatch.metrics.runtime import DEFAULT_RUNTIME_METRICS
from ragwatch.metrics.schema import KPIReport, KPIResult


class KPIEngine:
    """Compute a set of metrics from a RAGRun and produce a KPIReport."""

    def __init__(self, metrics: list[BaseMetric] | None = None) -> None:
        if metrics is None:
            self._metrics: list[BaseMetric] = (
                DEFAULT_RETRIEVAL_METRICS
                + DEFAULT_GENERATION_METRICS
                + DEFAULT_RUNTIME_METRICS
            )
        else:
            self._metrics = metrics

    @property
    def metrics(self) -> list[BaseMetric]:
        """Return the list of configured metrics."""
        return self._metrics

    def compute(self, run: RAGRun) -> KPIReport:
        """Compute all metrics for a RAGRun and return a KPIReport."""
        results: list[KPIResult] = []
        for metric in self._metrics:
            results.append(metric.compute(run))
        return KPIReport(
            run_id=run.run_id,
            query=run.query,
            results=results,
        )


def to_dict(report: KPIReport) -> dict[str, Any]:
    """Convert a KPIReport to a nested dictionary."""
    return {
        "run_id": report.run_id,
        "query": report.query,
        "results": [
            {
                "name": r.name,
                "value": r.value,
                "category": r.category,
                "stage": r.stage,
                "source": r.source,
                "description": r.description,
                "metadata": r.metadata,
            }
            for r in report.results
        ],
        "metadata": report.metadata,
    }


def to_flat_dict(report: KPIReport) -> dict[str, Any]:
    """Convert a KPIReport to a flat dictionary suitable for CSV export."""
    flat: dict[str, Any] = {
        "run_id": report.run_id,
        "query": report.query,
    }
    for r in report.results:
        flat[r.name] = r.value
    return flat


def merge_kpi_reports(*reports: KPIReport) -> KPIReport:
    """Merge multiple KPIReports into one, concatenating their results.

    The merged report uses the ``run_id`` and ``query`` of the first report and
    combines all metadata. This is useful for combining deterministic and
    semantic KPI reports for the same run.
    """
    if not reports:
        raise ValueError("merge_kpi_reports requires at least one report.")

    results: list[KPIResult] = []
    metadata: dict[str, Any] = {}
    for report in reports:
        results.extend(report.results)
        metadata.update(report.metadata)

    return KPIReport(
        run_id=reports[0].run_id,
        query=reports[0].query,
        results=results,
        metadata=metadata,
    )



class DBKPIEngine:
    """Compute database-level KPIs from pgvector/Postgres tables."""

    def __init__(self, metrics: list[BaseDBMetric] | None = None) -> None:
        if metrics is None:
            from ragwatch.metrics.db_stats import DEFAULT_DB_METRICS

            self._metrics: list[BaseDBMetric] = DEFAULT_DB_METRICS
        else:
            self._metrics = metrics

    @property
    def metrics(self) -> list[BaseDBMetric]:
        """Return the list of configured DB metrics."""
        return self._metrics

    def compute(self, connection_string: str | None = None) -> KPIReport:
        """Compute all DB metrics and return a KPIReport.

        Args:
            connection_string: PostgreSQL connection string. If None, reads
                from RAGWATCH_PGVECTOR_URL environment variable.
        """
        if connection_string is None:
            from ragwatch.config.env import get_env, load_env

            load_env()
            connection_string = get_env("RAGWATCH_PGVECTOR_URL", required=True)

        results: list[KPIResult] = []
        for metric in self._metrics:
            results.append(metric.compute(connection_string))  # type: ignore[arg-type]

        return KPIReport(
            run_id=None,
            query=None,
            results=results,
            metadata={"source": "database"},
        )
