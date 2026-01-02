"""Background job manager that computes KPIs for every registered monitor."""
from __future__ import annotations

import csv
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pydantic import ValidationError

from ragwatch.models import RAGRunRecord
from ragwatch.utils import resolve_kpi_dir, resolve_monitor_csv

from ragwatch.analytics.kpi_calculations import KPICalculationFn, KPI_CALCULATION_REGISTRY
from ragwatch.analytics.kpis import KPI_CONFIGS

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class MonitorRecord:
    """Normalized representation of a monitor row from the CSV registry."""

    monitor_id: str
    name: str
    interval_minutes: int
    log_path: str
    modules: str
    ip: str | None
    port: str | None
    use_localhost: bool
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def sanitized_id(self) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in self.monitor_id)


class KPICalculator:
    """Wrapper that fans out KPI calculations and handles missing implementations."""

    def __init__(self, registry: Mapping[str, KPICalculationFn] | None = None):
        registry = registry or KPI_CALCULATION_REGISTRY
        missing = set(KPI_CONFIGS.keys()) - set(registry.keys())
        if missing:
            raise ValueError(f"Missing KPI functions for: {', '.join(sorted(missing))}")
        self._registry = dict(registry)

    def compute(
        self,
        *,
        window_records: Sequence[RAGRunRecord],
        monitor_meta: Mapping[str, Any],
        interval_minutes: int,
        window_end: datetime,
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for key, func in self._registry.items():
            try:
                results[key] = func(
                    window_records,
                    monitor_meta=monitor_meta,
                    interval_minutes=interval_minutes,
                    window_end=window_end,
                )
            except NotImplementedError:
                results[key] = None
            except Exception as exc:  # pragma: no cover - defensive
                monitor_id = monitor_meta.get("monitor_id", "<unknown>")
                logger.exception("Failed to compute KPI %s for %s: %s", key, monitor_id, exc)
                results[key] = None
        return results


@dataclass
class MonitorKPIJob:
    """Independent job responsible for a single monitor's KPI CSV."""

    monitor: MonitorRecord
    output_dir: Path
    calculator: KPICalculator
    last_bucket_end: datetime | None = None

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = self.output_dir / f"monitor_{self.monitor.sanitized_id}.csv"
        self.last_bucket_end = _load_last_kpi_timestamp(self.output_path)

    @property
    def interval(self) -> timedelta:
        minutes = max(self.monitor.interval_minutes, 1)
        # TODO Shirin: Change back to minutes
        return timedelta(seconds=minutes)

    def run_if_due(self, now: datetime) -> bool:
        if not self.monitor.log_path:
            logger.debug("Monitor %s has no log path; skipping", self.monitor.monitor_id)
            return False

        logger.info("------------------------------------------------")
        logger.info("Current time: %s", _normalize_timestamp(now))
        logger.info("Last KPI timestamp for monitor %s: %s", self.monitor.monitor_id, self.last_bucket_end)
        
        records = _load_log_records(self.monitor.log_path)
        if not records:
            logger.debug("No log records found for monitor %s", self.monitor.monitor_id)
            return False

        due_buckets = self._pending_buckets(records, now, self.last_bucket_end)
        if not due_buckets:
            return False

        wrote_any = False        
        for bucket_end, bucket_records in due_buckets:
            
            kpi_payload = self.calculator.compute(
                window_records=bucket_records,
                monitor_meta=self.monitor.raw,
                interval_minutes=self.monitor.interval_minutes,
                window_end=bucket_end,
            )

            row = {
                "timestamp": bucket_end.isoformat(),
                "monitor_id": self.monitor.monitor_id,
                "monitor_name": self.monitor.name,
                "interval_minutes": self.monitor.interval_minutes,
                "monitoring_modules": self.monitor.modules,
                "log_source": self.monitor.log_path,
                **kpi_payload,
            }
            
            _append_row(self.output_path, row)
            wrote_any = True
            
            if self.last_bucket_end is None or bucket_end > self.last_bucket_end:
                self.last_bucket_end = bucket_end

        return wrote_any


    def _pending_buckets(
        self, records: Sequence[RAGRunRecord], now: datetime, last_bucket_end: datetime | None
    ) -> List[tuple[datetime, List[RAGRunRecord]]]:
        buckets = _group_records_by_bucket(records, self.interval)
        pending: List[tuple[datetime, List[RAGRunRecord]]] = []
        for bucket_end, bucket_records in buckets:
            if last_bucket_end and bucket_end <= last_bucket_end:
                continue
            if bucket_end > now:
                break
            logger.info("Monitor %s: pending bucket at %s with %d records", self.monitor.monitor_id, bucket_end, len(bucket_records))
            pending.append((bucket_end, bucket_records))
        return pending


class KPIComputationService:
    """Orchestrates KPI jobs for all monitors recorded in the CSV registry."""

    def __init__(
        self,
        *,
        monitors_csv: str | None = None,
        kpi_dir: str | None = None,
        calculator: KPICalculator | None = None,
        poll_interval_seconds: int = 30,
    ) -> None:
        self.monitors_csv = Path(monitors_csv or resolve_monitor_csv())
        self.kpi_dir = Path(kpi_dir or resolve_kpi_dir())
        self.kpi_dir.mkdir(parents=True, exist_ok=True)
        self.calculator = calculator or KPICalculator()
        self.poll_interval_seconds = max(poll_interval_seconds, 1)
        self.jobs = self._load_jobs()

    def _load_jobs(self) -> List[MonitorKPIJob]:
        monitors = _load_monitors(self.monitors_csv)
        jobs = [
            MonitorKPIJob(monitor=monitor, output_dir=self.kpi_dir, calculator=self.calculator)
            for monitor in monitors
            if monitor.log_path
        ]
        if not jobs:
            logger.warning("No KPI jobs initialized. Check monitor CSV at %s", self.monitors_csv)
        return jobs

    def run_once(self, *, now: datetime | None = None) -> bool:
        snapshot = now or datetime.utcnow()
        ran_any = False
        for job in self.jobs:
            if job.run_if_due(snapshot):
                ran_any = True
        return ran_any

    def run_forever(self) -> None:  # pragma: no cover - long-running loop
        logger.info(
            "Starting KPI computation service with %d jobs (poll every %d s)",
            len(self.jobs),
            self.poll_interval_seconds,
        )
        while True:
            self.run_once()
            time.sleep(self.poll_interval_seconds)


def _load_monitors(path: Path) -> List[MonitorRecord]:
    if not path.exists():
        logger.warning("Monitor CSV %s does not exist", path)
        return []

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader]

    monitors: List[MonitorRecord] = []
    for row in rows:
        monitor_id = row.get("monitor_id") or row.get("id")
        if not monitor_id:
            continue
        interval_raw = row.get("interval") or row.get("monitoring_interval") or 5
        try:
            interval = max(int(interval_raw), 1)
        except (TypeError, ValueError):
            interval = 5
        log_path = (row.get("log_dir") or "").strip()
        monitor = MonitorRecord(
            monitor_id=str(monitor_id),
            name=(row.get("name") or str(monitor_id)).strip(),
            interval_minutes=interval,
            log_path=log_path,
            modules=(row.get("monitoring_modules") or "Both").strip(),
            ip=(row.get("ip") or row.get("host")),
            port=row.get("port"),
            use_localhost=_to_bool(row.get("use_localhost")),
            raw=row,
        )
        monitors.append(monitor)
    return monitors


def _load_log_records(log_location: str) -> List[RAGRunRecord]:
    path = Path(log_location).expanduser()
    files: List[Path]
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        files = [path]

    records: List[RAGRunRecord] = []
    for file_path in files:
        if not file_path.exists():
            continue
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = line.strip()
                if not payload:
                    continue
                try:
                    record = RAGRunRecord.model_validate_json(payload)
                except ValidationError as exc:
                    logger.warning("Skipping malformed log entry in %s: %s", file_path, exc)
                    continue
                records.append(record)
    records.sort(key=lambda record: record.timestamp)
    return records


def _read_last_csv_row(path: Path) -> Dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        last_row: Dict[str, Any] | None = None
        for row in reader:
            last_row = row
    return last_row


def _load_last_kpi_timestamp(path: Path) -> datetime | None:
    last_row = _read_last_csv_row(path)
    if not last_row:
        return None

    timestamp_str = (last_row.get("timestamp") or "").strip()
    if not timestamp_str:
        return None

    try:
        parsed = datetime.fromisoformat(timestamp_str)
        return _normalize_timestamp(parsed)
    except ValueError:
        logger.warning("Invalid timestamp '%s' in %s", timestamp_str, path)
        return None


def _group_records_by_bucket(
    records: Sequence[RAGRunRecord], interval: timedelta
) -> List[tuple[datetime, List[RAGRunRecord]]]:
    buckets: Dict[datetime, List[RAGRunRecord]] = defaultdict(list)
    for record in records:
        bucket_end = _bucket_end(record.timestamp, interval)
        buckets[bucket_end].append(record)
    return sorted(buckets.items(), key=lambda item: item[0])


UTC = timezone.utc
EPOCH = datetime(1970, 1, 1)


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _bucket_end(timestamp: datetime, interval: timedelta) -> datetime:
    normalized = _normalize_timestamp(timestamp)
    seconds = max(interval.total_seconds(), 1)
    elapsed = (normalized - EPOCH).total_seconds()
    bucket_index = math.floor(elapsed / seconds)
    bucket_start = EPOCH + timedelta(seconds=bucket_index * seconds)
    return bucket_start + interval


def _append_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    write_header = not file_exists or path.stat().st_size == 0

    if not write_header:
        last_row = _read_last_csv_row(path)
        last_timestamp = (last_row or {}).get("timestamp") if last_row else None
        if last_timestamp == row.get("timestamp"):
            logger.debug("Skipping duplicate KPI row for %s", last_timestamp)
            return

    fieldnames = [
        "timestamp",
        "monitor_id",
        "monitor_name",
        "interval_minutes",
        "monitoring_modules",
        "log_source",
        *KPI_CONFIGS.keys(),
    ]

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        logger.info("Writing row to %s", path)
        writer.writerow(row)


def main() -> None:  # pragma: no cover - CLI helper
    service = KPIComputationService()
    service.run_forever()


if __name__ == "__main__":  # pragma: no cover
    main()
