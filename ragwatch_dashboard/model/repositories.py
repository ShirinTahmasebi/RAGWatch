"""Model layer for the Streamlit dashboard (data access + mutations)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import time

import pandas as pd

from utils.constants import Fields, Paths


@dataclass
class Monitor:
    monitor_id: str
    name: str
    ip: str
    port: str
    interval: int
    modules: str


class MonitorRepository:
    """Read/write access to monitor metadata stored in CSV format."""

    def __init__(self, monitor_csv: str = Paths.CSV_MONITOR) -> None:
        self.monitor_csv = Path(monitor_csv)
        self.monitor_csv.parent.mkdir(parents=True, exist_ok=True)

    def load_all(self) -> Dict[str, dict]:
        if not self.monitor_csv.exists():
            return {}
        df = pd.read_csv(self.monitor_csv)
        if df.empty:
            return {}
        return df.set_index(Fields.MONITOR_ID).to_dict(orient="index")

    def add_monitor(self, name: str, ip: str, port: str, interval: int, monitoring_modules: str) -> Monitor:
        new_monitor = Monitor(
            monitor_id=f"{time.time()}-{name}",
            name=name,
            ip=ip,
            port=str(port),
            interval=int(interval),
            modules=monitoring_modules,
        )

        new_row = {
            Fields.MONITOR_ID: new_monitor.monitor_id,
            Fields.NAME: new_monitor.name,
            Fields.IP: new_monitor.ip,
            Fields.PORT: new_monitor.port,
            Fields.INTERNAL: new_monitor.interval,
            Fields.MONITORING_MODULES: new_monitor.modules,
        }

        if self.monitor_csv.exists():
            df = pd.read_csv(self.monitor_csv)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(self.monitor_csv, index=False)
        return new_monitor


class KPIRepository:
    """Loader for KPI history per monitor."""

    def __init__(self, base_dir: str = Paths.CSV_KPIS) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_for_monitor(self, monitor_id: str) -> pd.DataFrame:
        file_path = self.base_dir / f"monitor_{monitor_id}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(file_path)
        except Exception:  # pragma: no cover - Streamlit will warn users instead
            return pd.DataFrame()


class AlertRepository:
    """Loader for alert history per monitor."""

    def __init__(self, base_dir: str = Paths.CSV_ALERTS) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_for_monitor(self, monitor_id: str) -> pd.DataFrame:
        file_path = self.base_dir / f"alert_{monitor_id}.csv"
        if not file_path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(file_path)
        except Exception:
            return pd.DataFrame()
