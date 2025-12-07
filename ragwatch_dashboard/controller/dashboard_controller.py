"""Controller layer for Streamlit dashboard events."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st

from utils.constants import Pages, States
from .state_adapter import SessionStateAdapter
from model.repositories import AlertRepository, KPIRepository, MonitorRepository


class DashboardController:
    """Coordinates UI events with the data layer."""

    def __init__(
        self,
        state: Optional[SessionStateAdapter] = None,
        monitor_repo: Optional[MonitorRepository] = None,
        kpi_repo: Optional[KPIRepository] = None,
        alert_repo: Optional[AlertRepository] = None,
    ) -> None:
        self.state = state or SessionStateAdapter(st.session_state)
        self.monitor_repo = monitor_repo or MonitorRepository()
        self.kpi_repo = kpi_repo or KPIRepository()
        self.alert_repo = alert_repo or AlertRepository()

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------
    def current_page(self) -> str:
        return self.state.get(States.CURRENT_PAGE, Pages.home)

    def navigate_to(self, page: str) -> None:
        self.state[States.CURRENT_PAGE] = page

    def open_create_monitor(self) -> None:
        self.state[States.SELECT_MONITOR_ID] = None
        self.navigate_to(Pages.monitor_create)

    def open_monitor_details(self, monitor_id: str) -> None:
        self.state[States.SELECT_MONITOR_ID] = monitor_id
        self.navigate_to(Pages.monitor_details)

    def go_home(self) -> None:
        self.state[States.SELECT_MONITOR_ID] = None
        self.navigate_to(Pages.home)

    # ------------------------------------------------------------------
    # Monitor caching
    # ------------------------------------------------------------------
    def _load_monitors_if_needed(self) -> Dict[str, dict]:
        if not self.state.get(States.MONITOR_LOADED, False):
            monitors = self.monitor_repo.load_all()
            self.state[States.MONITOR_DICT] = monitors
            self.state[States.MONITOR_LOADED] = True
        return self.state.get(States.MONITOR_DICT, {})

    def list_monitors(self) -> Dict[str, dict]:
        return self._load_monitors_if_needed()

    def get_monitor(self, monitor_id: Optional[str] = None) -> Optional[dict]:
        monitors = self._load_monitors_if_needed()
        target = monitor_id or self.state.get(States.SELECT_MONITOR_ID)
        if not target:
            return None
        return monitors.get(target)

    # ------------------------------------------------------------------
    # Data access wrappers
    # ------------------------------------------------------------------
    def get_kpis(self, monitor_id: str):
        return self.kpi_repo.load_for_monitor(monitor_id)

    def get_alerts(self, monitor_id: str):
        return self.alert_repo.load_for_monitor(monitor_id)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------
    def add_monitor(
        self,
        name: str,
        ip: str,
        port: str,
        interval: int,
        modules: str,
        log_dir: str,
        use_localhost: bool,
    ) -> None:
        resolved_path = str(Path(log_dir).expanduser())
        self.monitor_repo.add_monitor(name, ip, port, interval, modules, resolved_path, use_localhost)
        # Force reload next time to pick up the new entry
        self.state[States.MONITOR_LOADED] = False
        self.go_home()

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def verify_log_path(self, path_value: str) -> Tuple[bool, str, str]:
        """Check whether the provided log path is readable.

        Returns (is_valid, message, resolved_path).
        """

        if not path_value:
            return False, "Please select a log directory or file.", ""

        candidate = Path(path_value).expanduser()
        resolved = str(candidate)

        if not candidate.exists():
            return False, f"Path '{resolved}' does not exist.", resolved

        if not os.access(candidate, os.R_OK):
            return False, f"Missing read permissions for '{resolved}'.", resolved

        if candidate.is_file():
            try:
                with candidate.open("rb") as handle:
                    handle.read(1)
            except OSError as exc:
                return False, f"Unable to read file: {exc}", resolved
            return True, f"File '{candidate.name}' is readable.", resolved

        # Directory case
        try:
            entries = list(candidate.iterdir())
        except OSError as exc:
            return False, f"Unable to list directory: {exc}", resolved

        jsonl_files = [entry for entry in entries if entry.suffix == ".jsonl"]
        if jsonl_files:
            example = jsonl_files[0].name
            return True, f"Directory readable (found sample log '{example}').", resolved
        return True, "Directory is readable (no .jsonl files detected yet).", resolved


def get_controller() -> DashboardController:
    """Convenience factory to reuse the controller across UI modules."""

    if "_dashboard_controller" not in st.session_state:
        st.session_state["_dashboard_controller"] = DashboardController()
    return st.session_state["_dashboard_controller"]
