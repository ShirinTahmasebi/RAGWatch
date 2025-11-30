"""Controller layer for Streamlit dashboard events."""
from __future__ import annotations

from typing import Dict, Optional

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
    def add_monitor(self, name: str, ip: str, port: str, interval: int, modules: str) -> None:
        self.monitor_repo.add_monitor(name, ip, port, interval, modules)
        # Force reload next time to pick up the new entry
        self.state[States.MONITOR_LOADED] = False
        self.go_home()


def get_controller() -> DashboardController:
    """Convenience factory to reuse the controller across UI modules."""

    if "_dashboard_controller" not in st.session_state:
        st.session_state["_dashboard_controller"] = DashboardController()
    return st.session_state["_dashboard_controller"]
