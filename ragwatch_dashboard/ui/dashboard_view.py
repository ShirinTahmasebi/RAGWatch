"""UI layer: top-level dashboard layout and navigation."""
from __future__ import annotations

import streamlit as st

from controller.dashboard_controller import DashboardController, get_controller
from utils.constants import Fields, Pages, States

from .monitor_create_view import render_monitor_create_form
from .monitor_details_view import render_monitor_details


def _render_sidebar(controller: DashboardController) -> None:
    st.sidebar.header("Monitors")
    st.sidebar.button(
        "Add Monitor",
        key="btn_add_monitor",
        icon=":material/add:",
        use_container_width=True,
        type="primary",
        on_click=controller.open_create_monitor,
    )

    monitors = controller.list_monitors()
    selected_monitor_id = controller.state.get(States.SELECT_MONITOR_ID)

    if not monitors:
        st.sidebar.info("No monitors yet. Click 'Add Monitor' to create one.")
        return

    for monitor_id, monitor_data in monitors.items():
        monitor_name = monitor_data.get(Fields.NAME, monitor_id)
        if selected_monitor_id == monitor_id:
            label = f':red[**{monitor_name}**]'
        else:
            label = monitor_name

        st.sidebar.button(
            label,
            key=f"monitor_item_{monitor_id}",
            type="secondary",
            use_container_width=True,
            on_click=controller.open_monitor_details,
            kwargs={"monitor_id": monitor_id},
        )


def render_dashboard(controller: DashboardController | None = None) -> None:
    controller = controller or get_controller()

    _render_sidebar(controller)

    page = controller.current_page()
    if page == Pages.monitor_create:
        render_monitor_create_form(controller)
    elif page == Pages.monitor_details:
        render_monitor_details(controller)
    else:
        st.title("RAGWatch Dashboard")
        st.write("Select a monitor from the sidebar or create a new one to get started.")
