"""UI layer: top-level dashboard layout and navigation."""
from __future__ import annotations

import statistics

import streamlit as st

from controller.dashboard_controller import DashboardController, get_controller
from utils.constants import Fields, Pages, States

from .monitor_create_view import render_monitor_create_form
from .monitor_details_view import render_monitor_details

_PAGE_CONFIG_KEY = "_dashboard_page_config"


def _ensure_page_config() -> None:
    if not st.session_state.get(_PAGE_CONFIG_KEY):
        st.set_page_config(page_title="RAGWatch Dashboard", layout="wide")
        st.session_state[_PAGE_CONFIG_KEY] = True


def _render_sidebar(controller: DashboardController) -> None:
    monitors = controller.list_monitors()
    selected_monitor_id = controller.state.get(States.SELECT_MONITOR_ID)

    st.sidebar.button(
        "Add Monitor",
        key="btn_add_monitor",
        icon=":material/add:",
        use_container_width=True,
        type="secondary",
        on_click=controller.open_create_monitor,
    )

    st.sidebar.markdown(
        """
        <hr style="margin:0.75rem 0;border:none;height:1px;background:linear-gradient(90deg,rgba(229,231,235,0),rgba(229,231,235,0.8),rgba(229,231,235,0));" />
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Monitors")

    if not monitors:
        st.sidebar.info("No monitors yet. Click 'Add Monitor' to create one.")
        return

    for monitor_id, monitor_data in monitors.items():
        monitor_name = monitor_data.get(Fields.NAME, monitor_id)
        is_selected = selected_monitor_id == monitor_id
        label = monitor_name

        st.sidebar.button(
            label,
            key=f"monitor_item_{monitor_id}",
            type="primary" if is_selected else "secondary",
            use_container_width=True,
            on_click=controller.open_monitor_details,
            kwargs={"monitor_id": monitor_id},
        )


def _render_home(controller: DashboardController) -> None:
    st.title("RAGWatch Dashboard")
    st.write("Select a monitor from the sidebar or create a new one to get started.")

    monitors = controller.list_monitors()
    total_monitors = len(monitors)
    intervals = [int(item.get(Fields.INTERNAL, 0)) for item in monitors.values() if str(item.get(Fields.INTERNAL, "")).isdigit()]
    avg_interval = statistics.mean(intervals) if intervals else None

    col1, col2 = st.columns(2)
    with col1:
        st.write(
            f"""
            <div style='border:1px solid #e5e7eb;border-radius:16px;padding:1rem;background-color:#fafafa;'>
                <div style='font-size:0.8rem;text-transform:uppercase;color:#6b7280;'>Active monitors</div>
                <div style='font-size:2rem;font-weight:600;margin:0.4rem 0;'>{total_monitors}</div>
                <div style='color:#4b5563;'>Monitors currently streaming KPIs.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        avg_display = f"{avg_interval:.0f}" if avg_interval else "—"
        st.write(
            f"""
            <div style='border:1px solid #e5e7eb;border-radius:16px;padding:1rem;background-color:#fafafa;'>
                <div style='font-size:0.8rem;text-transform:uppercase;color:#6b7280;'>Average interval (min)</div>
                <div style='font-size:2rem;font-weight:600;margin:0.4rem 0;'>{avg_display}</div>
                <div style='color:#4b5563;'>Mean schedule across active monitors.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_dashboard(controller: DashboardController | None = None) -> None:
    controller = controller or get_controller()

    _ensure_page_config()
    _render_sidebar(controller)

    page = controller.current_page()
    if page == Pages.monitor_create:
        render_monitor_create_form(controller)
    elif page == Pages.monitor_details:
        render_monitor_details(controller)
    else:
        _render_home(controller)
