"""UI for creating a monitor (View layer)."""
from __future__ import annotations

import streamlit as st

from controller.dashboard_controller import DashboardController


def _open_success_dialog(controller: DashboardController, name, ip, port, interval, monitoring_modules):
    @st.dialog("Monitor created")
    def _dialog():
        st.success(f"{name} is now registered.")
        st.write(f"Endpoint: {ip}:{port}")
        st.write(f"Interval: every {interval} minutes")
        st.write(f"Modules: {monitoring_modules}")

        if st.button("Back to dashboard", use_container_width=True, type="primary"):
            controller.add_monitor(name, ip, port, interval, monitoring_modules)
            st.rerun()

    _dialog()


def render_monitor_create_form(controller: DashboardController) -> None:
    st.header("Add monitor")
    st.caption("Provide the endpoint details so we can track KPIs and alerts for it.")

    with st.form(key="form_monitor_create"):
        left, right = st.columns(2)
        name = left.text_input("Monitor Name", placeholder="Production QA")
        interval = right.number_input("Monitoring Interval (minutes)", min_value=1, max_value=60, value=5)

        ip = left.text_input("Server IP Address", placeholder="10.0.12.34")
        port = right.text_input("Port Number", placeholder="8080")

        monitoring_modules = st.selectbox("Modules to Monitor", ["Retriever", "Generator", "Both"])

        submit_button = st.form_submit_button("Create monitor", use_container_width=True)

        if submit_button:
            if not all([name, interval, ip, port, monitoring_modules]):
                st.warning("Please fill in all the fields.")
            elif not port.isdigit():
                st.warning("Port must be a numeric value.")
            else:
                _open_success_dialog(controller, name, ip, port, interval, monitoring_modules)
