"""UI for creating a monitor (View layer)."""
from __future__ import annotations

import streamlit as st

from controller.dashboard_controller import DashboardController


def _open_success_dialog(controller: DashboardController, name, ip, port, interval, monitoring_modules):
    @st.dialog("Success")
    def _dialog():
        st.write(f"Monitor :blue-background[**\"{name}\"**] will appear in the sidebar.")
        st.write(f"Server address: :blue-background[**{ip}:{port}**]")
        st.write(f"Monitoring interval: :blue-background[**{interval}**] minutes.")
        st.write("")

        _, middle, _ = st.columns(3)
        if middle.button(":green[**OK**]", use_container_width=True, type="tertiary"):
            controller.add_monitor(name, ip, port, interval, monitoring_modules)
            st.rerun()

    _dialog()


def render_monitor_create_form(controller: DashboardController) -> None:
    st.header("Add Monitor")

    with st.form(key="form_monitor_create"):
        name = st.text_input("Monitor Name")
        interval = st.number_input("Monitoring Interval (minutes)", min_value=1, max_value=60, value=5)
        ip = st.text_input("Server IP Address")
        port = st.text_input("Port Number")
        monitoring_modules = st.selectbox("Modules to Monitor", ["Retriever", "Generator", "Both"])

        _, _, right = st.columns(3)
        submit_button = right.form_submit_button("Create Monitor", use_container_width=True)

        if submit_button:
            if not all([name, interval, ip, port, monitoring_modules]):
                st.warning("Please fill in all the fields.")
            elif not port.isdigit():
                st.warning("Port must be a numeric value.")
            else:
                _open_success_dialog(controller, name, ip, port, interval, monitoring_modules)
