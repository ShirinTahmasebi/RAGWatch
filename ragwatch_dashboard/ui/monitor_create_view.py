"""UI for creating a monitor (View layer)."""
from __future__ import annotations

import streamlit as st

from controller.dashboard_controller import DashboardController

LOG_PATH_KEY = "monitor_log_path"
LOG_PATH_PENDING_KEY = "monitor_log_path_pending"
LOG_VERIFIED_KEY = "monitor_log_verified"
LOG_MESSAGE_KEY = "monitor_log_message"


def _rerun() -> None:
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()


def _ensure_state_defaults() -> None:
    state = st.session_state
    state.setdefault(LOG_PATH_KEY, "")
    state.setdefault(LOG_PATH_PENDING_KEY, None)
    state.setdefault(LOG_VERIFIED_KEY, False)
    state.setdefault(LOG_MESSAGE_KEY, "")


def _reset_verification() -> None:
    st.session_state[LOG_VERIFIED_KEY] = False
    st.session_state[LOG_MESSAGE_KEY] = ""


def _handle_log_input_change() -> None:
    _reset_verification()


def _section_divider() -> None:
    st.markdown(
        """
        <hr style="margin:1.25rem 0;border:none;height:1px;background:linear-gradient(90deg,rgba(229,231,235,0),rgba(229,231,235,0.8),rgba(229,231,235,0));" />
        """,
        unsafe_allow_html=True,
    )

def render_monitor_create_form(controller: DashboardController) -> None:
    _ensure_state_defaults()

    pending_value = st.session_state.get(LOG_PATH_PENDING_KEY)
    if pending_value:
        st.session_state[LOG_PATH_KEY] = pending_value
        st.session_state[LOG_PATH_PENDING_KEY] = None

    st.header("Add Monitor")
    st.caption("Provide the source path for logs and (optionally) remote endpoint details.")

    left, right = st.columns(2)
    name = left.text_input("Monitor Name", placeholder="Production QA", key="monitor_form_name")
    interval = right.number_input(
        "Monitoring Interval (minutes)", min_value=1, max_value=60, value=5, key="monitor_form_interval"
    )

    monitoring_modules = st.selectbox(
        "Modules to Monitor",
        ["Retriever", "Generator", "Both"],
        key="monitor_form_modules",
    )

    _section_divider()
    st.subheader("Endpoint Settings")
    use_localhost = st.checkbox(
        "Use local files (localhost)",
        value=True,
        disabled=True,
        help="Remote endpoints coming soon. For now everything runs locally.",
        key="monitor_use_localhost",
    )

    endpoint_left, endpoint_right = st.columns(2)
    ip = endpoint_left.text_input(
        "Server IP Address", value="localhost", disabled=use_localhost, key="monitor_form_ip"
    )
    port = endpoint_right.text_input(
        "Port Number", value="8080", disabled=use_localhost, key="monitor_form_port"
    )

    _section_divider()
    st.subheader("Log File")
    st.caption("Point to the .jsonl file where this monitor writes its logs.")

    try:
        log_cols = st.columns([4, 1], gap="small", vertical_alignment="center")
    except TypeError:  # Streamlit < 1.33 fallback
        log_cols = st.columns([4, 1], gap="small")
    with log_cols[0]:
        log_dir = st.text_input(
            label="Log directory or log file",
            key=LOG_PATH_KEY,
            placeholder="/Users/me/RAGWatch/logs/hotpotqa",
            label_visibility="collapsed",
            on_change=_handle_log_input_change,
        )
    with log_cols[1]:
        verify_disabled = not bool(log_dir)
        verify_clicked = st.button(
            "Verify",
            key="log_path_verify",
            disabled=verify_disabled,
            use_container_width=True,
        )

    if log_dir is None:
        log_dir = ""

    if 'verify_clicked' in locals() and verify_clicked:
        try:
            ok, message, resolved = controller.verify_log_path(log_dir)
        except Exception as exc:  # pragma: no cover - defensive guard
            ok = False
            message = f"Verification failed: {exc}"
            resolved = log_dir
        st.session_state[LOG_VERIFIED_KEY] = ok
        st.session_state[LOG_MESSAGE_KEY] = message
        if ok:
            st.session_state[LOG_PATH_PENDING_KEY] = resolved
            _rerun()

    if st.session_state[LOG_MESSAGE_KEY]:
        color = "#16a34a" if st.session_state[LOG_VERIFIED_KEY] else "#dc2626"
        st.markdown(
            f"<span style='color:{color}; font-size:0.9rem;'>{st.session_state[LOG_MESSAGE_KEY]}</span>",
            unsafe_allow_html=True,
        )

    _section_divider()

    can_submit = all([name, log_dir, monitoring_modules]) and st.session_state[LOG_VERIFIED_KEY]

    submit_cols = st.columns([1, 1])
    create_clicked = submit_cols[0].button(
        "Create monitor",
        type="primary",
        use_container_width=True,
        disabled=not can_submit,
        key="monitor_create_button",
    )
    cancel_clicked = submit_cols[1].button("Back to dashboard", use_container_width=True, key="monitor_cancel_button")

    if cancel_clicked:
        controller.go_home()
        _rerun()

    if create_clicked:
        controller.add_monitor(name, ip, port, int(interval), monitoring_modules, log_dir, use_localhost)
        st.success(f"{name} is now registered and ready for dashboards.")
        st.balloons()
        controller.go_home()
        _rerun()
