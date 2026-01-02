"""UI for displaying monitor details (View layer)."""
from __future__ import annotations

import html
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import streamlit as st

from controller.dashboard_controller import DashboardController
from ragwatch.analytics import KPI_CONFIGS, KPI_GROUPS, KPIConfig, KPIGroup
from utils.constants import Fields, States
from utils.enums import AlertType


def render_monitor_details(controller: DashboardController) -> None:
    selected_monitor_id = controller.state.get(States.SELECT_MONITOR_ID)
    monitor = controller.get_monitor(selected_monitor_id)

    if not monitor or not selected_monitor_id:
        st.info("Select a monitor from the sidebar to see its KPIs and alerts.")
        return

    st.button(
        "←  Back to dashboard",
        key="btn_back_to_home",
        type="secondary",
        on_click=controller.go_home,
    )

    log_dir = monitor.get(Fields.LOG_DIR) or "—"
    log_dir_display = html.escape(str(log_dir))
    st.header(monitor.get(Fields.NAME, "Monitor"))
    st.caption(f"{monitor.get(Fields.IP, '-')}: {monitor.get(Fields.PORT, '-')} • every {monitor.get(Fields.INTERNAL, '-') } minutes • {log_dir_display}")

        
    _render_static_details(monitor)

    kpi_tab, alerts_tab = st.tabs(["KPI trends", "Alerts"])
    with kpi_tab:
        _render_kpis(controller, selected_monitor_id)
    with alerts_tab:
        _render_alerts(controller, selected_monitor_id)


def _render_static_details(monitor: dict) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Interval (min)", monitor.get(Fields.INTERNAL, "-"))
    col2.metric("Modules", monitor.get(Fields.MONITORING_MODULES, "-"))
    col3.metric("IP", monitor.get(Fields.IP, "-"))
    col4.metric("Port", monitor.get(Fields.PORT, "-"))


def _grouped_kpis() -> List[Tuple[KPIGroup, List[KPIConfig]]]:
    grouped: List[Tuple[KPIGroup, List[KPIConfig]]] = []
    configs_in_order = list(KPI_CONFIGS.values())
    for group in KPI_GROUPS:
        configs = [cfg for cfg in configs_in_order if cfg.group == group.key]
        if configs:
            grouped.append((group, configs))
    return grouped


def _render_kpis(controller: DashboardController, monitor_id: str | None) -> None:
    if not monitor_id:
        st.info("No monitor selected.")
        return

    df = controller.get_kpis(monitor_id)
    if df.empty:
        st.info("No KPI data recorded yet.")
        return

    df[Fields.TIMESTAMP] = pd.to_datetime(df[Fields.TIMESTAMP], errors="coerce")
    df = df.set_index(Fields.TIMESTAMP)

    for group, group_kpis in _grouped_kpis():
        with st.expander(group.label):
            for kpi in group_kpis:
                card = st.container(border=True)
                card.markdown(f"**{kpi.label}**")
                card.caption(kpi.description)

                if kpi.key not in df.columns:
                    card.info("No datapoints yet.")
                    continue

                series = df[kpi.key].dropna()
                if series.empty:
                    card.info("No datapoints yet.")
                else:
                    card.line_chart(
                        data=series,
                        y_label=kpi.label,
                        x_label="Timestamp",
                        use_container_width=True,
                        height=250,
                    )


def _render_alerts(controller: DashboardController, monitor_id: str | None) -> None:
    if not monitor_id:
        st.info("No monitor selected.")
        return

    alert_df = controller.get_alerts(monitor_id)
    if alert_df.empty:
        st.success("No alerts for this monitor yet.")
        return

    for group, group_kpis in _grouped_kpis():
        with st.expander(group.label):
            for kpi in group_kpis:
                filtered_df = alert_df[alert_df[Fields.KPI] == kpi.key]
                with st.expander(kpi.label):
                    if filtered_df.empty:
                        st.write("No alerts for this KPI.")
                        continue
                    _render_alert_table(filtered_df)


def _render_alert_table(df: pd.DataFrame) -> None:
    con = st.container(border=False)
    alert_colors = {
        AlertType.INCREASE.value: ("#fff3cd88", "#ffc107"),
        AlertType.DROP.value: ("#d0b4e085", "#7f28a7"),
        AlertType.SPIKE.value: ("#f8d7da88", "#dc3545"),
        AlertType.NONE.value: ("#f0f0f088", "#6c757d"),
    }

    rows = []
    for _, row in df.iterrows():
        raw_timestamp = str(row.get(Fields.ALERT_TIME, ""))
        try:
            timestamp = datetime.strptime(raw_timestamp, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            timestamp = pd.to_datetime(raw_timestamp, errors="coerce")
        cleaned_timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(timestamp) else raw_timestamp

        alert_type = str(row.get(Fields.ALERT_TYPE, AlertType.NONE.value)).lower()
        bg_color, badge_color = alert_colors.get(alert_type, alert_colors[AlertType.NONE.value])
        badge = f"<span style='display:inline-block;padding:2px 10px;border-radius:999px;font-family:monospace;font-size:0.8rem;background-color:{badge_color};color:white;'>{alert_type.title()}</span>"

        rows.append(
            f"<tr style=\"background-color: {bg_color};\">"
            f"<td>{cleaned_timestamp}</td>"
            f"<td>{row.get(Fields.MESSAGE, '-')}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )

    table_html = (
        "<table class='alert-table' style='width:100%;border-collapse:collapse;margin-top:0.5rem;'>"
        "<thead><tr><th style='text-align:left;padding:8px;border-bottom:1px solid #ddd;'>Alert Time</th>"
        "<th style='text-align:left;padding:8px;border-bottom:1px solid #ddd;'>Message</th>"
        "<th style='text-align:left;padding:8px;border-bottom:1px solid #ddd;'>Alert Type</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )

    con.markdown(table_html, unsafe_allow_html=True)
