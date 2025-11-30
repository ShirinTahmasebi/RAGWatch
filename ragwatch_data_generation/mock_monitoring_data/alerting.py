import pandas as pd
import os
import time
from enum import Enum
from datetime import datetime

from utils.constants import AlertType, Fields, Paths, Kpis, KpiCategoriesAllignment
from utils.helper import read_csv_if_exists, is_non_empty_df

monitor_list_file = Paths.CSV_MONITOR

class AlertItem:
    def __init__(
        self,
        monitor_id: str,
        kpi_name: str,
        alert_type: AlertType,
        alert_message: str,
        window_start_time: str,
        window_end_time: str,
        alert_capturing_time: str
    ):
        self.alert_id = f"{monitor_id}_{kpi_name}_{alert_capturing_time}"
        self.monitor_id = monitor_id
        self.kpi_name = kpi_name
        self.alert_type = alert_type
        self.alert_message = alert_message
        self.window_start_time = window_start_time
        self.window_end_time = window_end_time
        self.alert_capturing_time = alert_capturing_time
        

def check_kpi_alert(
    df,
    monitor_id,
    kpi_item,
    increase_threshold=0.2,
    decrease_threshold=0.2,
    spike_threshold=0.5
):
    start_time = df["timestamp"].iloc[0]
    end_time = df["timestamp"].iloc[-1]
    alert_time = datetime.now()
    kpi_name = kpi_item[0]
    values = df[kpi_name].reset_index(drop=True)

    # Detect spike
    for i in range(1, len(values)):
        prev, curr = values[i - 1], values[i]
        if prev == 0:
            continue
        spike = abs(curr - prev) / prev
        if spike > spike_threshold:
            return AlertItem(
                monitor_id,
                kpi_name, 
                AlertType.SPIKE, 
                f"Detected {spike*100:.1f}% change",
                start_time,
                end_time,
                alert_time
            )

    # Detect trend
    start, end = values.iloc[0], values.iloc[-1]
    if start == 0:
        return None

    pct_change = (end - start) / start

    if pct_change > increase_threshold:
        return AlertItem(
            monitor_id,
            kpi_name, 
            AlertType.INCREASE, 
            f"Detected an increase of {pct_change*100:.1f}%",
            start_time,
            end_time,
            alert_time
        )
    elif pct_change < -decrease_threshold:
        return AlertItem(
            monitor_id,
            kpi_name, 
            AlertType.DROP, 
            f"Detected a drop of {pct_change*100:.1f}%",
            start_time,
            end_time,
            alert_time
        )

    return None


def generate_alert_rows(alert_items):
    alert_dict_lsit = []
    for alert_item in alert_items:
        alert_dict_lsit.append({
            Fields.ALERT_ID: alert_item.alert_id,
            Fields.MONITOR_ID: alert_item.monitor_id,
            Fields.KPI: alert_item.kpi_name,
            Fields.ALERT_TYPE: alert_item.alert_type.value,
            Fields.MESSAGE: alert_item.alert_message,
            Fields.START_TIME: alert_item.window_start_time,
            Fields.END_TIME: alert_item.window_end_time,
            Fields.ALERT_TIME: alert_item.alert_capturing_time,
        })
    return alert_dict_lsit
    
def log_alerts(alert_file_name, alert_items):
    if alert_items is None:
        return
        
    df_row = pd.DataFrame(generate_alert_rows(alert_items))

    df_row.to_csv(alert_file_name, mode="a", header=not os.path.exists(alert_file_name), index=False)


def get_last_alert_time(alert_file_name):
    if not os.path.exists(alert_file_name):
        return None
    df = read_csv_if_exists(alert_file_name)
    if df.empty:
        return None
    return df["end_time"].max()


def process_monitor_kpi(monitor_id, window_size=5):
    alert_file_name = Paths.CSV_ALERTS + f"alert_{monitor_id}.csv"
    monitor_file_name = Paths.CSV_KPIS + f"monitor_{monitor_id}.csv"
    
    df = read_csv_if_exists(monitor_file_name)
    if not is_non_empty_df(df):
        return
        
    df = df.sort_values(Fields.TIMESTAMP)

    last_alert_time = get_last_alert_time(alert_file_name)

    if last_alert_time:
        df = df[df[Fields.TIMESTAMP] > last_alert_time]

    if len(df) < window_size:
        print(f"Not enough new data for {monitor_id}")
        return


    # Sliding window
    kpi_list = []
    kpi_list += KpiCategoriesAllignment.retriever_confidence[1]
    kpi_list += KpiCategoriesAllignment.document_diversity[1]
    kpi_list += KpiCategoriesAllignment.context_utilization[1]
    kpi_list += KpiCategoriesAllignment.prompt_truncation[1]
    kpi_list += KpiCategoriesAllignment.retriever_generator_drift[1]
    kpi_list += KpiCategoriesAllignment.latency_timeout[1]

    alert_items = []
    for i in range(0, len(df) - window_size + 1):
        window = df.iloc[i : i + window_size]
        for kpi_name in kpi_list:
            alert_item = check_kpi_alert(window, monitor_id, kpi_name)
            if alert_item:
                alert_items.append(alert_item)
    alert_items.append(
        AlertItem(monitor_id, None, AlertType.DUMMY, None, None, df[Fields.TIMESTAMP].max(), None)
    )
    
    log_alerts(alert_file_name, alert_items)
                
    print(f"Finished evaluating {monitor_id}, new alerts logged in {alert_file_name}")


# Main loop
if __name__ == "__main__":
    if not os.path.exists(monitor_list_file):
        print("Monitor list file not found.")
        exit(1)

    monitor_df = pd.read_csv(monitor_list_file)

    while True:
        print(f"\nDetecting alerts at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for monitor_id in monitor_df[Fields.MONITOR_ID]:
            process_monitor_kpi(monitor_id)
        time.sleep(60)  # Wait 1 minute