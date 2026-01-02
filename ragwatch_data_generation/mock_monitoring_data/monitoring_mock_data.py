import pandas as pd
import numpy as np
import os
import time
from datetime import datetime

from ragwatch..constants import Paths, Kpis, KpiCategoriesAllignment, Fields

monitor_list_file = Paths.CSV_MONITOR


kpi_names = []
kpi_names += KpiCategoriesAllignment.retriever_confidence[1]
kpi_names += KpiCategoriesAllignment.document_diversity[1]
kpi_names += KpiCategoriesAllignment.context_utilization[1]
kpi_names += KpiCategoriesAllignment.prompt_truncation[1]
kpi_names += KpiCategoriesAllignment.retriever_generator_drift[1]
kpi_names += KpiCategoriesAllignment.latency_timeout[1]

def generate_kpi_row():
    return {
        Fields.TIMESTAMP: datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        Kpis.score_at_rank[0]: np.random.uniform(10, 90),
        Kpis.confidence_gap[0]: np.random.uniform(30, 95),
        Kpis.score_variance[0]: np.random.uniform(10, 100),
        Kpis.avg_similarity[0]: np.random.uniform(10, 100),
        Kpis.max_similarity[0]: np.random.uniform(90, 100),
        Kpis.diversity_score[0]: np.random.uniform(10, 100),
        Kpis.ngram_overlap[0]: np.random.randint(400, 1000),
        Kpis.cross_attention[0]: np.random.uniform(100, 300),
        Kpis.truncation_rate[0]: np.random.uniform(10, 90),
        Kpis.content_attribution_gap[0]: np.random.uniform(10, 90),
        Kpis.attention_based_analysis[0]: np.random.uniform(10, 90),
        Kpis.retrieval_time[0]: np.random.uniform(10, 90),
        Kpis.generation_time[0]: np.random.uniform(10, 90),
        Kpis.total_latency[0]: np.random.uniform(10, 90),
        Kpis.timeout_events[0]: np.random.uniform(10, 90),
        Kpis.slow_request_ratio[0]: np.random.uniform(10, 90)
    }

def write_kpi_for_monitor(monitor_id):
    row = generate_kpi_row()
    df_row = pd.DataFrame([row])
    file_path = Paths.CSV_KPIS + f"monitor_{monitor_id}.csv"

    file_exists = os.path.isfile(file_path)
    df_row.to_csv(file_path, mode='a', header=not file_exists, index=False)
    print(f"[{row['timestamp']}] Wrote row for {monitor_id}")

# Main loop
if __name__ == "__main__":
    if not os.path.exists(monitor_list_file):
        print("Monitor list file not found.")
        exit(1)

    monitor_df = pd.read_csv(monitor_list_file)

    while True:
        print(f"\nCollecting data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        for monitor_id in monitor_df[Fields.MONITOR_ID]:
            write_kpi_for_monitor(monitor_id)
        time.sleep(60)  # Wait 1 minute
