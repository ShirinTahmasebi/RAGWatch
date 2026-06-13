"""RAGWatch research dashboard (Streamlit).

A simple, local research dashboard that reads comparison output files produced by
``ragwatch.experiments.comparison.export_comparison`` and visualizes them.

Run with:

    streamlit run src/ragwatch/dashboard/app.py

This is a local research dashboard, not Grafana. It only reads local files and
requires no database, API keys, or running services.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ragwatch.dashboard.data_loader import (
    list_experiment_dirs,
    load_combined_kpis,
    load_comparison_summary,
    load_comparison_summary_json,
    load_runs_jsonl,
)

try:
    import plotly.express as px

    _HAS_PLOTLY = True
except Exception:  # noqa: BLE001
    _HAS_PLOTLY = False

DEFAULT_COMPARISON_DIR = "outputs/comparisons/squad_retrievers_base"
COMPARISONS_ROOT = "outputs/comparisons"

# Non-KPI columns in combined_kpis.csv (everything else is a KPI column).
_BASE_COLUMNS = {
    "comparison_name",
    "experiment_key",
    "dataset_name",
    "retriever_name",
    "generator_name",
    "top_k",
    "example_id",
    "query",
    "run_id",
}

# Summary chart specs: (column, friendly title).
_SUMMARY_CHARTS = [
    ("avg_total_latency_ms", "Average total latency (ms)"),
    ("avg_retrieval_latency_ms", "Average retrieval latency (ms)"),
    ("avg_retrieval_score_mean", "Average retrieval score mean"),
    ("avg_retrieval_redundancy", "Average retrieval redundancy"),
    ("avg_answer_length_words", "Average answer length (words)"),
]


def _numeric_kpi_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric KPI columns (excludes base/identifier columns)."""
    numeric = df.select_dtypes(include="number").columns
    return [c for c in numeric if c not in _BASE_COLUMNS]


def _list_comparison_dirs(root: str) -> list[str]:
    """Return comparison subdirectories under ``root`` that have a summary CSV.

    Returns an empty list if ``root`` does not exist, so the dashboard falls
    back to a plain text input.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        str(p)
        for p in root_path.iterdir()
        if p.is_dir() and (p / "comparison_summary.csv").exists()
    )


def _bar_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> None:
    """Render a grouped bar chart with Plotly if available, else Streamlit."""
    if df.empty or y not in df.columns:
        st.info(f"No data for {title}.")
        return
    if _HAS_PLOTLY:
        fig = px.bar(df, x=x, y=y, color=color, barmode="group", title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(title)
        st.bar_chart(df.pivot_table(index=x, columns=color, values=y))


def _render_overview(
    summary_json: dict, summary_df: pd.DataFrame, combined_df: pd.DataFrame
) -> None:
    st.header("Overview")

    retrievers: list[str] = []
    top_ks: list = []
    if not summary_df.empty:
        retrievers = sorted(summary_df["retriever_name"].unique().tolist())
        top_ks = sorted(summary_df["top_k"].unique().tolist())

    total_success = (
        int(summary_df["num_successful"].sum()) if not summary_df.empty else 0
    )
    total_failed = int(summary_df["num_failed"].sum()) if not summary_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Comparison", summary_json.get("comparison_name", "—"))
    col2.metric("Dataset", summary_json.get("dataset_name", "—"))
    col3.metric("Experiment settings", summary_json.get("num_experiments", len(summary_df)))

    col4, col5, col6 = st.columns(3)
    col4.metric("Successful runs", total_success)
    col5.metric("Failed runs", total_failed)
    col6.metric("Retrievers", len(retrievers))

    st.write("**Available retrievers:**", ", ".join(retrievers) if retrievers else "—")
    st.write(
        "**Available top_k values:**",
        ", ".join(str(t) for t in top_ks) if top_ks else "—",
    )


def _render_retriever_comparison(summary_df: pd.DataFrame) -> None:
    st.header("Retriever comparison")
    if summary_df.empty:
        st.info("No comparison summary available.")
        return

    st.dataframe(summary_df, use_container_width=True)

    for column, title in _SUMMARY_CHARTS:
        if column in summary_df.columns:
            _bar_chart(
                summary_df,
                x="top_k",
                y=column,
                color="retriever_name",
                title=title,
            )


def _render_kpi_explorer(
    combined_df: pd.DataFrame,
    retriever_filter: list[str],
    top_k_filter: list,
) -> None:
    st.header("KPI explorer")
    if combined_df.empty:
        st.info("No combined KPI data available.")
        return

    filtered = _apply_filters(combined_df, retriever_filter, top_k_filter)
    if filtered.empty:
        st.info("No rows match the current filters.")
        return

    kpi_columns = _numeric_kpi_columns(filtered)
    if not kpi_columns:
        st.info("No numeric KPI columns found.")
        return

    selected_kpi = st.selectbox("Select a KPI", kpi_columns)

    st.subheader("Filtered runs")
    display_cols = [
        c
        for c in ("experiment_key", "retriever_name", "top_k", "example_id", selected_kpi)
        if c in filtered.columns
    ]
    st.dataframe(filtered[display_cols], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"Distribution of {selected_kpi}")
        if _HAS_PLOTLY:
            fig = px.histogram(filtered, x=selected_kpi, nbins=20)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(filtered[selected_kpi].value_counts().sort_index())
    with col2:
        st.subheader(f"{selected_kpi} by retriever")
        if _HAS_PLOTLY:
            fig = px.box(filtered, x="retriever_name", y=selected_kpi, color="retriever_name")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(
                filtered.groupby("retriever_name")[selected_kpi].mean()
            )


def _render_run_inspector(
    combined_df: pd.DataFrame, comparison_dir: Path
) -> None:
    st.header("Query / run inspector")
    if combined_df.empty:
        st.info("No combined KPI data available.")
        return

    experiment_keys = sorted(combined_df["experiment_key"].unique().tolist())
    selected_key = st.selectbox("Experiment", experiment_keys)

    key_rows = combined_df[combined_df["experiment_key"] == selected_key]
    example_ids = sorted(key_rows["example_id"].astype(str).unique().tolist())
    if not example_ids:
        st.info("No examples for this experiment.")
        return
    selected_example = st.selectbox("Example ID", example_ids)

    row = key_rows[key_rows["example_id"].astype(str) == selected_example]
    if row.empty:
        st.info("No matching run.")
        return
    record = row.iloc[0].to_dict()

    col1, col2, col3 = st.columns(3)
    col1.metric("Retriever", record.get("retriever_name", "—"))
    col2.metric("Generator", record.get("generator_name", "—"))
    col3.metric("top_k", record.get("top_k", "—"))

    st.write("**Query:**", record.get("query", "—"))
    st.write("**Run ID:**", record.get("run_id", "—"))

    # Answer / retrieved docs live in the per-experiment runs.jsonl.
    runs_record = _lookup_run_record(comparison_dir, selected_key, selected_example)
    if runs_record is not None:
        if runs_record.get("answer") is not None:
            st.subheader("Answer")
            st.write(runs_record["answer"])
        retrieved = {
            "doc_ids": runs_record.get("retrieved_doc_ids"),
            "scores": runs_record.get("retrieved_scores"),
            "ranks": runs_record.get("retrieved_ranks"),
        }
        if any(v is not None for v in retrieved.values()):
            st.subheader("Retrieved documents")
            st.dataframe(pd.DataFrame(retrieved), use_container_width=True)

    st.subheader("KPI values")
    kpi_cols = _numeric_kpi_columns(combined_df)
    kpi_values = {c: record.get(c) for c in kpi_cols if c in record}
    st.dataframe(
        pd.DataFrame({"kpi": list(kpi_values.keys()), "value": list(kpi_values.values())}),
        use_container_width=True,
    )


def _render_failures(comparison_dir: Path) -> None:
    st.header("Failure inspection")

    failures: list[dict] = []
    for exp_dir in list_experiment_dirs(comparison_dir):
        try:
            runs_df = load_runs_jsonl(exp_dir)
        except FileNotFoundError:
            continue
        if runs_df.empty or "error" not in runs_df.columns:
            continue
        failed = runs_df[runs_df["error"].notna()]
        for _, r in failed.iterrows():
            failures.append(
                {
                    "experiment_key": exp_dir.name,
                    "example_id": r.get("example_id"),
                    "question": r.get("question"),
                    "error": r.get("error"),
                }
            )

    if failures:
        st.dataframe(pd.DataFrame(failures), use_container_width=True)
    else:
        st.success("No failed examples found.")


def _lookup_run_record(
    comparison_dir: Path, experiment_key: str, example_id: str
) -> dict | None:
    """Find a single run record in a per-experiment runs.jsonl."""
    exp_dir = comparison_dir / "experiments" / experiment_key
    try:
        runs_df = load_runs_jsonl(exp_dir)
    except FileNotFoundError:
        return None
    if runs_df.empty or "example_id" not in runs_df.columns:
        return None
    match = runs_df[runs_df["example_id"].astype(str) == str(example_id)]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def _apply_filters(
    df: pd.DataFrame, retriever_filter: list[str], top_k_filter: list
) -> pd.DataFrame:
    filtered = df
    if retriever_filter and "retriever_name" in filtered.columns:
        filtered = filtered[filtered["retriever_name"].isin(retriever_filter)]
    if top_k_filter and "top_k" in filtered.columns:
        filtered = filtered[filtered["top_k"].isin(top_k_filter)]
    return filtered


def main() -> None:
    st.set_page_config(page_title="RAGWatch Research Dashboard", layout="wide")
    st.title("RAGWatch Research Dashboard")

    st.sidebar.header("Settings")

    # Offer available comparison directories (if any) plus a custom path field.
    available_dirs = _list_comparison_dirs(COMPARISONS_ROOT)
    if available_dirs:
        choices = available_dirs + ["Custom path..."]
        default_index = (
            choices.index(DEFAULT_COMPARISON_DIR)
            if DEFAULT_COMPARISON_DIR in choices
            else 0
        )
        selected = st.sidebar.selectbox(
            "Comparison output directory", choices, index=default_index
        )
        if selected == "Custom path...":
            comparison_dir_str = st.sidebar.text_input(
                "Custom comparison output directory", value=DEFAULT_COMPARISON_DIR
            )
        else:
            comparison_dir_str = selected
    else:
        comparison_dir_str = st.sidebar.text_input(
            "Comparison output directory", value=DEFAULT_COMPARISON_DIR
        )
    comparison_dir = Path(comparison_dir_str)

    try:
        summary_json = load_comparison_summary_json(comparison_dir)
        summary_df = load_comparison_summary(comparison_dir)
        combined_df = load_combined_kpis(comparison_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Generate comparison outputs first:\n\n"
            "`python examples/run_squad_retriever_comparison.py`"
        )
        return

    # Sidebar filters.
    st.sidebar.subheader("Filters")
    retrievers = (
        sorted(combined_df["retriever_name"].unique().tolist())
        if not combined_df.empty and "retriever_name" in combined_df.columns
        else []
    )
    top_ks = (
        sorted(combined_df["top_k"].unique().tolist())
        if not combined_df.empty and "top_k" in combined_df.columns
        else []
    )
    retriever_filter = st.sidebar.multiselect(
        "retriever_name", retrievers, default=retrievers
    )
    top_k_filter = st.sidebar.multiselect("top_k", top_ks, default=top_ks)

    _render_overview(summary_json, summary_df, combined_df)
    st.divider()
    _render_retriever_comparison(summary_df)
    st.divider()
    _render_kpi_explorer(combined_df, retriever_filter, top_k_filter)
    st.divider()
    _render_run_inspector(combined_df, comparison_dir)
    st.divider()
    _render_failures(comparison_dir)


if __name__ == "__main__":
    main()
