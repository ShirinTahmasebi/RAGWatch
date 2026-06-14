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
    list_drift_scenario_dirs,
    list_experiment_dirs,
    load_combined_kpis,
    load_comparison_summary,
    load_comparison_summary_json,
    load_drift_manifest,
    load_drift_qa_examples,
    load_drift_summary,
    load_runs_jsonl,
    load_scenario_runs,
)
from ragwatch.datasets.registry import dataset_name_from_env
from ragwatch.experiments.output_naming import (
    comparison_output_dir,
    drift_output_dir,
)
from ragwatch.semantic.providers import (
    resolve_semantic_provider_name,
    semantic_kpis_enabled,
)

try:
    import plotly.express as px

    _HAS_PLOTLY = True
except Exception:  # noqa: BLE001
    _HAS_PLOTLY = False


def _default_dataset_mode() -> tuple[str, bool, str | None]:
    """Return the (dataset_name, semantic_enabled, provider_name) from env.

    Used to derive dataset-/mode-aware default directories that mirror the
    generic experiment scripts (default env -> squad, semantic_local).
    """
    dataset_name = dataset_name_from_env()
    semantic_enabled = semantic_kpis_enabled()
    provider_name = (
        resolve_semantic_provider_name() if semantic_enabled else None
    )
    return dataset_name, semantic_enabled, provider_name


_DATASET_NAME, _SEMANTIC_ENABLED, _PROVIDER_NAME = _default_dataset_mode()

DEFAULT_COMPARISON_DIR = str(
    comparison_output_dir(_DATASET_NAME, _SEMANTIC_ENABLED, _PROVIDER_NAME)
)
COMPARISONS_ROOT = "outputs/comparisons"

DEFAULT_DRIFT_DIR = str(
    drift_output_dir(_DATASET_NAME, _SEMANTIC_ENABLED, _PROVIDER_NAME)
)
DRIFT_ROOT = "outputs/drift"
CLEAN_SCENARIO_NAME = "clean"

# Drift scenario-level summary chart specs: (column, friendly title).
_DRIFT_SUMMARY_CHARTS = [
    ("avg_retrieval_score_mean", "Average retrieval score mean"),
    ("avg_retrieval_redundancy", "Average retrieval redundancy"),
    ("avg_answer_length_words", "Average answer length (words)"),
    ("avg_total_latency_ms", "Average total latency (ms)"),
]

# Scenario fields shown side-by-side in the clean vs drifted comparison.
_DRIFT_COMPARE_FIELDS = [
    ("scenario_name", "Scenario"),
    ("perturbation_type", "Perturbation type"),
    ("severity", "Severity"),
    ("production_interpretation", "Production interpretation"),
    ("num_documents", "Num documents"),
    ("num_examples", "Num examples"),
    ("avg_retrieval_score_mean", "Avg retrieval score"),
    ("avg_retrieval_redundancy", "Avg redundancy"),
    ("avg_answer_length_words", "Avg answer length (words)"),
    ("avg_total_latency_ms", "Avg total latency (ms)"),
]

# Manifest fields highlighted above the raw JSON.
_MANIFEST_HIGHLIGHT_FIELDS = [
    "perturbation_type",
    "severity",
    "random_seed",
    "production_interpretation",
    "num_original_documents",
    "num_final_documents",
    "num_added_documents",
    "num_removed_documents",
    "num_modified_documents",
    "num_modified_queries",
    "added_doc_ids",
    "removed_doc_ids",
    "modified_doc_ids",
    "modified_example_ids",
]


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


# ---------------------------------------------------------------------------
# Drift experiment view
# ---------------------------------------------------------------------------


def _list_drift_dirs(root: str) -> list[str]:
    """Return drift directories under ``root`` that have a drift_summary.csv.

    Returns an empty list if ``root`` does not exist, so the dashboard falls
    back to a plain text input.
    """
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(
        str(p)
        for p in root_path.iterdir()
        if p.is_dir() and (p / "drift_summary.csv").exists()
    )


def _scenario_name_to_dir(drift_dir: Path) -> dict[str, Path]:
    """Map each scenario name to its directory (by folder name)."""
    return {p.name: p for p in list_drift_scenario_dirs(drift_dir)}


def _scenario_row(summary_df: pd.DataFrame, scenario_name: str) -> dict:
    """Return the summary row for a scenario as a dict (empty if missing)."""
    if summary_df.empty or "scenario_name" not in summary_df.columns:
        return {}
    match = summary_df[summary_df["scenario_name"].astype(str) == scenario_name]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _render_drift_overview(summary_df: pd.DataFrame) -> None:
    st.subheader("Scenario overview")
    if summary_df.empty:
        st.info("No drift summary available.")
        return

    perturbations = (
        sorted(summary_df["perturbation_type"].unique().tolist())
        if "perturbation_type" in summary_df.columns
        else []
    )
    col1, col2 = st.columns(2)
    col1.metric("Scenarios", len(summary_df))
    col2.metric("Perturbation types", len(perturbations))
    st.write(
        "**Perturbation types:**",
        ", ".join(perturbations) if perturbations else "—",
    )

    st.dataframe(summary_df, use_container_width=True)

    for column, title in _DRIFT_SUMMARY_CHARTS:
        if column in summary_df.columns:
            chart_df = summary_df[["scenario_name", column]].dropna()
            if chart_df.empty:
                continue
            st.caption(title)
            st.bar_chart(chart_df.set_index("scenario_name")[column])


def _render_drift_scenario_comparison(
    summary_df: pd.DataFrame, scenario_names: list[str]
) -> tuple[str, str]:
    """Render clean vs drifted side-by-side cards. Returns (baseline, drift)."""
    st.subheader("Clean vs drifted scenario")
    if not scenario_names:
        st.info("No scenarios found in this drift directory.")
        return "", ""

    baseline_default = (
        scenario_names.index(CLEAN_SCENARIO_NAME)
        if CLEAN_SCENARIO_NAME in scenario_names
        else 0
    )
    drift_candidates = [s for s in scenario_names if s != CLEAN_SCENARIO_NAME]
    drift_options = drift_candidates or scenario_names

    col_select1, col_select2 = st.columns(2)
    baseline = col_select1.selectbox(
        "Baseline scenario",
        scenario_names,
        index=baseline_default,
        key="drift_baseline_scenario",
    )
    drift = col_select2.selectbox(
        "Drift scenario", drift_options, index=0, key="drift_drift_scenario"
    )

    baseline_row = _scenario_row(summary_df, baseline)
    drift_row = _scenario_row(summary_df, drift)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Baseline: {baseline}**")
        for field, label in _DRIFT_COMPARE_FIELDS:
            st.write(f"{label}:", baseline_row.get(field, "—"))
    with col2:
        st.markdown(f"**Drift: {drift}**")
        for field, label in _DRIFT_COMPARE_FIELDS:
            st.write(f"{label}:", drift_row.get(field, "—"))

    return baseline, drift


def _render_drift_manifest(scenario_dir: Path, scenario_name: str) -> None:
    st.subheader(f"Manifest — {scenario_name}")
    try:
        manifest = load_drift_manifest(scenario_dir)
    except FileNotFoundError:
        st.info(f"No manifest found for scenario '{scenario_name}'.")
        return

    highlights = {
        field: manifest[field]
        for field in _MANIFEST_HIGHLIGHT_FIELDS
        if field in manifest
    }
    if highlights:
        st.dataframe(
            pd.DataFrame(
                {
                    "field": list(highlights.keys()),
                    "value": [str(v) for v in highlights.values()],
                }
            ),
            use_container_width=True,
        )
    st.json(manifest)


def _retrieved_table(record: dict) -> pd.DataFrame | None:
    """Build a retrieved-documents table (doc_ids/scores/ranks) from a run."""
    retrieved = {
        "doc_ids": record.get("retrieved_doc_ids"),
        "scores": record.get("retrieved_scores"),
        "ranks": record.get("retrieved_ranks"),
    }
    if all(v is None for v in retrieved.values()):
        return None
    return pd.DataFrame(retrieved)


def _render_scenario_example_side(
    scenario_dir: Path, scenario_name: str, example_id: str
) -> None:
    """Render question/answer/retrieved docs for one scenario + example."""
    st.markdown(f"**{scenario_name}**")
    try:
        runs_df = load_scenario_runs(scenario_dir)
    except FileNotFoundError:
        st.info("No runs.jsonl for this scenario.")
        return

    if "example_id" not in runs_df.columns:
        st.info("No example_id column in runs.")
        return
    match = runs_df[runs_df["example_id"].astype(str) == str(example_id)]
    if match.empty:
        st.info(f"Example '{example_id}' not found in this scenario.")
        return
    record = match.iloc[0].to_dict()

    st.write("**Question:**", record.get("question", "—"))

    # Show the original question for shifted queries when available.
    try:
        qa_df = load_drift_qa_examples(scenario_dir)
        if "example_id" in qa_df.columns:
            qa_match = qa_df[qa_df["example_id"].astype(str) == str(example_id)]
            if not qa_match.empty:
                meta = qa_match.iloc[0].get("metadata")
                if isinstance(meta, dict) and meta.get("original_question"):
                    st.write("**Original question:**", meta["original_question"])
    except FileNotFoundError:
        pass

    if record.get("answer") is not None:
        st.write("**Answer:**", record["answer"])

    table = _retrieved_table(record)
    if table is not None:
        st.caption("Retrieved documents")
        st.dataframe(table, use_container_width=True)


def _render_drift_example_inspector(
    scenario_to_dir: dict[str, Path], baseline: str, drift: str
) -> None:
    st.subheader("Clean vs drifted example inspection")
    if not baseline or not drift:
        return
    baseline_dir = scenario_to_dir.get(baseline)
    drift_dir = scenario_to_dir.get(drift)
    if baseline_dir is None or drift_dir is None:
        st.info("Selected scenarios are missing their snapshot directories.")
        return

    try:
        baseline_runs = load_scenario_runs(baseline_dir)
    except FileNotFoundError:
        st.info("No runs.jsonl for the baseline scenario.")
        return

    example_ids = (
        sorted(baseline_runs["example_id"].astype(str).unique().tolist())
        if "example_id" in baseline_runs.columns
        else []
    )
    if not example_ids:
        st.info("No examples available in the baseline scenario.")
        return

    example_id = st.selectbox(
        "Example ID", example_ids, key="drift_example_id"
    )

    col1, col2 = st.columns(2)
    with col1:
        _render_scenario_example_side(baseline_dir, baseline, example_id)
    with col2:
        _render_scenario_example_side(drift_dir, drift, example_id)


def _render_drift_section(drift_dir: Path) -> None:
    try:
        summary_df = load_drift_summary(drift_dir)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info(
            "Generate drift outputs first:\n\n"
            "`python examples/run_squad_drift_experiment.py`"
        )
        return

    scenario_to_dir = _scenario_name_to_dir(drift_dir)
    scenario_names = sorted(scenario_to_dir.keys())

    _render_drift_overview(summary_df)
    st.divider()
    baseline, drift = _render_drift_scenario_comparison(
        summary_df, scenario_names
    )
    st.divider()
    if drift and drift in scenario_to_dir:
        _render_drift_manifest(scenario_to_dir[drift], drift)
        st.divider()
    _render_drift_example_inspector(scenario_to_dir, baseline, drift)


def _render_comparison_section(comparison_dir: Path) -> None:
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
    st.sidebar.subheader("Comparison filters")
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


def _comparison_dir_input() -> Path:
    """Render the comparison directory selector in the sidebar."""
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
    return Path(comparison_dir_str)


def _drift_dir_input() -> Path:
    """Render the drift directory selector in the sidebar."""
    available_dirs = _list_drift_dirs(DRIFT_ROOT)
    if available_dirs:
        choices = available_dirs + ["Custom path..."]
        default_index = (
            choices.index(DEFAULT_DRIFT_DIR)
            if DEFAULT_DRIFT_DIR in choices
            else 0
        )
        selected = st.sidebar.selectbox(
            "Drift output directory", choices, index=default_index
        )
        if selected == "Custom path...":
            drift_dir_str = st.sidebar.text_input(
                "Custom drift output directory", value=DEFAULT_DRIFT_DIR
            )
        else:
            drift_dir_str = selected
    else:
        drift_dir_str = st.sidebar.text_input(
            "Drift output directory", value=DEFAULT_DRIFT_DIR
        )
    return Path(drift_dir_str)


def main() -> None:
    st.set_page_config(page_title="RAGWatch Research Dashboard", layout="wide")
    st.title("RAGWatch Research Dashboard")

    st.sidebar.header("Settings")
    comparison_dir = _comparison_dir_input()
    drift_dir = _drift_dir_input()

    comparison_tab, drift_tab = st.tabs(
        ["Retriever comparison", "Drift experiments"]
    )
    with comparison_tab:
        _render_comparison_section(comparison_dir)
    with drift_tab:
        st.header("Drift experiments")
        _render_drift_section(drift_dir)


if __name__ == "__main__":
    main()

