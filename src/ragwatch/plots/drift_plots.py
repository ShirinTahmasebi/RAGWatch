"""Paper-quality static plots over RAGWatch drift experiment outputs.

These functions read the ``drift_summary.csv`` produced by the drift experiment
example and save static PNG figures with matplotlib. They are separate from the
interactive Streamlit drift view and are intended for papers, reports, slides,
and debugging drift behavior.

Column names mirror the drift summary columns (see the
run_squad_drift_experiment example), which already derive their averaged KPI
names from the central KPI catalog.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so figures are never shown, only saved.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection
import pandas as pd  # noqa: E402

from ragwatch.dashboard.data_loader import (  # noqa: E402
    load_drift_summary as _load_drift_summary,
)

# Column used to label each scenario along the x-axis.
SCENARIO_NAME_COLUMN = "scenario_name"

# Base drift summary metric columns (always attempted).
COL_AVG_RETRIEVAL_SCORE_MEAN = "avg_retrieval_score_mean"
COL_AVG_RETRIEVAL_REDUNDANCY = "avg_retrieval_redundancy"
COL_AVG_ANSWER_LENGTH_WORDS = "avg_answer_length_words"
COL_AVG_TOTAL_LATENCY_MS = "avg_total_latency_ms"

# Optional semantic drift summary columns (only present when semantic KPIs were
# enabled during the drift experiment).
COL_AVG_QUERY_CONTEXT_SIMILARITY_MEAN = "avg_query_context_similarity_mean"
COL_AVG_ANSWER_CONTEXT_SIMILARITY_MEAN = "avg_answer_context_similarity_mean"
COL_AVG_ANSWER_QUERY_SIMILARITY = "avg_answer_query_similarity"


def load_drift_summary(drift_dir: str | Path) -> pd.DataFrame:
    """Load ``drift_summary.csv`` from a drift experiment output directory.

    Raises ``FileNotFoundError`` if the file is missing. This reuses the
    Streamlit-free loader from ``ragwatch.dashboard.data_loader``.
    """
    return _load_drift_summary(drift_dir)


def _has_plottable_column(df: pd.DataFrame, column: str) -> bool:
    """Return True if ``column`` exists and has at least one non-null value."""
    return column in df.columns and df[column].notna().any()


def _save_figure(fig: "plt.Figure", output_path: str | Path) -> None:
    """Write a figure to ``output_path`` (creating parent dirs) and close it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_by_drift_scenario(
    summary_df: pd.DataFrame,
    metric_column: str,
    output_path: str | Path,
    title: str,
    ylabel: str,
) -> None:
    """Draw a bar chart of one summary metric per drift scenario.

    Raises ``KeyError`` if the scenario or metric column is missing.
    """
    missing = [
        c
        for c in (SCENARIO_NAME_COLUMN, metric_column)
        if c not in summary_df.columns
    ]
    if missing:
        raise KeyError(
            f"Missing required column(s) {missing}; available columns: "
            f"{list(summary_df.columns)}"
        )

    data = summary_df.dropna(subset=[metric_column])
    labels = data[SCENARIO_NAME_COLUMN].astype(str).tolist()
    values = data[metric_column].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Drift scenario")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    _save_figure(fig, output_path)


def plot_retrieval_score_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average mean retrieval score per drift scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_RETRIEVAL_SCORE_MEAN,
        output_path=output_path,
        title="Average retrieval score by drift scenario",
        ylabel="Avg retrieval score (mean)",
    )


def plot_retrieval_redundancy_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average retrieval redundancy per drift scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_RETRIEVAL_REDUNDANCY,
        output_path=output_path,
        title="Average retrieval redundancy by drift scenario",
        ylabel="Avg retrieval redundancy",
    )


def plot_answer_length_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average answer length (words) per drift scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_ANSWER_LENGTH_WORDS,
        output_path=output_path,
        title="Average answer length by drift scenario",
        ylabel="Avg answer length (words)",
    )


def plot_latency_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average total latency (ms) per drift scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_TOTAL_LATENCY_MS,
        output_path=output_path,
        title="Average total latency by drift scenario",
        ylabel="Avg total latency (ms)",
    )


def plot_query_context_similarity_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average query-context semantic similarity per scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_QUERY_CONTEXT_SIMILARITY_MEAN,
        output_path=output_path,
        title="Average query-context similarity by drift scenario",
        ylabel="Avg query-context similarity (mean)",
    )


def plot_answer_context_similarity_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average answer-context semantic similarity per scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_ANSWER_CONTEXT_SIMILARITY_MEAN,
        output_path=output_path,
        title="Average answer-context similarity by drift scenario",
        ylabel="Avg answer-context similarity (mean)",
    )


def plot_answer_query_similarity_by_drift_scenario(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average answer-query semantic similarity per scenario."""
    plot_metric_by_drift_scenario(
        summary_df,
        metric_column=COL_AVG_ANSWER_QUERY_SIMILARITY,
        output_path=output_path,
        title="Average answer-query similarity by drift scenario",
        ylabel="Avg answer-query similarity",
    )


# Default subdirectory for figures when no explicit output dir is given.
FIGURES_DIRNAME = "figures"


def generate_all_drift_plots(
    drift_dir: str | Path,
    output_dir: str | Path | None = None,
) -> list[Path]:
    """Generate all standard drift figures and return their paths.

    Reads ``drift_summary.csv`` from ``drift_dir``. When ``output_dir`` is
    ``None``, figures are written to ``drift_dir/figures/``. The base figures
    are always attempted; semantic figures are generated only when their
    columns exist and contain numeric data.
    """
    summary_df = load_drift_summary(drift_dir)

    if output_dir is None:
        output_dir = Path(drift_dir) / FIGURES_DIRNAME
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    base_plots = [
        (
            "retrieval_score_by_drift_scenario.png",
            plot_retrieval_score_by_drift_scenario,
        ),
        (
            "retrieval_redundancy_by_drift_scenario.png",
            plot_retrieval_redundancy_by_drift_scenario,
        ),
        (
            "answer_length_by_drift_scenario.png",
            plot_answer_length_by_drift_scenario,
        ),
        (
            "latency_by_drift_scenario.png",
            plot_latency_by_drift_scenario,
        ),
    ]
    for filename, plot_func in base_plots:
        path = output_dir / filename
        plot_func(summary_df, path)
        created.append(path)

    # --- Optional semantic plots (only when semantic columns have data) ---
    semantic_plots = [
        (
            COL_AVG_QUERY_CONTEXT_SIMILARITY_MEAN,
            "query_context_similarity_by_drift_scenario.png",
            plot_query_context_similarity_by_drift_scenario,
        ),
        (
            COL_AVG_ANSWER_CONTEXT_SIMILARITY_MEAN,
            "answer_context_similarity_by_drift_scenario.png",
            plot_answer_context_similarity_by_drift_scenario,
        ),
        (
            COL_AVG_ANSWER_QUERY_SIMILARITY,
            "answer_query_similarity_by_drift_scenario.png",
            plot_answer_query_similarity_by_drift_scenario,
        ),
    ]
    for column, filename, plot_func in semantic_plots:
        if _has_plottable_column(summary_df, column):
            path = output_dir / filename
            plot_func(summary_df, path)
            created.append(path)

    return created
