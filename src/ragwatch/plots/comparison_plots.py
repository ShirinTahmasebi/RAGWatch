"""Paper-quality static plots over RAGWatch comparison outputs.

These functions read the comparison files produced by
``ragwatch.experiments.comparison.export_comparison`` and save static PNG
figures with matplotlib. They are separate from the interactive Streamlit
dashboard and are intended for papers, reports, slides, and debugging.

Column names are derived from the central KPI catalog so KPI names are not
hardcoded as raw strings.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so figures are never shown, only saved.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection
import pandas as pd  # noqa: E402

from ragwatch.dashboard.data_loader import (  # noqa: E402
    load_combined_kpis,
    load_comparison_summary,
)
from ragwatch.metrics.catalog import KPIId  # noqa: E402

# Summary CSV averages KPIs with an ``avg_`` prefix (see comparison.py).
AVG_PREFIX = "avg_"

COL_AVG_TOTAL_LATENCY_MS = f"{AVG_PREFIX}{KPIId.TOTAL_LATENCY_MS}"
COL_AVG_RETRIEVAL_SCORE_MEAN = f"{AVG_PREFIX}{KPIId.RETRIEVAL_SCORE_MEAN}"
COL_AVG_RETRIEVAL_REDUNDANCY = f"{AVG_PREFIX}{KPIId.RETRIEVAL_REDUNDANCY}"
COL_AVG_ANSWER_LENGTH_WORDS = f"{AVG_PREFIX}{KPIId.ANSWER_LENGTH_WORDS}"

# Optional semantic KPI columns (only present when semantic KPIs were enabled).
COL_AVG_QUERY_CONTEXT_SIMILARITY_MEAN = (
    f"{AVG_PREFIX}{KPIId.QUERY_CONTEXT_SIMILARITY_MEAN}"
)
COL_AVG_ANSWER_CONTEXT_SIMILARITY_MEAN = (
    f"{AVG_PREFIX}{KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN}"
)
COL_AVG_ANSWER_QUERY_SIMILARITY = f"{AVG_PREFIX}{KPIId.ANSWER_QUERY_SIMILARITY}"

# Columns used to label/group experiment settings in the summary table.
EXPERIMENT_KEY_COLUMN = "experiment_key"
RETRIEVER_NAME_COLUMN = "retriever_name"
TOP_K_COLUMN = "top_k"


def load_comparison_data(
    comparison_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the comparison summary and combined KPI tables.

    Returns ``(summary_df, combined_df)`` loaded from ``comparison_summary.csv``
    and ``combined_kpis.csv``. Raises ``FileNotFoundError`` if either is missing.
    """
    summary_df = load_comparison_summary(comparison_dir)
    combined_df = load_combined_kpis(comparison_dir)
    return summary_df, combined_df


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear KeyError if any required column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required column(s) {missing}; available columns: "
            f"{list(df.columns)}"
        )


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


def _bar_by_experiment(
    summary_df: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> None:
    """Draw a bar chart of one summary metric per experiment setting."""
    _require_columns(summary_df, [EXPERIMENT_KEY_COLUMN, value_column])

    data = summary_df.dropna(subset=[value_column])
    labels = data[EXPERIMENT_KEY_COLUMN].astype(str).tolist()
    values = data[value_column].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_xlabel("Experiment (retriever x top-k)")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")

    _save_figure(fig, output_path)


def plot_latency_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average total latency (ms) per experiment setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_TOTAL_LATENCY_MS,
        title="Average total latency by retriever",
        ylabel="Avg total latency (ms)",
        output_path=output_path,
    )


def plot_retrieval_score_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average mean retrieval score per experiment setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_RETRIEVAL_SCORE_MEAN,
        title="Average retrieval score by retriever",
        ylabel="Avg retrieval score (mean)",
        output_path=output_path,
    )


def plot_redundancy_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average retrieval redundancy per experiment setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_RETRIEVAL_REDUNDANCY,
        title="Average retrieval redundancy by retriever",
        ylabel="Avg retrieval redundancy",
        output_path=output_path,
    )


def plot_answer_length_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average answer length (words) per experiment setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_ANSWER_LENGTH_WORDS,
        title="Average answer length by retriever",
        ylabel="Avg answer length (words)",
        output_path=output_path,
    )


def plot_query_context_similarity_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average query-context semantic similarity per setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_QUERY_CONTEXT_SIMILARITY_MEAN,
        title="Average query-context similarity by retriever",
        ylabel="Avg query-context similarity (mean)",
        output_path=output_path,
    )


def plot_answer_context_similarity_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average answer-context semantic similarity per setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_ANSWER_CONTEXT_SIMILARITY_MEAN,
        title="Average answer-context similarity by retriever",
        ylabel="Avg answer-context similarity (mean)",
        output_path=output_path,
    )


def plot_answer_query_similarity_by_retriever(
    summary_df: pd.DataFrame, output_path: str | Path
) -> None:
    """Bar chart of average answer-query semantic similarity per setting."""
    _bar_by_experiment(
        summary_df,
        value_column=COL_AVG_ANSWER_QUERY_SIMILARITY,
        title="Average answer-query similarity by retriever",
        ylabel="Avg answer-query similarity",
        output_path=output_path,
    )


def plot_top_k_sensitivity(
    summary_df: pd.DataFrame, metric: str, output_path: str | Path
) -> None:
    """Plot ``metric`` across top-k values, one line per retriever."""
    _require_columns(summary_df, [RETRIEVER_NAME_COLUMN, TOP_K_COLUMN, metric])

    data = summary_df.dropna(subset=[metric])

    fig, ax = plt.subplots(figsize=(8, 5))
    for retriever_name, group in data.groupby(RETRIEVER_NAME_COLUMN):
        ordered = group.sort_values(TOP_K_COLUMN)
        ax.plot(
            ordered[TOP_K_COLUMN].astype(int).tolist(),
            ordered[metric].astype(float).tolist(),
            marker="o",
            label=str(retriever_name),
        )

    ax.set_title(f"Top-k sensitivity: {metric}")
    ax.set_xlabel("top_k")
    ax.set_ylabel(metric)
    if data[TOP_K_COLUMN].nunique() > 0:
        ax.set_xticks(sorted(data[TOP_K_COLUMN].astype(int).unique().tolist()))
    ax.legend(title="retriever")

    _save_figure(fig, output_path)


def generate_all_comparison_plots(
    comparison_dir: str | Path,
    output_dir: str | Path,
) -> list[Path]:
    """Generate all standard comparison figures and return their paths."""
    summary_df, _ = load_comparison_data(comparison_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    latency_path = output_dir / "latency_by_retriever.png"
    plot_latency_by_retriever(summary_df, latency_path)
    created.append(latency_path)

    score_path = output_dir / "retrieval_score_by_retriever.png"
    plot_retrieval_score_by_retriever(summary_df, score_path)
    created.append(score_path)

    redundancy_path = output_dir / "retrieval_redundancy_by_retriever.png"
    plot_redundancy_by_retriever(summary_df, redundancy_path)
    created.append(redundancy_path)

    answer_length_path = output_dir / "answer_length_by_retriever.png"
    plot_answer_length_by_retriever(summary_df, answer_length_path)
    created.append(answer_length_path)

    top_k_latency_path = output_dir / "top_k_latency_sensitivity.png"
    plot_top_k_sensitivity(summary_df, COL_AVG_TOTAL_LATENCY_MS, top_k_latency_path)
    created.append(top_k_latency_path)

    top_k_score_path = output_dir / "top_k_retrieval_score_sensitivity.png"
    plot_top_k_sensitivity(
        summary_df, COL_AVG_RETRIEVAL_SCORE_MEAN, top_k_score_path
    )
    created.append(top_k_score_path)

    # --- Optional semantic plots (only when semantic columns have data) ---
    semantic_plots = [
        (
            COL_AVG_QUERY_CONTEXT_SIMILARITY_MEAN,
            "query_context_similarity_by_retriever.png",
            plot_query_context_similarity_by_retriever,
        ),
        (
            COL_AVG_ANSWER_CONTEXT_SIMILARITY_MEAN,
            "answer_context_similarity_by_retriever.png",
            plot_answer_context_similarity_by_retriever,
        ),
        (
            COL_AVG_ANSWER_QUERY_SIMILARITY,
            "answer_query_similarity_by_retriever.png",
            plot_answer_query_similarity_by_retriever,
        ),
    ]
    for column, filename, plot_func in semantic_plots:
        if _has_plottable_column(summary_df, column):
            path = output_dir / filename
            plot_func(summary_df, path)
            created.append(path)

    return created
