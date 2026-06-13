"""Tests for the paper-quality comparison plotting utilities.

These tests use small fake CSV files and the non-interactive matplotlib
backend; they require no Streamlit, Plotly, Hugging Face, vector DBs,
Postgres, OpenTelemetry, or API keys.
"""

from pathlib import Path

import pandas as pd
import pytest

from ragwatch.plots.comparison_plots import (
    COL_AVG_RETRIEVAL_SCORE_MEAN,
    COL_AVG_TOTAL_LATENCY_MS,
    generate_all_comparison_plots,
    load_comparison_data,
    plot_answer_context_similarity_by_retriever,
    plot_answer_length_by_retriever,
    plot_answer_query_similarity_by_retriever,
    plot_latency_by_retriever,
    plot_query_context_similarity_by_retriever,
    plot_redundancy_by_retriever,
    plot_retrieval_score_by_retriever,
    plot_top_k_sensitivity,
)

EXPECTED_FIGURES = [
    "latency_by_retriever.png",
    "retrieval_score_by_retriever.png",
    "retrieval_redundancy_by_retriever.png",
    "answer_length_by_retriever.png",
    "top_k_latency_sensitivity.png",
    "top_k_retrieval_score_sensitivity.png",
]


def _make_comparison_dir(tmp_path: Path) -> Path:
    """Create a fake comparison output directory with summary + combined CSVs."""
    comparison_dir = tmp_path / "squad_retrievers"
    comparison_dir.mkdir()

    (comparison_dir / "comparison_summary.csv").write_text(
        "comparison_name,experiment_key,retriever_name,top_k,"
        "num_examples,num_successful,num_failed,"
        "avg_total_latency_ms,avg_retrieval_latency_ms,avg_generation_latency_ms,"
        "avg_retrieval_score_mean,avg_retrieval_redundancy,avg_answer_length_words\n"
        "cmp,tfidf_top3,tfidf,3,2,2,0,10.0,4.0,6.0,0.5,0.2,12.0\n"
        "cmp,tfidf_top5,tfidf,5,2,2,0,14.0,6.0,8.0,0.45,0.3,15.0\n"
        "cmp,chroma_top3,chroma,3,2,2,0,20.0,9.0,11.0,0.7,0.1,11.0\n"
        "cmp,chroma_top5,chroma,5,2,2,0,24.0,11.0,13.0,0.65,0.15,13.0\n",
        encoding="utf-8",
    )

    (comparison_dir / "combined_kpis.csv").write_text(
        "comparison_name,experiment_key,retriever_name,top_k,example_id,"
        "total_latency_ms,retrieval_score_mean\n"
        "cmp,tfidf_top3,tfidf,3,ex0,10.0,0.5\n"
        "cmp,tfidf_top5,tfidf,5,ex0,14.0,0.45\n",
        encoding="utf-8",
    )

    return comparison_dir


def _summary_df(tmp_path: Path) -> pd.DataFrame:
    summary_df, _ = load_comparison_data(_make_comparison_dir(tmp_path))
    return summary_df


class TestLoadComparisonData:
    def test_loads_both_csvs(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        summary_df, combined_df = load_comparison_data(comparison_dir)
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(combined_df, pd.DataFrame)
        assert len(summary_df) == 4
        assert len(combined_df) == 2
        assert COL_AVG_TOTAL_LATENCY_MS in summary_df.columns

    def test_missing_summary_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_comparison_data(tmp_path)


class TestPerMetricPlots:
    def test_plot_latency_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "latency.png"
        plot_latency_by_retriever(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_retrieval_score_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "score.png"
        plot_retrieval_score_by_retriever(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_redundancy_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "redundancy.png"
        plot_redundancy_by_retriever(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_answer_length_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "answer_length.png"
        plot_answer_length_by_retriever(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_plot_top_k_sensitivity_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "sensitivity.png"
        plot_top_k_sensitivity(
            _summary_df(tmp_path), COL_AVG_RETRIEVAL_SCORE_MEAN, out
        )
        assert out.exists() and out.stat().st_size > 0

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"experiment_key": ["a"]})
        with pytest.raises(KeyError):
            plot_latency_by_retriever(df, tmp_path / "x.png")


class TestGenerateAll:
    def test_creates_all_expected_figures(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        output_dir = tmp_path / "figures"
        created = generate_all_comparison_plots(comparison_dir, output_dir)

        assert [p.name for p in created] == EXPECTED_FIGURES
        for path in created:
            assert path.exists() and path.stat().st_size > 0

    def test_missing_comparison_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_all_comparison_plots(tmp_path / "missing", tmp_path / "out")


def _make_semantic_comparison_dir(tmp_path: Path) -> Path:
    """Comparison dir whose summary CSV also has semantic average columns."""
    comparison_dir = tmp_path / "squad_retrievers_semantic"
    comparison_dir.mkdir()

    (comparison_dir / "comparison_summary.csv").write_text(
        "comparison_name,experiment_key,retriever_name,top_k,"
        "num_examples,num_successful,num_failed,"
        "avg_total_latency_ms,avg_retrieval_latency_ms,avg_generation_latency_ms,"
        "avg_retrieval_score_mean,avg_retrieval_redundancy,avg_answer_length_words,"
        "avg_query_context_similarity_mean,avg_answer_context_similarity_mean,"
        "avg_answer_query_similarity\n"
        "cmp,tfidf_top3,tfidf,3,2,2,0,10.0,4.0,6.0,0.5,0.2,12.0,0.7,0.8,0.6\n"
        "cmp,chroma_top3,chroma,3,2,2,0,20.0,9.0,11.0,0.7,0.1,11.0,0.75,0.85,0.65\n",
        encoding="utf-8",
    )
    (comparison_dir / "combined_kpis.csv").write_text(
        "comparison_name,experiment_key,retriever_name,top_k,example_id,"
        "total_latency_ms,retrieval_score_mean\n"
        "cmp,tfidf_top3,tfidf,3,ex0,10.0,0.5\n",
        encoding="utf-8",
    )
    return comparison_dir


class TestSemanticPlots:
    def test_per_metric_semantic_plots_create_pngs(self, tmp_path: Path) -> None:
        summary_df, _ = load_comparison_data(_make_semantic_comparison_dir(tmp_path))

        for plot_func, name in (
            (plot_query_context_similarity_by_retriever, "qc.png"),
            (plot_answer_context_similarity_by_retriever, "ac.png"),
            (plot_answer_query_similarity_by_retriever, "aq.png"),
        ):
            out = tmp_path / name
            plot_func(summary_df, out)
            assert out.exists() and out.stat().st_size > 0

    def test_generate_all_includes_semantic_when_present(
        self, tmp_path: Path
    ) -> None:
        comparison_dir = _make_semantic_comparison_dir(tmp_path)
        created = generate_all_comparison_plots(comparison_dir, tmp_path / "figs")
        names = [p.name for p in created]

        for expected in EXPECTED_FIGURES:
            assert expected in names
        assert "query_context_similarity_by_retriever.png" in names
        assert "answer_context_similarity_by_retriever.png" in names
        assert "answer_query_similarity_by_retriever.png" in names

    def test_generate_all_skips_semantic_when_absent(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        created = generate_all_comparison_plots(comparison_dir, tmp_path / "figs2")
        names = [p.name for p in created]

        assert names == EXPECTED_FIGURES
        assert "query_context_similarity_by_retriever.png" not in names
