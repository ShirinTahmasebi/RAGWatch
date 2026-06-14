"""Tests for the paper-quality drift plotting utilities.

These tests use small fake drift_summary.csv files and the non-interactive
matplotlib backend; they require no Streamlit, Plotly, Hugging Face, vector DBs,
Postgres, OpenTelemetry, or API keys.
"""

from pathlib import Path

import pandas as pd
import pytest

from ragwatch.plots.drift_plots import (
    COL_AVG_RETRIEVAL_SCORE_MEAN,
    generate_all_drift_plots,
    load_drift_summary,
    plot_answer_length_by_drift_scenario,
    plot_answer_context_similarity_by_drift_scenario,
    plot_answer_query_similarity_by_drift_scenario,
    plot_latency_by_drift_scenario,
    plot_metric_by_drift_scenario,
    plot_query_context_similarity_by_drift_scenario,
    plot_retrieval_redundancy_by_drift_scenario,
    plot_retrieval_score_by_drift_scenario,
)

EXPECTED_BASE_FIGURES = [
    "retrieval_score_by_drift_scenario.png",
    "retrieval_redundancy_by_drift_scenario.png",
    "answer_length_by_drift_scenario.png",
    "latency_by_drift_scenario.png",
]

EXPECTED_SEMANTIC_FIGURES = [
    "query_context_similarity_by_drift_scenario.png",
    "answer_context_similarity_by_drift_scenario.png",
    "answer_query_similarity_by_drift_scenario.png",
]

_BASE_HEADER = (
    "scenario_name,perturbation_type,severity,production_interpretation,"
    "num_documents,num_examples,num_successful,num_failed,"
    "avg_total_latency_ms,avg_retrieval_score_mean,"
    "avg_retrieval_redundancy,avg_answer_length_words"
)
_BASE_ROWS = (
    "clean,none,0.0,clean baseline,10,20,20,0,0.36,0.85,0.10,5.0\n"
    "corpus_contamination_30,corpus_contamination,0.3,noisy corpus,"
    "13,20,20,0,0.30,0.70,0.20,4.0\n"
)


def _make_drift_dir(tmp_path: Path) -> Path:
    """Create a fake drift output directory with a base drift_summary.csv."""
    drift_dir = tmp_path / "squad_tfidf"
    drift_dir.mkdir()
    (drift_dir / "drift_summary.csv").write_text(
        _BASE_HEADER + "\n" + _BASE_ROWS, encoding="utf-8"
    )
    return drift_dir


def _make_semantic_drift_dir(tmp_path: Path) -> Path:
    """Create a fake drift output directory whose summary has semantic columns."""
    drift_dir = tmp_path / "squad_tfidf_semantic"
    drift_dir.mkdir()
    (drift_dir / "drift_summary.csv").write_text(
        _BASE_HEADER
        + ",avg_query_context_similarity_mean,"
        "avg_answer_context_similarity_mean,avg_answer_query_similarity\n"
        "clean,none,0.0,clean baseline,10,20,20,0,0.36,0.85,0.10,5.0,"
        "0.70,0.80,0.60\n"
        "corpus_contamination_30,corpus_contamination,0.3,noisy corpus,"
        "13,20,20,0,0.30,0.70,0.20,4.0,0.65,0.75,0.55\n",
        encoding="utf-8",
    )
    return drift_dir


def _summary_df(tmp_path: Path) -> pd.DataFrame:
    return load_drift_summary(_make_drift_dir(tmp_path))


class TestLoadDriftSummary:
    def test_loads_csv(self, tmp_path: Path) -> None:
        df = load_drift_summary(_make_drift_dir(tmp_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "scenario_name" in df.columns
        assert COL_AVG_RETRIEVAL_SCORE_MEAN in df.columns

    def test_missing_summary_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_drift_summary(tmp_path / "missing")


class TestMetricHelper:
    def test_plot_metric_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "metric.png"
        plot_metric_by_drift_scenario(
            _summary_df(tmp_path),
            metric_column=COL_AVG_RETRIEVAL_SCORE_MEAN,
            output_path=out,
            title="Title",
            ylabel="Ylabel",
        )
        assert out.exists() and out.stat().st_size > 0

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"scenario_name": ["clean"]})
        with pytest.raises(KeyError):
            plot_metric_by_drift_scenario(
                df,
                metric_column="avg_total_latency_ms",
                output_path=tmp_path / "x.png",
                title="t",
                ylabel="y",
            )


class TestBasePlots:
    def test_retrieval_score_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "score.png"
        plot_retrieval_score_by_drift_scenario(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_redundancy_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "redundancy.png"
        plot_retrieval_redundancy_by_drift_scenario(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_answer_length_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "answer_length.png"
        plot_answer_length_by_drift_scenario(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0

    def test_latency_creates_png(self, tmp_path: Path) -> None:
        out = tmp_path / "latency.png"
        plot_latency_by_drift_scenario(_summary_df(tmp_path), out)
        assert out.exists() and out.stat().st_size > 0


class TestGenerateAll:
    def test_creates_base_figures(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        output_dir = tmp_path / "figures"
        created = generate_all_drift_plots(drift_dir, output_dir)

        assert [p.name for p in created] == EXPECTED_BASE_FIGURES
        for path in created:
            assert path.exists() and path.stat().st_size > 0

    def test_default_output_dir_is_created(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        created = generate_all_drift_plots(drift_dir)

        figures_dir = drift_dir / "figures"
        assert figures_dir.is_dir()
        assert all(p.parent == figures_dir for p in created)
        assert [p.name for p in created] == EXPECTED_BASE_FIGURES

    def test_skips_semantic_when_absent(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        created = generate_all_drift_plots(drift_dir, tmp_path / "figs")
        names = [p.name for p in created]

        assert names == EXPECTED_BASE_FIGURES
        for semantic in EXPECTED_SEMANTIC_FIGURES:
            assert semantic not in names

    def test_includes_semantic_when_present(self, tmp_path: Path) -> None:
        drift_dir = _make_semantic_drift_dir(tmp_path)
        created = generate_all_drift_plots(drift_dir, tmp_path / "figs2")
        names = [p.name for p in created]

        for expected in EXPECTED_BASE_FIGURES:
            assert expected in names
        for semantic in EXPECTED_SEMANTIC_FIGURES:
            assert semantic in names
        for path in created:
            assert path.exists() and path.stat().st_size > 0

    def test_missing_drift_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            generate_all_drift_plots(tmp_path / "missing", tmp_path / "out")


class TestSemanticPerMetricPlots:
    def test_semantic_plots_create_pngs(self, tmp_path: Path) -> None:
        summary_df = load_drift_summary(_make_semantic_drift_dir(tmp_path))

        for plot_func, name in (
            (plot_query_context_similarity_by_drift_scenario, "qc.png"),
            (plot_answer_context_similarity_by_drift_scenario, "ac.png"),
            (plot_answer_query_similarity_by_drift_scenario, "aq.png"),
        ):
            out = tmp_path / name
            plot_func(summary_df, out)
            assert out.exists() and out.stat().st_size > 0
