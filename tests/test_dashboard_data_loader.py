"""Tests for the dashboard data-loading utilities (no Streamlit UI)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ragwatch.dashboard.data_loader import (
    list_experiment_dirs,
    load_combined_kpis,
    load_comparison_summary,
    load_comparison_summary_json,
    load_runs_jsonl,
)


def _make_comparison_dir(tmp_path: Path) -> Path:
    """Create a small fake comparison output directory."""
    comparison_dir = tmp_path / "squad_retrievers"
    comparison_dir.mkdir()

    (comparison_dir / "combined_kpis.csv").write_text(
        "comparison_name,experiment_key,retriever_name,top_k,example_id,total_latency_ms\n"
        "cmp,tfidf_top3,tfidf,3,ex0,1.5\n"
        "cmp,tfidf_top3,tfidf,3,ex1,2.5\n",
        encoding="utf-8",
    )

    (comparison_dir / "comparison_summary.csv").write_text(
        "comparison_name,experiment_key,retriever_name,top_k,num_examples,num_successful,num_failed\n"
        "cmp,tfidf_top3,tfidf,3,2,2,0\n",
        encoding="utf-8",
    )

    (comparison_dir / "comparison_summary.json").write_text(
        json.dumps(
            {
                "comparison_name": "cmp",
                "dataset_name": "squad_validation",
                "top_k_values": [3],
                "num_experiments": 1,
                "experiments": [{"experiment_key": "tfidf_top3"}],
            }
        ),
        encoding="utf-8",
    )

    experiments_dir = comparison_dir / "experiments"
    exp_dir = experiments_dir / "tfidf_top3"
    exp_dir.mkdir(parents=True)
    (exp_dir / "runs.jsonl").write_text(
        json.dumps({"example_id": "ex0", "answer": "a0", "run_id": "r0"})
        + "\n"
        + json.dumps({"example_id": "ex1", "answer": "a1", "run_id": "r1"})
        + "\n",
        encoding="utf-8",
    )

    return comparison_dir


class TestLoaders:
    def test_load_combined_kpis(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        df = load_combined_kpis(comparison_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "total_latency_ms" in df.columns
        assert df["retriever_name"].iloc[0] == "tfidf"

    def test_load_comparison_summary(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        df = load_comparison_summary(comparison_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df["num_successful"].iloc[0] == 2

    def test_load_comparison_summary_json(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        data = load_comparison_summary_json(comparison_dir)
        assert data["comparison_name"] == "cmp"
        assert data["top_k_values"] == [3]
        assert data["num_experiments"] == 1

    def test_list_experiment_dirs(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        dirs = list_experiment_dirs(comparison_dir)
        assert len(dirs) == 1
        assert dirs[0].name == "tfidf_top3"

    def test_list_experiment_dirs_missing_returns_empty(self, tmp_path: Path) -> None:
        empty = tmp_path / "no_experiments"
        empty.mkdir()
        assert list_experiment_dirs(empty) == []

    def test_load_runs_jsonl(self, tmp_path: Path) -> None:
        comparison_dir = _make_comparison_dir(tmp_path)
        exp_dir = comparison_dir / "experiments" / "tfidf_top3"
        df = load_runs_jsonl(exp_dir)
        assert len(df) == 2
        assert set(df["example_id"]) == {"ex0", "ex1"}
        assert df["answer"].iloc[0] == "a0"


class TestMissingFiles:
    def test_load_combined_kpis_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_combined_kpis(tmp_path)

    def test_load_comparison_summary_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_comparison_summary(tmp_path)

    def test_load_comparison_summary_json_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_comparison_summary_json(tmp_path)

    def test_load_runs_jsonl_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_runs_jsonl(tmp_path)
