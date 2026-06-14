"""Tests for the drift dashboard data loaders (no Streamlit UI)."""

import json
from pathlib import Path

import pytest

from ragwatch.dashboard.data_loader import (
    list_drift_scenario_dirs,
    load_drift_corpus,
    load_drift_manifest,
    load_drift_qa_examples,
    load_drift_summary,
    load_scenario_runs,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(rec) + "\n" for rec in records),
        encoding="utf-8",
    )


def _make_scenario_dir(
    drift_dir: Path,
    scenario_name: str,
    perturbation_type: str,
) -> Path:
    scenario_dir = drift_dir / scenario_name
    snapshot_dir = scenario_dir / "drifted_dataset"
    snapshot_dir.mkdir(parents=True)

    _write_jsonl(
        snapshot_dir / "corpus.jsonl",
        [
            {"doc_id": "d0", "text": "doc zero", "metadata": {"source": "wiki"}},
            {"doc_id": "d1", "text": "doc one", "metadata": {"source": "wiki"}},
        ],
    )
    _write_jsonl(
        snapshot_dir / "qa_examples.jsonl",
        [
            {
                "example_id": "ex0",
                "question": "Q0?",
                "answers": ["a0"],
                "metadata": {"original_question": "orig Q0?"},
            },
            {
                "example_id": "ex1",
                "question": "Q1?",
                "answers": ["a1"],
                "metadata": {},
            },
        ],
    )
    (snapshot_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scenario_name": scenario_name,
                "perturbation_type": perturbation_type,
                "severity": 0.3,
                "random_seed": 7,
                "production_interpretation": "noisy corpus",
                "num_original_documents": 2,
                "num_final_documents": 2,
            }
        ),
        encoding="utf-8",
    )

    _write_jsonl(
        scenario_dir / "runs.jsonl",
        [
            {
                "example_id": "ex0",
                "question": "Q0?",
                "answer": "a0",
                "retrieved_doc_ids": ["d0"],
                "retrieved_scores": [0.9],
                "retrieved_ranks": [0],
            },
            {
                "example_id": "ex1",
                "question": "Q1?",
                "answer": "a1",
                "retrieved_doc_ids": ["d1"],
                "retrieved_scores": [0.8],
                "retrieved_ranks": [0],
            },
        ],
    )
    return scenario_dir


def _make_drift_dir(tmp_path: Path) -> Path:
    drift_dir = tmp_path / "squad_tfidf"
    drift_dir.mkdir()

    _make_scenario_dir(drift_dir, "clean", "none")
    _make_scenario_dir(drift_dir, "corpus_contamination_30", "corpus_contamination")

    (drift_dir / "drift_summary.csv").write_text(
        "scenario_name,perturbation_type,severity,production_interpretation,"
        "num_documents,num_examples,avg_retrieval_score_mean,"
        "avg_retrieval_redundancy,avg_answer_length_words,avg_total_latency_ms\n"
        "clean,none,0.0,clean baseline,2,2,0.85,0.1,5.0,1.2\n"
        "corpus_contamination_30,corpus_contamination,0.3,noisy corpus,3,2,0.7,0.2,4.0,1.5\n",
        encoding="utf-8",
    )
    return drift_dir


class TestDriftLoaders:
    def test_load_drift_summary(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        df = load_drift_summary(drift_dir)
        assert len(df) == 2
        assert set(df["scenario_name"]) == {"clean", "corpus_contamination_30"}
        assert "avg_retrieval_score_mean" in df.columns

    def test_list_drift_scenario_dirs(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        scenarios = list_drift_scenario_dirs(drift_dir)
        names = [p.name for p in scenarios]
        assert names == ["clean", "corpus_contamination_30"]

    def test_list_drift_scenario_dirs_missing(self, tmp_path: Path) -> None:
        assert list_drift_scenario_dirs(tmp_path / "nope") == []

    def test_load_drift_manifest(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        manifest = load_drift_manifest(drift_dir / "corpus_contamination_30")
        assert manifest["perturbation_type"] == "corpus_contamination"
        assert manifest["severity"] == 0.3

    def test_load_drift_qa_examples(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        df = load_drift_qa_examples(drift_dir / "clean")
        assert len(df) == 2
        assert set(df["example_id"]) == {"ex0", "ex1"}

    def test_load_drift_corpus(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        df = load_drift_corpus(drift_dir / "clean")
        assert len(df) == 2
        assert set(df["doc_id"]) == {"d0", "d1"}

    def test_load_scenario_runs(self, tmp_path: Path) -> None:
        drift_dir = _make_drift_dir(tmp_path)
        df = load_scenario_runs(drift_dir / "clean")
        assert len(df) == 2
        assert set(df["example_id"]) == {"ex0", "ex1"}


class TestMissingDriftFiles:
    def test_missing_summary(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_drift_summary(tmp_path / "empty")

    def test_missing_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_drift_manifest(tmp_path / "empty")

    def test_missing_qa_examples(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_drift_qa_examples(tmp_path / "empty")

    def test_missing_corpus(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_drift_corpus(tmp_path / "empty")

    def test_missing_runs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_scenario_runs(tmp_path / "empty")
