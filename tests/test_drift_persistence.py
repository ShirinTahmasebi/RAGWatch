"""Tests for drifted dataset snapshot persistence.

Uses toy ``RAGDataset`` and ``DriftedDataset`` objects; no external services or
API keys required.
"""

import json
from pathlib import Path

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.drift.persistence import (
    export_corpus_jsonl,
    export_drift_manifest,
    export_drifted_dataset,
    export_qa_examples_jsonl,
)
from ragwatch.drift.schema import (
    PERTURBATION_CORPUS_CONTAMINATION,
    DriftedDataset,
    DriftScenarioConfig,
)


def _toy_drifted() -> DriftedDataset:
    corpus = [
        Document(
            doc_id="doc_0",
            text="Original document about chemistry.",
            metadata={"source": "squad", "title": "t0"},
        ),
        Document(
            doc_id="contaminated_doc_0",
            text="Background information regarding general subject matter.",
            metadata={
                "source": "contaminated",
                "drift_type": PERTURBATION_CORPUS_CONTAMINATION,
                "synthetic": True,
                "production_interpretation": "hard_negative_or_stale_content_entered_index",
            },
        ),
    ]
    qa = [
        QAExample(
            example_id="q_0",
            question="What is chemistry?",
            answers=["the study of matter"],
            metadata={"title": "t0", "original_question": "What is chemistry?"},
        )
    ]
    dataset = RAGDataset(
        name="toy",
        corpus=corpus,
        qa_examples=qa,
        metadata={
            "drift": {
                "perturbation_type": PERTURBATION_CORPUS_CONTAMINATION,
                "contamination_ratio": 0.5,
                "seed": 42,
                "num_injected": 1,
                "injected_doc_ids": ["contaminated_doc_0"],
                "production_interpretation": "hard_negative_or_stale_content_entered_index",
            }
        },
    )
    config = DriftScenarioConfig(
        scenario_name="corpus_contamination_30",
        perturbation_type=PERTURBATION_CORPUS_CONTAMINATION,
        severity=0.3,
        random_seed=42,
        metadata={"note": "toy"},
    )
    return DriftedDataset(
        original_name="toy",
        scenario_name="corpus_contamination_30",
        dataset=dataset,
        config=config,
        metadata={
            "perturbation_type": PERTURBATION_CORPUS_CONTAMINATION,
            "severity": 0.3,
            "random_seed": 42,
            "production_interpretation": "hard_negative_or_stale_content_entered_index",
            "num_documents": 2,
            "num_examples": 1,
        },
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_export_corpus_jsonl_one_row_per_document(tmp_path) -> None:
    drifted = _toy_drifted()
    path = tmp_path / "corpus.jsonl"
    export_corpus_jsonl(drifted.dataset, path)
    rows = _read_jsonl(path)
    assert len(rows) == len(drifted.dataset.corpus)
    assert {r["doc_id"] for r in rows} == {"doc_0", "contaminated_doc_0"}


def test_export_qa_examples_jsonl_one_row_per_example(tmp_path) -> None:
    drifted = _toy_drifted()
    path = tmp_path / "qa_examples.jsonl"
    export_qa_examples_jsonl(drifted.dataset, path)
    rows = _read_jsonl(path)
    assert len(rows) == len(drifted.dataset.qa_examples)
    assert rows[0]["example_id"] == "q_0"


def test_document_metadata_is_preserved(tmp_path) -> None:
    drifted = _toy_drifted()
    path = tmp_path / "corpus.jsonl"
    export_corpus_jsonl(drifted.dataset, path)
    rows = {r["doc_id"]: r for r in _read_jsonl(path)}
    injected = rows["contaminated_doc_0"]
    assert injected["metadata"]["synthetic"] is True
    assert injected["metadata"]["drift_type"] == PERTURBATION_CORPUS_CONTAMINATION


def test_qa_metadata_is_preserved(tmp_path) -> None:
    drifted = _toy_drifted()
    path = tmp_path / "qa_examples.jsonl"
    export_qa_examples_jsonl(drifted.dataset, path)
    rows = _read_jsonl(path)
    assert rows[0]["metadata"]["original_question"] == "What is chemistry?"
    assert rows[0]["answers"] == ["the study of matter"]


def test_manifest_writes_scenario_config_fields(tmp_path) -> None:
    drifted = _toy_drifted()
    path = tmp_path / "manifest.json"
    export_drift_manifest(drifted, path)
    manifest = json.loads(path.read_text())
    assert manifest["original_name"] == "toy"
    assert manifest["scenario_name"] == "corpus_contamination_30"
    assert manifest["perturbation_type"] == PERTURBATION_CORPUS_CONTAMINATION
    assert manifest["severity"] == 0.3
    assert manifest["random_seed"] == 42
    assert manifest["config_metadata"] == {"note": "toy"}
    assert manifest["production_interpretation"]


def test_manifest_includes_counts_and_change_details(tmp_path) -> None:
    drifted = _toy_drifted()
    path = tmp_path / "manifest.json"
    export_drift_manifest(drifted, path)
    manifest = json.loads(path.read_text())
    assert manifest["num_documents"] == 2
    assert manifest["num_final_documents"] == 2
    assert manifest["num_qa_examples"] == 1
    # Change details derived from drift metadata.
    assert manifest["num_added_documents"] == 1
    assert manifest["added_doc_ids"] == ["contaminated_doc_0"]
    # Original = final - added + removed = 2 - 1 + 0 = 1.
    assert manifest["num_original_documents"] == 1


def test_export_drifted_dataset_creates_all_three_files(tmp_path) -> None:
    drifted = _toy_drifted()
    out = tmp_path / "drifted_dataset"
    export_drifted_dataset(drifted, out)
    assert (out / "corpus.jsonl").exists()
    assert (out / "qa_examples.jsonl").exists()
    assert (out / "manifest.json").exists()


def test_exported_files_are_valid_json(tmp_path) -> None:
    drifted = _toy_drifted()
    out = tmp_path / "drifted_dataset"
    export_drifted_dataset(drifted, out)
    # Manifest is valid JSON.
    json.loads((out / "manifest.json").read_text())
    # Every JSONL line parses.
    for name in ("corpus.jsonl", "qa_examples.jsonl"):
        for line in (out / name).read_text().splitlines():
            if line:
                json.loads(line)


def test_output_directory_created_automatically(tmp_path) -> None:
    drifted = _toy_drifted()
    out = tmp_path / "nested" / "deep" / "drifted_dataset"
    assert not out.exists()
    export_drifted_dataset(drifted, out)
    assert out.is_dir()


def test_repeated_export_overwrites_safely(tmp_path) -> None:
    drifted = _toy_drifted()
    out = tmp_path / "drifted_dataset"
    export_drifted_dataset(drifted, out)
    export_drifted_dataset(drifted, out)  # second export should not error
    rows = _read_jsonl(out / "corpus.jsonl")
    # Not appended/duplicated; still one row per document.
    assert len(rows) == len(drifted.dataset.corpus)
