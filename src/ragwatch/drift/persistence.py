"""Persist drifted dataset snapshots for reproducibility and inspection.

For a given :class:`DriftedDataset`, this module writes a self-contained
snapshot directory:

    output_dir/
        corpus.jsonl        # one JSON object per document
        qa_examples.jsonl   # one JSON object per QA example
        manifest.json       # scenario config + counts + change details

The snapshot captures exactly what the drift transforms produced, including all
drift metadata, so a scenario can be reproduced and inspected later.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragwatch.core.schema import RAGDataset
from ragwatch.drift.schema import DriftedDataset

# Keys in dataset.metadata["drift"] mapped to manifest change fields. Each entry
# maps (count_key, ids_key) from the transform metadata to the manifest fields.
# Only the keys present for a given perturbation type are emitted.
_DRIFT_CHANGE_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("num_injected", "injected_doc_ids", "num_added_documents", "added_doc_ids"),
    ("num_removed", "removed_doc_ids", "num_removed_documents", "removed_doc_ids"),
    (
        "num_degraded",
        "degraded_doc_ids",
        "num_modified_documents",
        "modified_doc_ids",
    ),
    (
        "num_shifted",
        "shifted_example_ids",
        "num_modified_queries",
        "modified_example_ids",
    ),
)


def export_corpus_jsonl(dataset: RAGDataset, path: str | Path) -> None:
    """Write the corpus to JSONL, one document object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc in dataset.corpus:
            record = {
                "doc_id": doc.doc_id,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            f.write(json.dumps(record, default=str) + "\n")


def export_qa_examples_jsonl(dataset: RAGDataset, path: str | Path) -> None:
    """Write the QA examples to JSONL, one example object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in dataset.qa_examples:
            record = {
                "example_id": example.example_id,
                "question": example.question,
                "answers": example.answers,
                "metadata": example.metadata,
            }
            f.write(json.dumps(record, default=str) + "\n")


def _build_manifest(drifted_dataset: DriftedDataset) -> dict[str, Any]:
    """Build the manifest payload from a drifted dataset.

    Always includes core scenario fields and counts. Change details (added /
    removed / modified) are included only when present in the drift metadata.
    """
    dataset = drifted_dataset.dataset
    config = drifted_dataset.config
    scenario_meta = drifted_dataset.metadata
    drift_meta: dict[str, Any] = dict(dataset.metadata.get("drift", {}))

    num_final_documents = len(dataset.corpus)
    num_qa_examples = len(dataset.qa_examples)

    manifest: dict[str, Any] = {
        "original_name": drifted_dataset.original_name,
        "scenario_name": drifted_dataset.scenario_name,
        "perturbation_type": config.perturbation_type,
        "severity": config.severity,
        "random_seed": config.random_seed,
        "production_interpretation": scenario_meta.get("production_interpretation"),
        "num_documents": num_final_documents,
        "num_qa_examples": num_qa_examples,
        "num_final_documents": num_final_documents,
        "metadata": scenario_meta,
        "config_metadata": config.metadata,
    }

    # Derive change details from the transform's drift metadata when available.
    num_added = 0
    num_removed = 0
    for count_key, ids_key, manifest_count, manifest_ids in _DRIFT_CHANGE_FIELDS:
        if count_key in drift_meta:
            manifest[manifest_count] = drift_meta[count_key]
        if ids_key in drift_meta:
            manifest[manifest_ids] = drift_meta[ids_key]
        if count_key == "num_injected" and count_key in drift_meta:
            num_added = int(drift_meta[count_key])
        if count_key == "num_removed" and count_key in drift_meta:
            num_removed = int(drift_meta[count_key])

    # Reconstruct the original document count from net additions/removals.
    manifest["num_original_documents"] = num_final_documents - num_added + num_removed

    return manifest


def export_drift_manifest(
    drifted_dataset: DriftedDataset, path: str | Path
) -> None:
    """Write the drift manifest as stable, human-readable JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(drifted_dataset)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def export_drifted_dataset(
    drifted_dataset: DriftedDataset,
    output_dir: str | Path,
) -> None:
    """Export a full drifted dataset snapshot to ``output_dir``.

    Creates ``corpus.jsonl``, ``qa_examples.jsonl``, and ``manifest.json``.
    Existing files are overwritten.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_corpus_jsonl(drifted_dataset.dataset, output_dir / "corpus.jsonl")
    export_qa_examples_jsonl(
        drifted_dataset.dataset, output_dir / "qa_examples.jsonl"
    )
    export_drift_manifest(drifted_dataset, output_dir / "manifest.json")
