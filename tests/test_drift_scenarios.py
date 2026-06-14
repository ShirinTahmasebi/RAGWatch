"""Tests for drift scenario builders.

Uses small toy ``RAGDataset`` objects; no external services or API keys.
"""

import pytest

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.drift.scenarios import create_drift_scenario, create_drift_scenarios
from ragwatch.drift.schema import (
    PERTURBATION_CLEAN,
    DriftedDataset,
    DriftScenarioConfig,
)

EXPECTED_SCENARIO_NAMES = [
    "clean",
    "corpus_contamination_10",
    "corpus_contamination_30",
    "source_outage_10",
    "source_outage_30",
    "parser_noise_10",
    "query_shift_10",
    "query_shift_30",
]


def _toy_dataset() -> RAGDataset:
    corpus = [
        Document(
            doc_id=f"doc_{i}",
            text=f"Background about region {i} and its scientific developments.",
            metadata={"source": "alpha" if i % 2 == 0 else "beta"},
        )
        for i in range(10)
    ]
    qa = [
        QAExample(
            example_id=f"q_{i}",
            question=f"What happened in region {i}?",
            answers=[f"event {i}"],
        )
        for i in range(10)
    ]
    return RAGDataset(name="toy", corpus=corpus, qa_examples=qa)


def test_create_drift_scenarios_includes_expected_names() -> None:
    dataset = _toy_dataset()
    scenarios = create_drift_scenarios(dataset)
    assert [s.scenario_name for s in scenarios] == EXPECTED_SCENARIO_NAMES


def test_clean_scenario_is_a_copy() -> None:
    dataset = _toy_dataset()
    scenarios = create_drift_scenarios(dataset)
    clean = next(s for s in scenarios if s.scenario_name == "clean")
    assert isinstance(clean, DriftedDataset)
    assert clean.config.perturbation_type == PERTURBATION_CLEAN
    assert len(clean.dataset.corpus) == len(dataset.corpus)
    # It is a copy, not the same object.
    assert clean.dataset is not dataset
    assert clean.dataset.corpus[0] is not dataset.corpus[0]


def test_scenarios_do_not_mutate_original_dataset() -> None:
    dataset = _toy_dataset()
    original_len = len(dataset.corpus)
    create_drift_scenarios(dataset)
    assert len(dataset.corpus) == original_len


def test_create_drift_scenario_unknown_type_raises() -> None:
    dataset = _toy_dataset()
    config = DriftScenarioConfig(
        scenario_name="bogus",
        perturbation_type="totally_unknown_perturbation",
        severity=0.5,
        random_seed=1,
    )
    with pytest.raises(ValueError, match="Unknown perturbation_type"):
        create_drift_scenario(dataset, config)


def test_scenario_metadata_has_production_interpretation() -> None:
    dataset = _toy_dataset()
    scenarios = create_drift_scenarios(dataset)
    for scenario in scenarios:
        assert scenario.metadata["production_interpretation"]
        assert scenario.metadata["num_documents"] == len(scenario.dataset.corpus)
        assert scenario.metadata["num_examples"] == len(scenario.dataset.qa_examples)


def test_contamination_scenarios_grow_corpus() -> None:
    dataset = _toy_dataset()
    scenarios = {s.scenario_name: s for s in create_drift_scenarios(dataset)}
    assert len(scenarios["corpus_contamination_30"].dataset.corpus) > len(
        dataset.corpus
    )
    assert len(scenarios["source_outage_30"].dataset.corpus) < len(dataset.corpus)
