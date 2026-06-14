"""Tests for deterministic drift perturbation transforms.

These use small toy ``RAGDataset`` objects and never require Hugging Face,
OpenAI, Azure, Postgres, Chroma, Qdrant, Streamlit, Plotly, or API keys.
"""

import pytest

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.drift.schema import (
    PERTURBATION_CORPUS_CONTAMINATION,
    PERTURBATION_PARSER_NOISE,
    PERTURBATION_QUERY_DISTRIBUTION_SHIFT,
    PERTURBATION_SOURCE_OUTAGE,
)
from ragwatch.drift.transforms import (
    apply_parser_noise,
    apply_query_distribution_shift,
    inject_corpus_contamination,
    simulate_source_outage,
)


def _toy_dataset() -> RAGDataset:
    corpus = [
        Document(
            doc_id=f"doc_{i}",
            text=(
                f"The history of region {i} includes notable scientific "
                f"discoveries about chemistry and physics over many years."
            ),
            metadata={"source": "alpha" if i % 2 == 0 else "beta", "title": f"t{i}"},
        )
        for i in range(10)
    ]
    qa = [
        QAExample(
            example_id=f"q_{i}",
            question=f"What discovery happened in region {i} during that period?",
            answers=[f"discovery {i}"],
            metadata={"title": f"t{i}"},
        )
        for i in range(10)
    ]
    return RAGDataset(name="toy", corpus=corpus, qa_examples=qa, metadata={"k": "v"})


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------


def test_transforms_do_not_mutate_original_dataset() -> None:
    dataset = _toy_dataset()
    original_corpus_len = len(dataset.corpus)
    original_qa_len = len(dataset.qa_examples)
    original_first_doc_text = dataset.corpus[0].text
    original_first_question = dataset.qa_examples[0].question

    inject_corpus_contamination(dataset, 0.5, seed=1)
    simulate_source_outage(dataset, 0.5, seed=1)
    apply_parser_noise(dataset, 0.5, seed=1)
    apply_query_distribution_shift(dataset, 0.5, seed=1)

    assert len(dataset.corpus) == original_corpus_len
    assert len(dataset.qa_examples) == original_qa_len
    assert dataset.corpus[0].text == original_first_doc_text
    assert dataset.qa_examples[0].question == original_first_question


# ---------------------------------------------------------------------------
# Corpus contamination
# ---------------------------------------------------------------------------


def test_corpus_contamination_increases_corpus_size() -> None:
    dataset = _toy_dataset()
    drifted = inject_corpus_contamination(dataset, 0.3, seed=1)
    # round(10 * 0.3) == 3 injected docs.
    assert len(drifted.corpus) == len(dataset.corpus) + 3


def test_contaminated_documents_have_drift_metadata() -> None:
    dataset = _toy_dataset()
    drifted = inject_corpus_contamination(dataset, 0.5, seed=1)
    injected = [d for d in drifted.corpus if d.metadata.get("synthetic")]
    assert injected
    for doc in injected:
        assert doc.metadata["drift_type"] == PERTURBATION_CORPUS_CONTAMINATION
        assert doc.metadata["synthetic"] is True
        assert "production_interpretation" in doc.metadata


def test_corpus_contamination_preserves_qa_examples() -> None:
    dataset = _toy_dataset()
    drifted = inject_corpus_contamination(dataset, 0.5, seed=1)
    assert [e.question for e in drifted.qa_examples] == [
        e.question for e in dataset.qa_examples
    ]


# ---------------------------------------------------------------------------
# Source outage
# ---------------------------------------------------------------------------


def test_source_outage_decreases_corpus_size() -> None:
    dataset = _toy_dataset()
    drifted = simulate_source_outage(dataset, 0.5, seed=1)
    assert len(drifted.corpus) < len(dataset.corpus)
    # QA examples are never removed.
    assert len(drifted.qa_examples) == len(dataset.qa_examples)


def test_source_outage_records_removed_ids() -> None:
    dataset = _toy_dataset()
    drifted = simulate_source_outage(dataset, 0.5, seed=1)
    drift_meta = drifted.metadata["drift"]
    assert drift_meta["perturbation_type"] == PERTURBATION_SOURCE_OUTAGE
    assert drift_meta["num_removed"] == len(drift_meta["removed_doc_ids"])
    assert drift_meta["num_removed"] > 0


def test_source_outage_without_source_metadata() -> None:
    corpus = [Document(doc_id=f"d{i}", text=f"text {i}") for i in range(8)]
    qa = [QAExample(example_id=f"q{i}", question="q?", answers=["a"]) for i in range(4)]
    dataset = RAGDataset(name="nosrc", corpus=corpus, qa_examples=qa)
    drifted = simulate_source_outage(dataset, 0.5, seed=1)
    assert drifted.metadata["drift"]["source_aware"] is False
    assert len(drifted.corpus) == 4


# ---------------------------------------------------------------------------
# Parser noise
# ---------------------------------------------------------------------------


def test_parser_noise_changes_selected_document_text() -> None:
    dataset = _toy_dataset()
    drifted = apply_parser_noise(dataset, 0.5, seed=1)
    changed = [
        d
        for d in drifted.corpus
        if d.metadata.get("drift_type") == PERTURBATION_PARSER_NOISE
    ]
    assert changed
    original_by_id = {d.doc_id: d.text for d in dataset.corpus}
    for doc in changed:
        assert doc.text != original_by_id[doc.doc_id]
        assert "original_text_length" in doc.metadata


def test_parser_noise_preserves_unselected_documents() -> None:
    dataset = _toy_dataset()
    drifted = apply_parser_noise(dataset, 0.5, seed=1)
    original_by_id = {d.doc_id: d.text for d in dataset.corpus}
    untouched = [
        d
        for d in drifted.corpus
        if d.metadata.get("drift_type") != PERTURBATION_PARSER_NOISE
    ]
    assert untouched
    for doc in untouched:
        assert doc.text == original_by_id[doc.doc_id]


# ---------------------------------------------------------------------------
# Query distribution shift
# ---------------------------------------------------------------------------


def test_query_shift_changes_questions_but_preserves_answers() -> None:
    dataset = _toy_dataset()
    drifted = apply_query_distribution_shift(dataset, 0.5, seed=1)
    original_by_id = {e.example_id: e for e in dataset.qa_examples}
    shifted = [
        e
        for e in drifted.qa_examples
        if e.metadata.get("drift_type") == PERTURBATION_QUERY_DISTRIBUTION_SHIFT
    ]
    assert shifted
    for ex in shifted:
        original = original_by_id[ex.example_id]
        assert ex.question != original.question
        assert ex.answers == original.answers
        assert ex.metadata["original_question"] == original.question


# ---------------------------------------------------------------------------
# Ratio validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ratio", [-0.1, 1.1, 2.0, -5.0])
def test_ratio_validation_rejects_out_of_range(ratio: float) -> None:
    dataset = _toy_dataset()
    with pytest.raises(ValueError):
        inject_corpus_contamination(dataset, ratio, seed=1)
    with pytest.raises(ValueError):
        simulate_source_outage(dataset, ratio, seed=1)
    with pytest.raises(ValueError):
        apply_parser_noise(dataset, ratio, seed=1)
    with pytest.raises(ValueError):
        apply_query_distribution_shift(dataset, ratio, seed=1)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_output() -> None:
    dataset = _toy_dataset()
    a = simulate_source_outage(dataset, 0.4, seed=7)
    b = simulate_source_outage(dataset, 0.4, seed=7)
    assert [d.doc_id for d in a.corpus] == [d.doc_id for d in b.corpus]

    qa_a = apply_query_distribution_shift(dataset, 0.6, seed=7)
    qa_b = apply_query_distribution_shift(dataset, 0.6, seed=7)
    assert [e.question for e in qa_a.qa_examples] == [
        e.question for e in qa_b.qa_examples
    ]


def test_different_seeds_can_select_different_docs() -> None:
    dataset = _toy_dataset()
    a = simulate_source_outage(dataset, 0.3, seed=1)
    b = simulate_source_outage(dataset, 0.3, seed=999)
    a_removed = set(a.metadata["drift"]["removed_doc_ids"])
    b_removed = set(b.metadata["drift"]["removed_doc_ids"])
    assert a_removed != b_removed


# ---------------------------------------------------------------------------
# Production interpretation metadata
# ---------------------------------------------------------------------------


def test_production_interpretation_present_in_metadata() -> None:
    dataset = _toy_dataset()
    for transform, ptype in (
        (inject_corpus_contamination, PERTURBATION_CORPUS_CONTAMINATION),
        (simulate_source_outage, PERTURBATION_SOURCE_OUTAGE),
        (apply_parser_noise, PERTURBATION_PARSER_NOISE),
        (apply_query_distribution_shift, PERTURBATION_QUERY_DISTRIBUTION_SHIFT),
    ):
        drifted = transform(dataset, 0.3, seed=1)
        drift_meta = drifted.metadata["drift"]
        assert drift_meta["perturbation_type"] == ptype
        assert drift_meta["production_interpretation"]
