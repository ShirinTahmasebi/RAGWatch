"""Deterministic, rule-based perturbation transforms over a ``RAGDataset``.

Every transform here:
    * never mutates the input ``RAGDataset`` (always returns a new one),
    * is deterministic for a given seed (uses ``random.Random(seed)``),
    * preserves QA answers unless explicitly documented,
    * records useful metadata for later drift analysis.

These are production-relevant degradation rules, not LLM-generated drift.
"""

from __future__ import annotations

import random
import re
from typing import Any

from ragwatch.core.schema import Document, QAExample, RAGDataset
from ragwatch.drift.schema import (
    PERTURBATION_CORPUS_CONTAMINATION,
    PERTURBATION_PARSER_NOISE,
    PERTURBATION_QUERY_DISTRIBUTION_SHIFT,
    PERTURBATION_SOURCE_OUTAGE,
    PRODUCTION_INTERPRETATIONS,
)

# Boilerplate tokens injected to simulate degraded PDF/OCR/HTML parsing.
PARSER_BOILERPLATE_TOKENS: tuple[str, ...] = (
    "page footer",
    "navigation menu",
    "copyright",
    "cookie banner",
    "table of contents",
)

# Generic, topically-plausible phrases for synthetic hard-negative documents.
CONTAMINATION_PHRASES: tuple[str, ...] = (
    "background information",
    "related topic",
    "historical note",
    "general overview",
    "not sufficient to answer",
)

# Conversational and irrelevant-context phrases for query distribution shift.
CONVERSATIONAL_PREFIXES: tuple[str, ...] = (
    "hey, quick question",
    "i was wondering",
    "can you help me figure out",
    "just curious",
)

IRRELEVANT_CONTEXT_SUFFIXES: tuple[str, ...] = (
    "by the way i am in a hurry",
    "asking for a friend",
    "this is probably a silly question",
    "thanks in advance",
)

_WORD_RE = re.compile(r"[A-Za-z]{4,}")


# ---------------------------------------------------------------------------
# Validation and copy helpers
# ---------------------------------------------------------------------------


def _validate_ratio(value: float, name: str) -> None:
    """Validate that ``value`` is a ratio in the closed interval [0.0, 1.0]."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    if not (0.0 <= float(value) <= 1.0):
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")


def _copy_document(doc: Document) -> Document:
    """Return a deep-enough copy of a document (independent metadata dict)."""
    return Document(doc_id=doc.doc_id, text=doc.text, metadata=dict(doc.metadata))


def _copy_qa_example(example: QAExample) -> QAExample:
    """Return a deep-enough copy of a QA example (independent collections)."""
    return QAExample(
        example_id=example.example_id,
        question=example.question,
        answers=list(example.answers),
        metadata=dict(example.metadata),
    )


def _copy_corpus(corpus: list[Document]) -> list[Document]:
    return [_copy_document(doc) for doc in corpus]


def _copy_qa_examples(examples: list[QAExample]) -> list[QAExample]:
    return [_copy_qa_example(ex) for ex in examples]


def copy_dataset(dataset: RAGDataset, *, name: str | None = None) -> RAGDataset:
    """Return a new ``RAGDataset`` with copied corpus and QA examples.

    The input dataset is never mutated. ``name`` overrides the dataset name
    when provided.
    """
    return RAGDataset(
        name=name if name is not None else dataset.name,
        corpus=_copy_corpus(dataset.corpus),
        qa_examples=_copy_qa_examples(dataset.qa_examples),
        metadata=dict(dataset.metadata),
    )


def _select_indices(count: int, ratio: float, rng: random.Random) -> set[int]:
    """Deterministically select ``round(count * ratio)`` indices in [0, count)."""
    k = round(count * ratio)
    if k <= 0:
        return set()
    if k >= count:
        return set(range(count))
    return set(rng.sample(range(count), k))


def _topic_terms(dataset: RAGDataset, rng: random.Random, limit: int = 12) -> list[str]:
    """Sample topically-plausible terms from QA questions (answer-free)."""
    answer_terms = {
        term.lower()
        for ex in dataset.qa_examples
        for ans in ex.answers
        for term in _WORD_RE.findall(ans)
    }
    candidates: list[str] = []
    for ex in dataset.qa_examples:
        for term in _WORD_RE.findall(ex.question):
            if term.lower() not in answer_terms:
                candidates.append(term)
    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[:limit]


# ---------------------------------------------------------------------------
# 1. Corpus contamination / hard-negative injection
# ---------------------------------------------------------------------------


def inject_corpus_contamination(
    dataset: RAGDataset,
    contamination_ratio: float,
    seed: int = 42,
    prefix: str = "contaminated",
) -> RAGDataset:
    """Inject topically-plausible but answer-irrelevant documents.

    Production interpretation: stale pages, wrong tenant/domain docs, near-topic
    chunks, or duplicate noisy pages enter the index as hard negatives.

    The number of injected documents is ``round(len(corpus) * ratio)``. Original
    documents and all QA examples are preserved unchanged.
    """
    _validate_ratio(contamination_ratio, "contamination_ratio")
    rng = random.Random(seed)

    new_corpus = _copy_corpus(dataset.corpus)
    num_inject = round(len(dataset.corpus) * contamination_ratio)

    injected_ids: list[str] = []
    for i in range(num_inject):
        terms = _topic_terms(dataset, rng, limit=8)
        topic = " ".join(terms) if terms else "general subject matter"
        phrases = ", ".join(rng.sample(CONTAMINATION_PHRASES, k=3))
        text = (
            f"{CONTAMINATION_PHRASES[0].capitalize()} regarding {topic}. "
            f"This is a {phrases}. The following content is not sufficient to "
            f"answer the question and is provided only as a historical note."
        )
        doc_id = f"{prefix}_doc_{i}"
        injected_ids.append(doc_id)
        new_corpus.append(
            Document(
                doc_id=doc_id,
                text=text,
                metadata={
                    "source": prefix,
                    "drift_type": PERTURBATION_CORPUS_CONTAMINATION,
                    "synthetic": True,
                    "production_interpretation": PRODUCTION_INTERPRETATIONS[
                        PERTURBATION_CORPUS_CONTAMINATION
                    ],
                },
            )
        )

    metadata = dict(dataset.metadata)
    metadata["drift"] = {
        "perturbation_type": PERTURBATION_CORPUS_CONTAMINATION,
        "contamination_ratio": contamination_ratio,
        "seed": seed,
        "num_injected": num_inject,
        "injected_doc_ids": injected_ids,
        "production_interpretation": PRODUCTION_INTERPRETATIONS[
            PERTURBATION_CORPUS_CONTAMINATION
        ],
    }
    return RAGDataset(
        name=dataset.name,
        corpus=new_corpus,
        qa_examples=_copy_qa_examples(dataset.qa_examples),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 2. Source outage / partial index failure
# ---------------------------------------------------------------------------


def simulate_source_outage(
    dataset: RAGDataset,
    outage_ratio: float,
    seed: int = 42,
) -> RAGDataset:
    """Drop a fraction of documents to simulate a partial index failure.

    Production interpretation: a connector, source, ACL filter, ingestion job,
    or index rebuild silently drops part of the corpus. Removal is source-aware
    when documents carry a ``source`` field, otherwise a deterministic sample is
    removed. QA examples are never removed.
    """
    _validate_ratio(outage_ratio, "outage_ratio")
    rng = random.Random(seed)

    corpus = dataset.corpus
    has_source = any("source" in doc.metadata for doc in corpus)

    removed_ids: set[str] = set()
    if has_source:
        # Source-aware: remove a fraction within each source group so removal
        # respects source boundaries (mirrors a per-connector/per-ACL outage).
        groups: dict[Any, list[int]] = {}
        for idx, doc in enumerate(corpus):
            groups.setdefault(doc.metadata.get("source"), []).append(idx)
        for source in sorted(groups, key=lambda s: str(s)):
            indices = groups[source]
            local = _select_indices(len(indices), outage_ratio, rng)
            for local_idx in local:
                removed_ids.add(corpus[indices[local_idx]].doc_id)
    else:
        selected = _select_indices(len(corpus), outage_ratio, rng)
        removed_ids = {corpus[i].doc_id for i in selected}

    new_corpus = [
        _copy_document(doc) for doc in corpus if doc.doc_id not in removed_ids
    ]

    metadata = dict(dataset.metadata)
    metadata["drift"] = {
        "perturbation_type": PERTURBATION_SOURCE_OUTAGE,
        "outage_ratio": outage_ratio,
        "seed": seed,
        "source_aware": has_source,
        "num_removed": len(removed_ids),
        "removed_doc_ids": sorted(removed_ids),
        "production_interpretation": PRODUCTION_INTERPRETATIONS[
            PERTURBATION_SOURCE_OUTAGE
        ],
    }
    return RAGDataset(
        name=dataset.name,
        corpus=new_corpus,
        qa_examples=_copy_qa_examples(dataset.qa_examples),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 3. Parser noise / ingestion degradation
# ---------------------------------------------------------------------------


def _degrade_text(text: str, rng: random.Random) -> str:
    """Apply deterministic ingestion-style degradation to document text."""
    words = text.split()
    if words:
        # Truncate the tail (lost content from bad chunking/parsing).
        keep = max(1, int(len(words) * 0.6))
        words = words[:keep]
        # Drop a few interior words (dropped tokens during extraction).
        if len(words) > 4:
            drop = _select_indices(len(words), 0.1, rng)
            words = [w for i, w in enumerate(words) if i not in drop]
    # Inject boilerplate tokens (footers/menus that leaked into the text).
    boilerplate = list(rng.sample(PARSER_BOILERPLATE_TOKENS, k=2))
    return " ".join([boilerplate[0]] + words + [boilerplate[1]])


def apply_parser_noise(
    dataset: RAGDataset,
    noise_ratio: float,
    seed: int = 42,
) -> RAGDataset:
    """Degrade the text of a fraction of documents (ingestion degradation).

    Production interpretation: PDF/OCR/HTML parsing, chunking, boilerplate
    removal, or encoding quality gets worse. Document IDs are kept stable and
    non-selected documents are preserved unchanged. QA examples are untouched.
    """
    _validate_ratio(noise_ratio, "noise_ratio")
    rng = random.Random(seed)

    selected = _select_indices(len(dataset.corpus), noise_ratio, rng)
    degraded_ids: list[str] = []
    new_corpus: list[Document] = []
    for idx, doc in enumerate(dataset.corpus):
        if idx in selected:
            degraded_ids.append(doc.doc_id)
            new_metadata = dict(doc.metadata)
            new_metadata["drift_type"] = PERTURBATION_PARSER_NOISE
            new_metadata["original_text_length"] = len(doc.text)
            new_metadata["production_interpretation"] = PRODUCTION_INTERPRETATIONS[
                PERTURBATION_PARSER_NOISE
            ]
            new_corpus.append(
                Document(
                    doc_id=doc.doc_id,
                    text=_degrade_text(doc.text, rng),
                    metadata=new_metadata,
                )
            )
        else:
            new_corpus.append(_copy_document(doc))

    metadata = dict(dataset.metadata)
    metadata["drift"] = {
        "perturbation_type": PERTURBATION_PARSER_NOISE,
        "noise_ratio": noise_ratio,
        "seed": seed,
        "num_degraded": len(degraded_ids),
        "degraded_doc_ids": degraded_ids,
        "production_interpretation": PRODUCTION_INTERPRETATIONS[
            PERTURBATION_PARSER_NOISE
        ],
    }
    return RAGDataset(
        name=dataset.name,
        corpus=new_corpus,
        qa_examples=_copy_qa_examples(dataset.qa_examples),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# 4. Query distribution shift / query noise
# ---------------------------------------------------------------------------


def _introduce_typo(word: str, rng: random.Random) -> str:
    """Swap two adjacent characters in a word to mimic a typo."""
    if len(word) < 4:
        return word
    i = rng.randrange(len(word) - 1)
    chars = list(word)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def _shift_question(question: str, rng: random.Random) -> str:
    """Apply deterministic query-noise perturbations to a question."""
    words = question.split()

    # Typo in one eligible word.
    eligible = [i for i, w in enumerate(words) if len(w) >= 4]
    if eligible:
        idx = rng.choice(eligible)
        words[idx] = _introduce_typo(words[idx], rng)

    # Drop one non-critical (short) word if available, keeping the question
    # answerable.
    short = [i for i, w in enumerate(words) if len(w) <= 3]
    if short and len(words) > 3:
        drop_idx = rng.choice(short)
        words = [w for i, w in enumerate(words) if i != drop_idx]

    prefix = rng.choice(CONVERSATIONAL_PREFIXES)
    suffix = rng.choice(IRRELEVANT_CONTEXT_SUFFIXES)
    return f"{prefix}, {' '.join(words)} ({suffix})"


def apply_query_distribution_shift(
    dataset: RAGDataset,
    shift_ratio: float,
    seed: int = 42,
) -> RAGDataset:
    """Perturb the questions of a fraction of QA examples.

    Production interpretation: users start asking with different wording, typos,
    conversational phrasing, or irrelevant extra tokens. Answers and example IDs
    are preserved; the original question is stored in metadata.
    """
    _validate_ratio(shift_ratio, "shift_ratio")
    rng = random.Random(seed)

    selected = _select_indices(len(dataset.qa_examples), shift_ratio, rng)
    shifted_ids: list[str] = []
    new_examples: list[QAExample] = []
    for idx, ex in enumerate(dataset.qa_examples):
        if idx in selected:
            shifted_ids.append(ex.example_id)
            new_metadata = dict(ex.metadata)
            new_metadata["drift_type"] = PERTURBATION_QUERY_DISTRIBUTION_SHIFT
            new_metadata["original_question"] = ex.question
            new_metadata["production_interpretation"] = PRODUCTION_INTERPRETATIONS[
                PERTURBATION_QUERY_DISTRIBUTION_SHIFT
            ]
            new_examples.append(
                QAExample(
                    example_id=ex.example_id,
                    question=_shift_question(ex.question, rng),
                    answers=list(ex.answers),
                    metadata=new_metadata,
                )
            )
        else:
            new_examples.append(_copy_qa_example(ex))

    metadata = dict(dataset.metadata)
    metadata["drift"] = {
        "perturbation_type": PERTURBATION_QUERY_DISTRIBUTION_SHIFT,
        "shift_ratio": shift_ratio,
        "seed": seed,
        "num_shifted": len(shifted_ids),
        "shifted_example_ids": shifted_ids,
        "production_interpretation": PRODUCTION_INTERPRETATIONS[
            PERTURBATION_QUERY_DISTRIBUTION_SHIFT
        ],
    }
    return RAGDataset(
        name=dataset.name,
        corpus=_copy_corpus(dataset.corpus),
        qa_examples=new_examples,
        metadata=metadata,
    )
