"""Data preparation utilities for the HotpotQA pipeline."""
from __future__ import annotations

from functools import lru_cache
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ragwatch.utils import EnvKeys

logger = logging.getLogger(__name__)

DEFAULT_SPLIT = "validation"
DEFAULT_SAMPLE_SIZE = 25

_DUMMY_QUESTIONS = [
    {
        "id": "q1",
        "question": "Who is Barack Obama?",
        "answer": "Barack Obama is the 44th President of the United States.",
    },
    {
        "id": "q2",
        "question": "What is the capital of France?",
        "answer": "Paris is the capital and most populous city of France.",
    },
]

_DUMMY_DOCS = [
    {
        "doc_id": "d1",
        "title": "Obama",
        "text": "Barack Obama was the 44th president of the United States...",
    },
    {
        "doc_id": "d2",
        "title": "Paris",
        "text": "Paris is the capital and most populous city of France...",
    },
]


def load_questions(sample_size: Optional[int] = None) -> List[Dict[str, str]]:
    """Load questions/answers from the HotpotQA split (fallback to dummy data)."""

    dataset = _maybe_get_dataset()
    if dataset is None:
        return _DUMMY_QUESTIONS[: sample_size or len(_DUMMY_QUESTIONS)]

    limit = sample_size or _sample_size()
    records: List[Dict[str, str]] = []
    for example in dataset:
        records.append(
            {
                "id": example["id"],
                "question": example["question"],
                "answer": example["answer"],
                "supporting_facts": example.get("supporting_facts"),
            }
        )
        if limit and len(records) >= limit:
            break

    return records or _DUMMY_QUESTIONS[: limit or len(_DUMMY_QUESTIONS)]


def build_document_corpus(sample_size: Optional[int] = None) -> List[Dict[str, str]]:
    """Build a corpus of documents derived from HotpotQA contexts (or dummy data)."""

    dataset = _maybe_get_dataset()
    if dataset is None:
        return _DUMMY_DOCS[: sample_size or len(_DUMMY_DOCS)]

    limit = sample_size or _sample_size()
    docs: Dict[str, Dict[str, str]] = {}

    for example in dataset:
        for title, sentences in _iter_context_entries(example.get("context")):
            if not title:
                continue
            doc_id = str(title)
            if doc_id in docs:
                continue
            text = _join_sentences(sentences)
            if not text:
                continue
            docs[doc_id] = {"doc_id": doc_id, "title": title, "text": text}
            if limit and len(docs) >= limit:
                return list(docs.values())

    return list(docs.values()) or _DUMMY_DOCS[: limit or len(_DUMMY_DOCS)]


def _iter_context_entries(raw_context: Any) -> Iterable[Tuple[str, Any]]:
    if not raw_context:
        return []

    if isinstance(raw_context, dict):
        titles = raw_context.get("title")
        sentences_list = raw_context.get("sentences")
        if isinstance(titles, list) and isinstance(sentences_list, list):
            return [(title, sentences) for title, sentences in zip(titles, sentences_list)]
        return []

    entries: List[Tuple[str, Any]] = []
    for item in raw_context:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            entries.append((item[0], item[1]))
        elif isinstance(item, dict) and "title" in item:
            entries.append((item.get("title"), item.get("sentences") or item.get("text")))
    return entries


def _join_sentences(sentences: Any) -> str:
    if sentences is None:
        return ""
    if isinstance(sentences, str):
        return sentences
    if isinstance(sentences, (list, tuple)):
        parts: List[str] = []
        for chunk in sentences:
            if isinstance(chunk, str):
                parts.append(chunk)
            elif isinstance(chunk, (list, tuple)):
                parts.append(" ".join(str(token) for token in chunk))
            elif chunk is not None:
                parts.append(str(chunk))
        return " ".join(part for part in parts if part)
    return str(sentences)


def clear_hotpotqa_cache() -> None:
    """Reset cached dataset loaders (useful for tests)."""

    _load_hotpotqa_split.cache_clear()


def _maybe_get_dataset():
    if _use_dummy_data():
        return None
    try:
        return _load_hotpotqa_split(_target_split())
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("Falling back to dummy HotpotQA data: %s", exc)
        return None


def _target_split() -> str:
    return os.getenv(EnvKeys.HOTPOTQA_SPLIT, DEFAULT_SPLIT)


def _sample_size() -> int:
    raw = os.getenv(EnvKeys.HOTPOTQA_SAMPLE_SIZE, str(DEFAULT_SAMPLE_SIZE))
    try:
        value = int(raw)
        return value if value > 0 else DEFAULT_SAMPLE_SIZE
    except ValueError:
        return DEFAULT_SAMPLE_SIZE


def _use_dummy_data() -> bool:
    return os.getenv(EnvKeys.HOTPOTQA_USE_DUMMY_DATA, "false").lower() == "true"


@lru_cache(maxsize=1)
def _load_hotpotqa_split(split: str):
    from datasets import load_dataset

    return load_dataset("hotpot_qa", "distractor", split=split)
