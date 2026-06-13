"""Tests for the JSONL dataset loader."""

import json
import tempfile
from pathlib import Path

from ragwatch.datasets.jsonl_loader import JsonlLoader


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_load_corpus() -> None:
    corpus_data = [
        {"doc_id": "d1", "text": "Hello world.", "metadata": {"source": "test"}},
        {"doc_id": "d2", "text": "Goodbye world.", "metadata": {"source": "test"}},
    ]
    qa_data = [
        {"example_id": "q1", "question": "Hi?", "answers": ["Hello"], "metadata": {}}
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.jsonl"
        qa_path = Path(tmpdir) / "qa.jsonl"
        _write_jsonl(corpus_path, corpus_data)
        _write_jsonl(qa_path, qa_data)

        loader = JsonlLoader(corpus_path=corpus_path, qa_path=qa_path)
        documents = loader.load_corpus()

        assert len(documents) == 2
        assert documents[0].doc_id == "d1"
        assert documents[0].text == "Hello world."
        assert documents[0].metadata == {"source": "test"}
        assert documents[1].doc_id == "d2"


def test_load_qa_examples() -> None:
    corpus_data = [{"doc_id": "d1", "text": "Text.", "metadata": {}}]
    qa_data = [
        {
            "example_id": "q1",
            "question": "What?",
            "answers": ["Answer1", "Answer2"],
            "metadata": {"dataset": "test"},
        },
        {
            "example_id": "q2",
            "question": "Why?",
            "answers": ["Because"],
            "metadata": {},
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.jsonl"
        qa_path = Path(tmpdir) / "qa.jsonl"
        _write_jsonl(corpus_path, corpus_data)
        _write_jsonl(qa_path, qa_data)

        loader = JsonlLoader(corpus_path=corpus_path, qa_path=qa_path)
        examples = loader.load_qa_examples()

        assert len(examples) == 2
        assert examples[0].example_id == "q1"
        assert examples[0].question == "What?"
        assert examples[0].answers == ["Answer1", "Answer2"]
        assert examples[0].metadata == {"dataset": "test"}


def test_load_corpus_skips_blank_lines() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        corpus_path = Path(tmpdir) / "corpus.jsonl"
        qa_path = Path(tmpdir) / "qa.jsonl"

        # Write corpus with blank lines
        with corpus_path.open("w") as f:
            f.write('{"doc_id": "d1", "text": "A.", "metadata": {}}\n')
            f.write("\n")
            f.write('{"doc_id": "d2", "text": "B.", "metadata": {}}\n')

        _write_jsonl(qa_path, [])

        loader = JsonlLoader(corpus_path=corpus_path, qa_path=qa_path)
        documents = loader.load_corpus()
        assert len(documents) == 2
