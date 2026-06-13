"""JSONL dataset loader for corpus and QA files."""

import json
from pathlib import Path

from ragwatch.core.interfaces import BaseDatasetLoader
from ragwatch.core.schema import Document, QAExample


class JsonlLoader(BaseDatasetLoader):
    """Loads documents and QA examples from JSONL files."""

    def __init__(self, corpus_path: Path | str, qa_path: Path | str) -> None:
        self._corpus_path = Path(corpus_path)
        self._qa_path = Path(qa_path)

    def load_corpus(self) -> list[Document]:
        """Load documents from a JSONL file."""
        documents: list[Document] = []
        with self._corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                documents.append(
                    Document(
                        doc_id=data["doc_id"],
                        text=data["text"],
                        metadata=data.get("metadata", {}),
                    )
                )
        return documents

    def load_qa_examples(self) -> list[QAExample]:
        """Load QA examples from a JSONL file."""
        examples: list[QAExample] = []
        with self._qa_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                examples.append(
                    QAExample(
                        example_id=data["example_id"],
                        question=data["question"],
                        answers=data.get("answers", []),
                        metadata=data.get("metadata", {}),
                    )
                )
        return examples
