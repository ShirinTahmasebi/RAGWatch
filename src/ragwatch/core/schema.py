"""Structured data models for RAGWatch."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document in the corpus."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QAExample:
    """A question-answer example from a dataset."""

    example_id: str
    question: str
    answers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    """A document retrieved by a retriever, with scoring information."""

    document: Document
    score: float
    rank: int
    retriever_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    """The output of a generator."""

    answer: str
    generator_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGRun:
    """A complete RAG pipeline execution."""

    run_id: str
    query: str
    retrieved_documents: list[RetrievedDocument]
    generation: GenerationResult
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGDataset:
    """A bundled dataset containing a corpus and QA examples."""

    name: str
    corpus: list[Document]
    qa_examples: list[QAExample]
    metadata: dict[str, Any] = field(default_factory=dict)
