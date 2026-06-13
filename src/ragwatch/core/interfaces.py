"""Abstract base classes defining the RAGWatch component interfaces."""

from abc import ABC, abstractmethod

from ragwatch.core.schema import (
    Document,
    GenerationResult,
    QAExample,
    RAGRun,
    RetrievedDocument,
)


class BaseDatasetLoader(ABC):
    """Interface for loading datasets."""

    @abstractmethod
    def load_corpus(self) -> list[Document]:
        """Load the document corpus."""
        ...

    @abstractmethod
    def load_qa_examples(self) -> list[QAExample]:
        """Load question-answer examples."""
        ...


class BaseRetriever(ABC):
    """Interface for document retrievers."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Index a list of documents for retrieval."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve the top-k most relevant documents for a query."""
        ...


class BaseGenerator(ABC):
    """Interface for answer generators."""

    @abstractmethod
    def generate(
        self, query: str, retrieved_documents: list[RetrievedDocument]
    ) -> GenerationResult:
        """Generate an answer given a query and retrieved documents."""
        ...


class BaseRAGClient(ABC):
    """Interface for end-to-end RAG pipeline clients."""

    @abstractmethod
    def run(self, query: str, top_k: int = 5) -> RAGRun:
        """Execute the full RAG pipeline for a query."""
        ...
