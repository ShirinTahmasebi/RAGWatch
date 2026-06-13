"""Base class for vector DB retrievers."""

from ragwatch.core.interfaces import BaseRetriever
from ragwatch.core.schema import Document, RetrievedDocument
from ragwatch.embeddings.base import BaseEmbeddingModel


class BaseVectorRetriever(BaseRetriever):
    """Base class for vector database retrievers.

    Subclasses implement the storage/query layer; this class handles
    the common embedding model reference.
    """

    def __init__(
        self, embedding_model: BaseEmbeddingModel, collection_name: str
    ) -> None:
        self._embedding_model = embedding_model
        self._collection_name = collection_name
        self._documents: dict[str, Document] = {}  # doc_id -> Document
