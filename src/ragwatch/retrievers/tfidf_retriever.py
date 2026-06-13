"""TF-IDF based document retriever."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ragwatch.core.interfaces import BaseRetriever
from ragwatch.core.schema import Document, RetrievedDocument


class TfidfRetriever(BaseRetriever):
    """Retrieves documents using TF-IDF vectorization and cosine similarity."""

    def __init__(self) -> None:
        self._vectorizer = TfidfVectorizer()
        self._documents: list[Document] = []
        self._tfidf_matrix: np.ndarray | None = None

    def index(self, documents: list[Document]) -> None:
        """Build TF-IDF index from documents."""
        self._documents = documents
        texts = [doc.text for doc in documents]
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve top-k documents by cosine similarity to the query."""
        if self._tfidf_matrix is None or not self._documents:
            return []

        query_vector = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self._tfidf_matrix).flatten()

        num_results = min(top_k, len(self._documents))
        top_indices = np.argsort(similarities)[::-1][:num_results]

        results: list[RetrievedDocument] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(similarities[idx])
            if score <= 0.0:
                break
            results.append(
                RetrievedDocument(
                    document=self._documents[idx],
                    score=score,
                    rank=rank,
                    retriever_name="tfidf",
                )
            )
        return results
