"""Heuristic-based answer generator (no LLM calls)."""

from ragwatch.core.interfaces import BaseGenerator
from ragwatch.core.schema import GenerationResult, RetrievedDocument


class HeuristicGenerator(BaseGenerator):
    """Generates answers by extracting text from the top retrieved document.

    This is a placeholder generator for testing the RAG pipeline.
    It does not call any external LLM.
    """

    def generate(
        self, query: str, retrieved_documents: list[RetrievedDocument]
    ) -> GenerationResult:
        """Generate an answer from the top retrieved document."""
        if not retrieved_documents:
            answer = "No relevant documents found."
        else:
            top_doc = retrieved_documents[0]
            answer = f"Based on the retrieved context: {top_doc.document.text}"

        return GenerationResult(
            answer=answer,
            generator_name="heuristic",
            metadata={"num_documents_used": len(retrieved_documents)},
        )
