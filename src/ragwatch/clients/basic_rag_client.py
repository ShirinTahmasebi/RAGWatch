"""Basic end-to-end RAG client."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from ragwatch.core.interfaces import BaseGenerator, BaseRAGClient, BaseRetriever
from ragwatch.core.schema import RAGRun
from ragwatch.utils.timing import Timer

if TYPE_CHECKING:
    from ragwatch.instrumentation.tracer import RAGWatchTracer


class BasicRAGClient(BaseRAGClient):
    """A minimal RAG client that orchestrates retrieval and generation.

    If a tracer is provided, an InstrumentedRAGClient is used internally
    to emit OpenTelemetry spans. Otherwise, runs without tracing.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        tracer: RAGWatchTracer | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._tracer = tracer

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        """Execute the full RAG pipeline: retrieve then generate."""
        if self._tracer is not None:
            from ragwatch.instrumentation.instrumented_client import InstrumentedRAGClient

            instrumented = InstrumentedRAGClient(
                retriever=self._retriever,
                generator=self._generator,
                tracer=self._tracer,
            )
            return instrumented.run(query, top_k=top_k)

        retrieval_timer = Timer()
        generation_timer = Timer()
        overall_timer = Timer()

        with overall_timer:
            with retrieval_timer:
                retrieved_documents = self._retriever.retrieve(query, top_k=top_k)
            with generation_timer:
                generation = self._generator.generate(query, retrieved_documents)

        return RAGRun(
            run_id=str(uuid.uuid4()),
            query=query,
            retrieved_documents=retrieved_documents,
            generation=generation,
            latency_ms=overall_timer.elapsed_ms,
            metadata={
                "top_k": top_k,
                "num_retrieved_documents": len(retrieved_documents),
                "retriever": self._retriever.__class__.__name__,
                "generator": self._generator.__class__.__name__,
                "retrieval_latency_ms": retrieval_timer.elapsed_ms,
                "generation_latency_ms": generation_timer.elapsed_ms,
                "total_latency_ms": overall_timer.elapsed_ms,
            },
        )
