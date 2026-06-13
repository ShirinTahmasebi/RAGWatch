"""Instrumented RAG client that emits OpenTelemetry spans."""

from __future__ import annotations

import uuid
from typing import Any

from ragwatch.core.interfaces import BaseGenerator, BaseRAGClient, BaseRetriever
from ragwatch.core.schema import RAGRun
from ragwatch.instrumentation.span_schema import (
    ATTR_ANSWER_LENGTH_CHARS,
    ATTR_CLIENT_CLASS,
    ATTR_GENERATION_LATENCY_MS,
    ATTR_GENERATOR_CLASS,
    ATTR_GENERATOR_NAME,
    ATTR_NUM_RETRIEVED_DOCS,
    ATTR_QUERY,
    ATTR_RETRIEVED_DOC_IDS,
    ATTR_RETRIEVAL_LATENCY_MS,
    ATTR_RETRIEVAL_SCORES,
    ATTR_RETRIEVER_CLASS,
    ATTR_RETRIEVER_NAME,
    ATTR_RUN_ID,
    ATTR_TOP_K,
    ATTR_TOTAL_LATENCY_MS,
    ATTR_VECTOR_DB_TYPE,
    EVENT_GENERATION_COMPLETED,
    EVENT_RAG_RUN_COMPLETED,
    EVENT_RETRIEVAL_COMPLETED,
    SPAN_GENERATION,
    SPAN_RAG_RUN,
    SPAN_RETRIEVAL,
)
from ragwatch.instrumentation.tracer import RAGWatchTracer
from ragwatch.utils.timing import Timer


def _infer_vector_db_type(retriever: BaseRetriever) -> str:
    """Infer the vector DB type from the retriever class name."""
    class_name = retriever.__class__.__name__.lower()
    if "tfidf" in class_name:
        return "tfidf"
    if "chroma" in class_name:
        return "chroma"
    if "qdrant" in class_name:
        return "qdrant"
    if "pgvector" in class_name or "pg_vector" in class_name:
        return "pgvector"
    return "unknown"


class InstrumentedRAGClient(BaseRAGClient):
    """A RAG client wrapper that adds OpenTelemetry tracing to pipeline execution."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        tracer: RAGWatchTracer,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._tracer = tracer

    def run(self, query: str, top_k: int = 5) -> RAGRun:
        """Execute the full RAG pipeline with OpenTelemetry tracing."""
        from opentelemetry import trace

        run_id = str(uuid.uuid4())
        otel_tracer = self._tracer.tracer
        overall_timer = Timer()

        with otel_tracer.start_as_current_span(SPAN_RAG_RUN) as rag_span:
            rag_span.set_attribute(ATTR_RUN_ID, run_id)
            rag_span.set_attribute(ATTR_QUERY, query)
            rag_span.set_attribute(ATTR_TOP_K, top_k)
            rag_span.set_attribute(ATTR_CLIENT_CLASS, self.__class__.__name__)

            with overall_timer:
                # --- Retrieval ---
                retrieval_timer = Timer()
                with otel_tracer.start_as_current_span(SPAN_RETRIEVAL) as ret_span:
                    ret_span.set_attribute(ATTR_RUN_ID, run_id)
                    ret_span.set_attribute(ATTR_QUERY, query)
                    ret_span.set_attribute(ATTR_TOP_K, top_k)
                    ret_span.set_attribute(ATTR_RETRIEVER_CLASS, self._retriever.__class__.__name__)
                    ret_span.set_attribute(ATTR_VECTOR_DB_TYPE, _infer_vector_db_type(self._retriever))

                    with retrieval_timer:
                        retrieved_documents = self._retriever.retrieve(query, top_k=top_k)

                    doc_ids = [rd.document.doc_id for rd in retrieved_documents]
                    scores = [rd.score for rd in retrieved_documents]
                    retriever_name = retrieved_documents[0].retriever_name if retrieved_documents else "unknown"

                    ret_span.set_attribute(ATTR_RETRIEVER_NAME, retriever_name)
                    ret_span.set_attribute(ATTR_NUM_RETRIEVED_DOCS, len(retrieved_documents))
                    ret_span.set_attribute(ATTR_RETRIEVAL_LATENCY_MS, retrieval_timer.elapsed_ms)
                    ret_span.set_attribute(ATTR_RETRIEVED_DOC_IDS, doc_ids)
                    ret_span.set_attribute(ATTR_RETRIEVAL_SCORES, scores)

                    ret_span.add_event(
                        EVENT_RETRIEVAL_COMPLETED,
                        attributes={
                            "num_documents": len(retrieved_documents),
                            "latency_ms": retrieval_timer.elapsed_ms,
                            "retriever": retriever_name,
                        },
                    )

                # --- Generation ---
                generation_timer = Timer()
                with otel_tracer.start_as_current_span(SPAN_GENERATION) as gen_span:
                    gen_span.set_attribute(ATTR_RUN_ID, run_id)
                    gen_span.set_attribute(ATTR_GENERATOR_CLASS, self._generator.__class__.__name__)

                    with generation_timer:
                        generation = self._generator.generate(query, retrieved_documents)

                    gen_span.set_attribute(ATTR_GENERATOR_NAME, generation.generator_name)
                    gen_span.set_attribute(ATTR_ANSWER_LENGTH_CHARS, len(generation.answer))
                    gen_span.set_attribute(ATTR_GENERATION_LATENCY_MS, generation_timer.elapsed_ms)

                    gen_span.add_event(
                        EVENT_GENERATION_COMPLETED,
                        attributes={
                            "generator": generation.generator_name,
                            "latency_ms": generation_timer.elapsed_ms,
                            "answer_length": len(generation.answer),
                        },
                    )

            # Set final attributes on the top-level span
            rag_span.set_attribute(ATTR_NUM_RETRIEVED_DOCS, len(retrieved_documents))
            rag_span.set_attribute(ATTR_TOTAL_LATENCY_MS, overall_timer.elapsed_ms)

            rag_span.add_event(
                EVENT_RAG_RUN_COMPLETED,
                attributes={
                    "total_latency_ms": overall_timer.elapsed_ms,
                    "num_retrieved_documents": len(retrieved_documents),
                },
            )

        return RAGRun(
            run_id=run_id,
            query=query,
            retrieved_documents=retrieved_documents,
            generation=generation,
            latency_ms=overall_timer.elapsed_ms,
            metadata={
                "top_k": top_k,
                "num_retrieved_documents": len(retrieved_documents),
                "retriever": self._retriever.__class__.__name__,
                "generator": self._generator.__class__.__name__,
                "traced": True,
                "retrieval_latency_ms": retrieval_timer.elapsed_ms,
                "generation_latency_ms": generation_timer.elapsed_ms,
                "total_latency_ms": overall_timer.elapsed_ms,
            },
        )


def instrument_client(
    retriever: BaseRetriever,
    generator: BaseGenerator,
    tracer: RAGWatchTracer,
) -> InstrumentedRAGClient:
    """Create an instrumented RAG client with OpenTelemetry tracing.

    This is a convenience factory that creates an InstrumentedRAGClient.
    """
    return InstrumentedRAGClient(
        retriever=retriever,
        generator=generator,
        tracer=tracer,
    )
