"""Span attribute schema constants for RAGWatch instrumentation."""

# Span names
SPAN_RAG_RUN = "ragwatch.rag_run"
SPAN_RETRIEVAL = "ragwatch.retrieval"
SPAN_GENERATION = "ragwatch.generation"

# Event names
EVENT_RETRIEVAL_COMPLETED = "retrieval.completed"
EVENT_GENERATION_COMPLETED = "generation.completed"
EVENT_RAG_RUN_COMPLETED = "rag_run.completed"

# Attribute keys
ATTR_RUN_ID = "ragwatch.run_id"
ATTR_QUERY = "ragwatch.query"
ATTR_TOP_K = "ragwatch.top_k"
ATTR_NUM_RETRIEVED_DOCS = "ragwatch.num_retrieved_documents"
ATTR_TOTAL_LATENCY_MS = "ragwatch.total_latency_ms"
ATTR_CLIENT_CLASS = "ragwatch.client_class"

ATTR_RETRIEVER_NAME = "ragwatch.retriever_name"
ATTR_RETRIEVER_CLASS = "ragwatch.retriever_class"
ATTR_RETRIEVAL_LATENCY_MS = "ragwatch.retrieval_latency_ms"
ATTR_RETRIEVED_DOC_IDS = "ragwatch.retrieved_doc_ids"
ATTR_RETRIEVAL_SCORES = "ragwatch.retrieval_scores"
ATTR_VECTOR_DB_TYPE = "ragwatch.vector_db_type"

ATTR_GENERATOR_NAME = "ragwatch.generator_name"
ATTR_GENERATOR_CLASS = "ragwatch.generator_class"
ATTR_ANSWER_LENGTH_CHARS = "ragwatch.answer_length_chars"
ATTR_GENERATION_LATENCY_MS = "ragwatch.generation_latency_ms"
