"""Central KPI catalog — single source of truth for all KPI metadata."""

from dataclasses import dataclass
from enum import StrEnum


class KPICategory(StrEnum):
    RETRIEVAL_QUALITY = "retrieval_quality"
    GENERATION_QUALITY = "generation_quality"
    SEMANTIC_QUALITY = "semantic_quality"
    RUNTIME = "runtime"
    DB_INDEX_STATS = "db_index_stats"


class KPIStage(StrEnum):
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    END_TO_END = "end_to_end"
    DATABASE = "database"


class KPISource(StrEnum):
    RAG_RUN = "rag_run"
    RETRIEVED_DOCUMENTS = "retrieved_documents"
    GENERATION = "generation"
    METADATA = "metadata"
    POSTGRES = "postgres"
    PGVECTOR = "pgvector"
    SEMANTIC_EMBEDDING_MODEL = "semantic_embedding_model"


class KPIId(StrEnum):
    # Retrieval quality
    NUM_RETRIEVED_DOCUMENTS = "num_retrieved_documents"
    RETRIEVAL_SCORE_MIN = "retrieval_score_min"
    RETRIEVAL_SCORE_MAX = "retrieval_score_max"
    RETRIEVAL_SCORE_MEAN = "retrieval_score_mean"
    RETRIEVAL_SCORE_STD = "retrieval_score_std"
    RETRIEVAL_SCORE_RANGE = "retrieval_score_range"
    RETRIEVAL_SCORE_GAP_TOP1_TOP2 = "retrieval_score_gap_top1_top2"
    CONTEXT_LENGTH_CHARS = "context_length_chars"
    UNIQUE_RETRIEVED_SOURCES = "unique_retrieved_sources"
    RETRIEVAL_REDUNDANCY = "retrieval_redundancy"

    # Generation quality
    ANSWER_LENGTH_CHARS = "answer_length_chars"
    ANSWER_LENGTH_WORDS = "answer_length_words"
    ANSWER_TO_CONTEXT_LENGTH_RATIO = "answer_to_context_length_ratio"

    # Semantic quality (optional, embedding-based)
    QUERY_CONTEXT_SIMILARITY_MEAN = "query_context_similarity_mean"
    QUERY_CONTEXT_SIMILARITY_MAX = "query_context_similarity_max"
    ANSWER_CONTEXT_SIMILARITY_MEAN = "answer_context_similarity_mean"
    ANSWER_CONTEXT_SIMILARITY_MAX = "answer_context_similarity_max"
    ANSWER_QUERY_SIMILARITY = "answer_query_similarity"

    # Runtime
    TOTAL_LATENCY_MS = "total_latency_ms"
    RETRIEVAL_LATENCY_MS = "retrieval_latency_ms"
    GENERATION_LATENCY_MS = "generation_latency_ms"

    # DB index stats
    DB_DOCUMENT_COUNT = "db_document_count"
    DB_EMBEDDING_COUNT = "db_embedding_count"
    DB_COLLECTION_COUNT = "db_collection_count"
    DB_AVG_DOCUMENT_LENGTH = "db_avg_document_length"
    DB_AVG_EMBEDDING_DIM = "db_avg_embedding_dim"
    DB_COLLECTIONS = "db_collections"
    DB_DOCUMENTS_PER_SOURCE = "db_documents_per_source"


@dataclass(frozen=True)
class KPIDefinition:
    id: KPIId
    category: KPICategory
    stage: KPIStage
    source: KPISource
    description: str


KPI_CATALOG: dict[KPIId, KPIDefinition] = {
    # --- Retrieval quality ---
    KPIId.NUM_RETRIEVED_DOCUMENTS: KPIDefinition(
        id=KPIId.NUM_RETRIEVED_DOCUMENTS,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Number of documents returned by the retriever.",
    ),
    KPIId.RETRIEVAL_SCORE_MIN: KPIDefinition(
        id=KPIId.RETRIEVAL_SCORE_MIN,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Minimum retrieval score across all retrieved documents.",
    ),
    KPIId.RETRIEVAL_SCORE_MAX: KPIDefinition(
        id=KPIId.RETRIEVAL_SCORE_MAX,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Maximum retrieval score across all retrieved documents.",
    ),
    KPIId.RETRIEVAL_SCORE_MEAN: KPIDefinition(
        id=KPIId.RETRIEVAL_SCORE_MEAN,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Mean retrieval score across all retrieved documents.",
    ),
    KPIId.RETRIEVAL_SCORE_STD: KPIDefinition(
        id=KPIId.RETRIEVAL_SCORE_STD,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Population standard deviation of retrieval scores.",
    ),
    KPIId.RETRIEVAL_SCORE_RANGE: KPIDefinition(
        id=KPIId.RETRIEVAL_SCORE_RANGE,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Difference between max and min retrieval scores.",
    ),
    KPIId.RETRIEVAL_SCORE_GAP_TOP1_TOP2: KPIDefinition(
        id=KPIId.RETRIEVAL_SCORE_GAP_TOP1_TOP2,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Score difference between rank-1 and rank-2 documents.",
    ),
    KPIId.CONTEXT_LENGTH_CHARS: KPIDefinition(
        id=KPIId.CONTEXT_LENGTH_CHARS,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Sum of character lengths of all retrieved document texts.",
    ),
    KPIId.UNIQUE_RETRIEVED_SOURCES: KPIDefinition(
        id=KPIId.UNIQUE_RETRIEVED_SOURCES,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Number of unique sources in retrieved documents.",
    ),
    KPIId.RETRIEVAL_REDUNDANCY: KPIDefinition(
        id=KPIId.RETRIEVAL_REDUNDANCY,
        category=KPICategory.RETRIEVAL_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.RETRIEVED_DOCUMENTS,
        description="Mean pairwise Jaccard similarity of 3-gram word shingles.",
    ),
    # --- Generation quality ---
    KPIId.ANSWER_LENGTH_CHARS: KPIDefinition(
        id=KPIId.ANSWER_LENGTH_CHARS,
        category=KPICategory.GENERATION_QUALITY,
        stage=KPIStage.GENERATION,
        source=KPISource.GENERATION,
        description="Number of characters in the generated answer.",
    ),
    KPIId.ANSWER_LENGTH_WORDS: KPIDefinition(
        id=KPIId.ANSWER_LENGTH_WORDS,
        category=KPICategory.GENERATION_QUALITY,
        stage=KPIStage.GENERATION,
        source=KPISource.GENERATION,
        description="Number of whitespace-separated words in the generated answer.",
    ),
    KPIId.ANSWER_TO_CONTEXT_LENGTH_RATIO: KPIDefinition(
        id=KPIId.ANSWER_TO_CONTEXT_LENGTH_RATIO,
        category=KPICategory.GENERATION_QUALITY,
        stage=KPIStage.GENERATION,
        source=KPISource.GENERATION,
        description="Answer character length divided by total retrieved context character length.",
    ),
    # --- Semantic quality (optional, embedding-based) ---
    KPIId.QUERY_CONTEXT_SIMILARITY_MEAN: KPIDefinition(
        id=KPIId.QUERY_CONTEXT_SIMILARITY_MEAN,
        category=KPICategory.SEMANTIC_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.SEMANTIC_EMBEDDING_MODEL,
        description="Mean cosine similarity between the query and each retrieved document.",
    ),
    KPIId.QUERY_CONTEXT_SIMILARITY_MAX: KPIDefinition(
        id=KPIId.QUERY_CONTEXT_SIMILARITY_MAX,
        category=KPICategory.SEMANTIC_QUALITY,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.SEMANTIC_EMBEDDING_MODEL,
        description="Maximum cosine similarity between the query and any retrieved document.",
    ),
    KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN: KPIDefinition(
        id=KPIId.ANSWER_CONTEXT_SIMILARITY_MEAN,
        category=KPICategory.SEMANTIC_QUALITY,
        stage=KPIStage.GENERATION,
        source=KPISource.SEMANTIC_EMBEDDING_MODEL,
        description="Mean cosine similarity between the generated answer and each retrieved document.",
    ),
    KPIId.ANSWER_CONTEXT_SIMILARITY_MAX: KPIDefinition(
        id=KPIId.ANSWER_CONTEXT_SIMILARITY_MAX,
        category=KPICategory.SEMANTIC_QUALITY,
        stage=KPIStage.GENERATION,
        source=KPISource.SEMANTIC_EMBEDDING_MODEL,
        description="Maximum cosine similarity between the generated answer and any retrieved document.",
    ),
    KPIId.ANSWER_QUERY_SIMILARITY: KPIDefinition(
        id=KPIId.ANSWER_QUERY_SIMILARITY,
        category=KPICategory.SEMANTIC_QUALITY,
        stage=KPIStage.END_TO_END,
        source=KPISource.SEMANTIC_EMBEDDING_MODEL,
        description="Cosine similarity between the generated answer and the query.",
    ),
    # --- Runtime ---
    KPIId.TOTAL_LATENCY_MS: KPIDefinition(
        id=KPIId.TOTAL_LATENCY_MS,
        category=KPICategory.RUNTIME,
        stage=KPIStage.END_TO_END,
        source=KPISource.METADATA,
        description="Total end-to-end latency of the RAG run in milliseconds.",
    ),
    KPIId.RETRIEVAL_LATENCY_MS: KPIDefinition(
        id=KPIId.RETRIEVAL_LATENCY_MS,
        category=KPICategory.RUNTIME,
        stage=KPIStage.RETRIEVAL,
        source=KPISource.METADATA,
        description="Retrieval stage latency in milliseconds (from metadata).",
    ),
    KPIId.GENERATION_LATENCY_MS: KPIDefinition(
        id=KPIId.GENERATION_LATENCY_MS,
        category=KPICategory.RUNTIME,
        stage=KPIStage.GENERATION,
        source=KPISource.METADATA,
        description="Generation stage latency in milliseconds (from metadata).",
    ),
    # --- DB index stats ---
    KPIId.DB_DOCUMENT_COUNT: KPIDefinition(
        id=KPIId.DB_DOCUMENT_COUNT,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.POSTGRES,
        description="Total number of documents in ragwatch_documents table.",
    ),
    KPIId.DB_EMBEDDING_COUNT: KPIDefinition(
        id=KPIId.DB_EMBEDDING_COUNT,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.PGVECTOR,
        description="Total number of embeddings in ragwatch_embeddings table.",
    ),
    KPIId.DB_COLLECTION_COUNT: KPIDefinition(
        id=KPIId.DB_COLLECTION_COUNT,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.PGVECTOR,
        description="Number of distinct collections in ragwatch_embeddings.",
    ),
    KPIId.DB_AVG_DOCUMENT_LENGTH: KPIDefinition(
        id=KPIId.DB_AVG_DOCUMENT_LENGTH,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.POSTGRES,
        description="Average character length of text in ragwatch_documents.",
    ),
    KPIId.DB_AVG_EMBEDDING_DIM: KPIDefinition(
        id=KPIId.DB_AVG_EMBEDDING_DIM,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.PGVECTOR,
        description="Average embedding dimension in ragwatch_embeddings.",
    ),
    KPIId.DB_COLLECTIONS: KPIDefinition(
        id=KPIId.DB_COLLECTIONS,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.PGVECTOR,
        description="Comma-separated list of collection names in ragwatch_embeddings.",
    ),
    KPIId.DB_DOCUMENTS_PER_SOURCE: KPIDefinition(
        id=KPIId.DB_DOCUMENTS_PER_SOURCE,
        category=KPICategory.DB_INDEX_STATS,
        stage=KPIStage.DATABASE,
        source=KPISource.POSTGRES,
        description="Document counts per source as JSON string.",
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_kpi_definition(kpi_id: KPIId | str) -> KPIDefinition:
    """Look up a KPI definition by ID. Raises KeyError if not found."""
    if isinstance(kpi_id, str):
        kpi_id = KPIId(kpi_id)
    return KPI_CATALOG[kpi_id]


def list_kpis() -> list[KPIDefinition]:
    """Return all KPI definitions in catalog order."""
    return list(KPI_CATALOG.values())


def list_kpis_by_category(category: KPICategory | str) -> list[KPIDefinition]:
    """Return KPI definitions filtered by category."""
    cat = KPICategory(category) if isinstance(category, str) else category
    return [d for d in KPI_CATALOG.values() if d.category == cat]


def list_kpis_by_stage(stage: KPIStage | str) -> list[KPIDefinition]:
    """Return KPI definitions filtered by stage."""
    stg = KPIStage(stage) if isinstance(stage, str) else stage
    return [d for d in KPI_CATALOG.values() if d.stage == stg]


def list_kpis_by_source(source: KPISource | str) -> list[KPIDefinition]:
    """Return KPI definitions filtered by source."""
    src = KPISource(source) if isinstance(source, str) else source
    return [d for d in KPI_CATALOG.values() if d.source == src]
