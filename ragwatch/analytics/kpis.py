"""Shared KPI definitions and thresholds for RAGWatch analytics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

DEFAULT_ALERT_STD_THRESHOLD = 1.5
DEFAULT_TIMEOUT_MS = 10_000
DEFAULT_SLOW_REQUEST_MS = 3_000
ALERT_HISTORY_WINDOW = 8


@dataclass(frozen=True)
class KPIGroup:
    key: str
    label: str


@dataclass(frozen=True)
class KPIConfig:
    key: str
    label: str
    description: str
    chart: str = "line"
    group: str = "retriever_confidence"


KPI_GROUPS: List[KPIGroup] = [
    KPIGroup("retriever_confidence", "Retriever Confidence"),
    KPIGroup("document_diversity", "Document Diversity"),
    KPIGroup("context_utilization", "Context Utilization"),
    KPIGroup("prompt_truncation", "Prompt Truncation"),
    KPIGroup("retriever_generator_drift", "Retriever/Generator Drift"),
    KPIGroup("latency_timeout", "Latency & Timeout"),
]


KPI_CONFIGS: Dict[str, KPIConfig] = {
    "score_at_rank": KPIConfig(
        "score_at_rank",
        "Score@Rank",
        "Top-1 similarity score returned by the retriever.",
        group="retriever_confidence",
    ),
    "confidence_gap": KPIConfig(
        "confidence_gap",
        "Confidence Gap",
        "Difference between the first and second retrieved documents.",
        group="retriever_confidence",
    ),
    "score_variance": KPIConfig(
        "score_variance",
        "Score Variance",
        "Variance across the top-k retrieved scores.",
        group="retriever_confidence",
    ),
    "avg_similarity": KPIConfig(
        "avg_similarity",
        "Average Similarity",
        "Mean similarity score over retrieved candidates.",
        group="document_diversity",
    ),
    "max_similarity": KPIConfig(
        "max_similarity",
        "Max Similarity",
        "Peak similarity score within the retrieved set.",
        group="document_diversity",
    ),
    "diversity_score": KPIConfig(
        "diversity_score",
        "Diversity Score",
        "Share of unique documents within the retrieved context.",
        group="document_diversity",
    ),
    "ngram_overlap": KPIConfig(
        "ngram_overlap",
        "N-gram Overlap",
        "Bigram overlap between generated answers and gold answers.",
        group="context_utilization",
    ),
    "cross_attention": KPIConfig(
        "cross_attention",
        "Question Coverage",
        "Token overlap between the user question and the generated answer.",
        group="context_utilization",
    ),
    "truncation_rate": KPIConfig(
        "truncation_rate",
        "Truncation Rate",
        "Estimated fraction of prompt tokens truncated before generation.",
        group="prompt_truncation",
    ),
    "content_attribution_gap": KPIConfig(
        "content_attribution_gap",
        "Attribution Gap",
        "How much of the answer references retrieved documents (1 - coverage).",
        group="retriever_generator_drift",
    ),
    "attention_based_analysis": KPIConfig(
        "attention_based_analysis",
        "Document Coverage",
        "Share of retrieved titles referenced in the final answer.",
        group="retriever_generator_drift",
    ),
    "retrieval_time": KPIConfig(
        "retrieval_time",
        "Retrieval Latency",
        "Latency attributed to the retrieval stage (ms).",
        group="latency_timeout",
    ),
    "generation_time": KPIConfig(
        "generation_time",
        "Generation Latency",
        "Latency attributed to the generation stage (ms).",
        group="latency_timeout",
    ),
    "total_latency": KPIConfig(
        "total_latency",
        "Total Latency",
        "End-to-end latency in milliseconds.",
        group="latency_timeout",
    ),
    "timeout_events": KPIConfig(
        "timeout_events",
        "Timeout Events",
        "Binary indicator flagging runs above the timeout threshold.",
        group="latency_timeout",
    ),
    "slow_request_ratio": KPIConfig(
        "slow_request_ratio",
        "Slow Request Ratio",
        "Rolling ratio of runs exceeding the slow-request threshold.",
        group="latency_timeout",
    ),
}
