"""Streamlit dashboard constants that remain UI-specific."""
from __future__ import annotations


class Paths:
    CSV_MONITOR = "data/monitors.csv"
    CSV_KPIS = "logs/kpis/"
    CSV_ALERTS = "logs/alerts/"


class Fields:
    TIMESTAMP = "timestamp"
    MONITOR_ID = "monitor_id"
    ALERT_ID = "alert_id"
    NAME = "name"
    IP = "ip"
    PORT = "port"
    INTERNAL = "interval"
    MONITORING_MODULES = "monitoring_modules"
    ALERT_TYPE = "alert_type"
    MESSAGE = "message"
    START_TIME = "start_time"
    END_TIME = "end_time"
    ALERT_TIME = "alert_time"
    KPI = "kpi"
    RUN_ID = "run_id"
    EVENT = "event"
    LATENCY_MS = "latency_ms"
    LATENCY_RETRIEVE = "latency_retrieve"
    LATENCY_GENERATE = "latency_generate"
    LATENCY_REWRITE = "latency_rewrite"
    LATENCY_E2E = "latency_e2e"
    QUERY_ID = "query_id"
    USER_QUERY = "user_query"
    USECASE = "usecase"
    RAG_STRUCTURE = "rag_structure"
    GENERATOR = "generator"
    RETRIEVER = "retriever"
    RETRIEVED_DOCS_INFO = "retrieved_docs_info"
    RETRIEVED_DOC_ID = "retrieved_doc_id"
    RETRIEVED_DOC_RANK = "retrieved_doc_rank"
    RETRIEVED_DOC_SCORE = "retrieved_doc_score"
    REWRITTEN_QUERIES = "rewritten_queries"
    PER_VARIANT_COUNT = "per_variant_counts"
    ANSWER = "answer"
    TEXT = "text"
    CITATIONS = "citations"
    TOKENS = "tokens"


class States:
    MONITOR_LOADED = "is_monitor_loaded"
    MONITOR_NAMES = "monitor_names"
    MONITOR_DICT = "monitor_dict"
    CURRENT_PAGE = "current_page"
    SELECT_MONITOR_ID = "selected_monitor_id"


class Pages:
    home = "home"
    monitor_create = "monitor_create"
    monitor_details = "monitor_details"

