"""Simple evaluation harness that logs HotpotQA runs via RAGWatch."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional

from itertools import cycle

from colorama import Fore, Style, init as colorama_init

from ragwatch import RAGWatchLogger
from ragwatch.settings import env_str

from .data_prep import load_questions
from .build_rag import build_rag_chain
from .retriever import (
    DEFAULT_TOP_K,
    build_document_corpus,
    build_retriever,
    load_vectorstore,
)
colorama_init(autoreset=False)


def _init_console_logger() -> logging.Logger:
    logger = logging.getLogger("ragwatch.hotpotqa")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[RAGWatch] %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


_CONSOLE = _init_console_logger()

_STAGE_COLORS = {
    "prep": Fore.CYAN,
    "vector": Fore.MAGENTA,
    "retriever": Fore.BLUE,
    "rag": Fore.CYAN,
    "stream": Fore.YELLOW,
    "session": Fore.GREEN,
    "retrieval": Fore.LIGHTBLUE_EX,
    "write": Fore.WHITE,
    "error": Fore.RED,
}


def _log(stage: str, message: str, *args, level: int = logging.INFO, exc_info=False):
    color = _STAGE_COLORS.get(stage, "")
    reset = Style.RESET_ALL if color else ""
    prefix = f"[{stage.upper()}] "
    fmt = f"{color}{prefix}{message}{reset}"
    _CONSOLE.log(level, fmt, *args, exc_info=exc_info)


def run_hotpotqa_eval(
    log_dir: Optional[str] = None,
    dataset_name: str = "hotpotqa",
    pipeline_name: str = "v1",
) -> List[Dict[str, Any]]:
    """Run the stub HotpotQA pipeline and log each QA pair once."""

    context = _prepare_context(log_dir, dataset_name, pipeline_name)
    qa_pairs = load_questions()
    run_summaries: List[Dict[str, Any]] = []

    _log("stream", "Starting single-pass HotpotQA evaluation over %d questions.", len(qa_pairs))
    for idx, qa in enumerate(qa_pairs, start=1):
        summary = _execute_single_question(
            qa=qa,
            default_session=f"qa-{idx}",
            ctx=context,
        )
        run_summaries.append(summary)

    return run_summaries


def run_hotpotqa_stream(
    *,
    questions: Optional[Iterable[Dict[str, Any]]] = None,
    interval_seconds: float = 1.0,
    log_dir: Optional[str] = None,
    dataset_name: str = "hotpotqa",
    pipeline_name: str = "v1",
    max_iterations: Optional[int] = None,
):
    """Continuously run the RAG pipeline over questions at a fixed cadence."""

    ctx = _prepare_context(log_dir, dataset_name, pipeline_name)
    source = questions or load_questions()
    materialized = list(source)
    if not materialized:
        return
    _log(
        "stream",
        "Starting HotpotQA stream (interval=%.2fs, max_iterations=%s).",
        interval_seconds,
        max_iterations if max_iterations is not None else "∞",
    )
    iterable: Iterable[Dict[str, Any]] = cycle(materialized)

    iterations = 0
    for idx, qa in enumerate(iterable, start=1):
        _execute_single_question(qa=qa, default_session=f"qa-{idx}", ctx=ctx)
        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            break
        time.sleep(interval_seconds)


def _format_retrieved_docs(results):
    formatted: List[Dict[str, Any]] = []
    for idx, (doc, score) in enumerate(results, start=1):
        metadata = dict(doc.metadata or {})
        doc_id = str(
            metadata.get("doc_id")
            or metadata.get("id")
            or metadata.get("source")
            or f"doc-{idx}"
        )
        source = str(metadata.get("source") or metadata.get("title") or doc_id)
        formatted.append(
            {
                "doc_id": doc_id,
                "score": float(score),
                "source": source,
                "metadata": metadata,
            }
        )
    return formatted


class _EvalContext:
    def __init__(self, logger, vectorstore, retriever, rag):
        self.logger = logger
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.rag = rag


def _prepare_context(log_dir, dataset_name, pipeline_name) -> _EvalContext:
    _log(
        "prep",
        "Preparing evaluation context (dataset=%s, pipeline=%s).",
        dataset_name,
        pipeline_name,
    )
    docs = build_document_corpus()
    _log("prep", "Document corpus ready with %d entries.", len(docs))
    vectorstore = _load_or_create_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": DEFAULT_TOP_K})
    _log("retriever", "Retriever initialised (top_k=%d).", DEFAULT_TOP_K)
    rag = build_rag_chain(retriever=retriever)
    _log("rag", "RAG chain constructed and ready for questions.")

    resolved_log_dir = log_dir or env_str("RAGWATCH_HOTPOTQA_LOG_DIR")
    logger = RAGWatchLogger(
        log_dir=resolved_log_dir,
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
    )
    return _EvalContext(logger=logger, vectorstore=vectorstore, retriever=retriever, rag=rag)


def _execute_single_question(
    *, qa: Dict[str, Any], default_session: str, ctx: _EvalContext
) -> Dict[str, Any]:
    question = qa["question"]
    session_id = qa.get("id", default_session)

    _log(
        "session",
        "Session %s: answering question '%s'",
        session_id,
        _preview(question),
    )
    t0 = time.time()
    try:
        _log("session", "Session %s: invoking RAG chain.", session_id)
        result = ctx.rag.invoke(question)
    except Exception:
        elapsed = (time.time() - t0) * 1000
        _log(
            "error",
            "Session %s: RAG invocation failed after %.0f ms.",
            session_id,
            elapsed,
            level=logging.ERROR,
            exc_info=True,
        )
        raise
    t1 = time.time()
    _log(
        "session",
        "Session %s: model answered in %.0f ms.", session_id, (t1 - t0) * 1000
    )

    answer = getattr(result, "content", None) or str(result)
    latency_ms = {"total": (t1 - t0) * 1000}
    token_usage: Dict[str, int] = {}
    _log(
        "retrieval",
        "Session %s: retrieving top %d supporting documents.",
        session_id,
        DEFAULT_TOP_K,
    )
    retrieved_docs = _format_retrieved_docs(
        ctx.vectorstore.similarity_search_with_score(question, k=DEFAULT_TOP_K)
    )
    if retrieved_docs:
        doc_list = ", ".join(doc["doc_id"] for doc in retrieved_docs[:3])
        if len(retrieved_docs) > 3:
            doc_list += ", …"
        _log(
            "retrieval",
            "Session %s: retrieved documents %s",
            session_id,
            doc_list,
        )
    else:
        _log("retrieval", "Session %s: no supporting documents retrieved.", session_id)

    ctx.logger.log_run(
        question=question,
        answer=answer,
        retrieved_docs=retrieved_docs,
        latency_ms=latency_ms,
        token_usage=token_usage,
        session_id=session_id,
        extra={"gold_answer": qa.get("answer")},
    )
    _log(
        "write",
        "Session %s: run logged to %s",
        session_id,
        ctx.logger.writer.path,
    )

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "latency_ms": latency_ms,
    }


def _preview(question: str, limit: int = 80) -> str:
    return question if len(question) <= limit else f"{question[:limit]}…"


def _load_or_create_vectorstore(docs: List[Dict[str, Any]]):
    try:
        vectorstore = load_vectorstore()
        _log("vector", "Loaded existing FAISS vectorstore from disk.")
        return vectorstore
    except FileNotFoundError:
        _log(
            "vector",
            "No cached FAISS index found; building a new one with %d documents.",
            len(docs),
        )
        build_retriever(docs)
        vectorstore = load_vectorstore()
        _log("vector", "FAISS index built and cached.")
        return vectorstore
