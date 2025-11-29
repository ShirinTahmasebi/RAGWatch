"""Generic dataset runner that powers the ragwatch_client CLI."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Iterable, List, Optional, Tuple

from colorama import Fore, Style, init as colorama_init

from ragwatch import RAGMonitor
from ragwatch.settings import env_str

from .datasets.base import DatasetClient, DatasetResources

colorama_init(autoreset=False)


@dataclass(slots=True)
class DatasetRunConfig:
    dataset_name: Optional[str] = None
    pipeline_name: Optional[str] = None
    log_dir: Optional[str] = None
    interval_seconds: float = 1.0
    max_iterations: Optional[int] = None


@dataclass(slots=True)
class PreparedContext:
    monitor: RAGMonitor
    vectorstore: Any
    rag_chain: Any
    top_k: int


def run_eval(dataset: DatasetClient, config: Optional[DatasetRunConfig] = None) -> List[Dict[str, Any]]:
    config = config or DatasetRunConfig()
    dataset_name, pipeline_name = _resolve_names(dataset, config)
    prepared = _prepare_context(dataset, dataset_name, pipeline_name, config.log_dir)

    qa_pairs = dataset.load_questions()
    summaries: List[Dict[str, Any]] = []

    _log(
        "stream",
        "Starting %s evaluation over %d questions.",
        dataset.slug,
        len(qa_pairs),
    )

    for idx, qa in enumerate(qa_pairs, start=1):
        summary = _execute_single_question(
            qa=qa,
            default_session=f"{dataset.slug}-{idx}",
            monitor=prepared.monitor,
            vectorstore=prepared.vectorstore,
            rag=prepared.rag_chain,
            top_k=prepared.top_k,
        )
        summaries.append(summary)

    return summaries


def run_stream(
    dataset: DatasetClient,
    config: Optional[DatasetRunConfig] = None,
    *,
    questions: Optional[Iterable[Dict[str, Any]]] = None,
):
    config = config or DatasetRunConfig()
    dataset_name, pipeline_name = _resolve_names(dataset, config)
    prepared = _prepare_context(dataset, dataset_name, pipeline_name, config.log_dir)

    source = questions or dataset.load_questions()
    materialized = list(source)
    if not materialized:
        _log("stream", "Dataset %s produced no questions; nothing to stream.", dataset.slug)
        return

    _log(
        "stream",
        "Starting %s stream (interval=%.2fs, max_iterations=%s).",
        dataset.slug,
        config.interval_seconds,
        config.max_iterations if config.max_iterations is not None else "∞",
    )

    iterable: Iterable[Dict[str, Any]] = cycle(materialized)
    iterations = 0

    for idx, qa in enumerate(iterable, start=1):
        _execute_single_question(
            qa=qa,
            default_session=f"{dataset.slug}-{idx}",
            monitor=prepared.monitor,
            vectorstore=prepared.vectorstore,
            rag=prepared.rag_chain,
            top_k=prepared.top_k,
        )
        iterations += 1
        if config.max_iterations is not None and iterations >= config.max_iterations:
            break
        time.sleep(config.interval_seconds)


# ----- Internal helpers ------------------------------------------------------------


def _init_console_logger() -> logging.Logger:
    logger = logging.getLogger("ragwatch.client")
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


def _log(stage: str, message: str, *args, level: int = logging.INFO, exc_info=False) -> None:
    color = _STAGE_COLORS.get(stage, "")
    reset = Style.RESET_ALL if color else ""
    prefix = f"[{stage.upper()}] "
    fmt = f"{color}{prefix}{message}{reset}"
    _CONSOLE.log(level, fmt, *args, exc_info=exc_info)


@dataclass(slots=True)
class _RuntimeArtifacts:
    dataset_name: str
    pipeline_name: str
    context: PreparedContext


def _prepare_context(
    dataset: DatasetClient,
    dataset_name: str,
    pipeline_name: str,
    log_dir_override: Optional[str],
) -> PreparedContext:
    _log(
        "prep",
        "Preparing resources for dataset '%s' (pipeline=%s).",
        dataset.slug,
        pipeline_name,
    )

    resources: DatasetResources = dataset.prepare_resources()
    metadata = resources.metadata or {}

    if metadata.get("doc_count") is not None:
        _log("prep", "Document corpus ready with %d entries.", metadata["doc_count"])
    if metadata.get("vectorstore_source"):
        _log("vector", "FAISS index %s successfully.", metadata["vectorstore_source"])

    retriever_top_k = int(
        metadata.get(
            "top_k",
            getattr(dataset, "default_top_k", 5),
        )
    )
    _log("retriever", "Retriever initialised (top_k=%d).", retriever_top_k)
    _log("rag", "RAG chain constructed and ready for questions.")

    resolved_log_dir = _resolve_log_dir(dataset, log_dir_override)
    monitor = RAGMonitor(
        dataset_name=dataset_name,
        pipeline_name=pipeline_name,
        log_dir=resolved_log_dir,
    )

    return PreparedContext(
        monitor=monitor,
        vectorstore=resources.vectorstore,
        rag_chain=resources.rag_chain,
        top_k=retriever_top_k,
    )


def _resolve_log_dir(dataset: DatasetClient, override: Optional[str]) -> str:
    if override:
        return override

    env_var = getattr(dataset, "log_env_var", None)
    if env_var:
        try:
            return env_str(env_var)
        except RuntimeError:
            pass

    try:
        return env_str("RAGWATCH_LOG_DIR")
    except RuntimeError as exc:
        raise RuntimeError(
            "Log directory not configured. Provide --log-dir or set RAGWATCH_LOG_DIR."
        ) from exc


def _resolve_names(dataset: DatasetClient, config: DatasetRunConfig) -> Tuple[str, str]:
    dataset_name = config.dataset_name or getattr(dataset, "default_dataset_name", dataset.slug)
    pipeline_name = config.pipeline_name or getattr(dataset, "default_pipeline_name", "default")
    return dataset_name, pipeline_name


def _execute_single_question(
    *,
    qa: Dict[str, Any],
    default_session: str,
    monitor: RAGMonitor,
    vectorstore: Any,
    rag: Any,
    top_k: int,
) -> Dict[str, Any]:
    question = qa["question"]
    session_id = qa.get("id", default_session)

    _log(
        "session",
        "Session %s: answering question '%s'",
        session_id,
        _preview(question),
    )

    with monitor.session(question=question, session_id=session_id) as session:
        t0 = time.time()
        try:
            _log("session", "Session %s: invoking RAG chain.", session_id)
            result = rag.invoke(question)
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

        latency_ms = {"total": (time.time() - t0) * 1000}
        _log("session", "Session %s: model answered in %.0f ms.", session_id, latency_ms["total"])

        answer = getattr(result, "content", None) or str(result)
        token_usage: Dict[str, int] = {}
        _log(
            "retrieval",
            "Session %s: retrieving top %d supporting documents.",
            session_id,
            top_k,
        )
        retrieved = vectorstore.similarity_search_with_score(question, k=top_k)
        session.record_retrieval(retrieved)
        if retrieved:
            doc_list = ", ".join(
                _doc_id_from_result(pair, idx)
                for idx, pair in enumerate(retrieved[:3], start=1)
            )
            if len(retrieved) > 3:
                doc_list += ", …"
            _log("retrieval", "Session %s: retrieved %s", session_id, doc_list)
        else:
            _log("retrieval", "Session %s: no supporting documents retrieved.", session_id)

        if qa.get("answer"):
            session.set_gold_answer(qa["answer"])
        session.record_answer(answer, latency_ms=latency_ms, token_usage=token_usage)
        _log(
            "write",
            "Session %s: run logged to %s",
            session_id,
            monitor.logger.writer.path,
        )

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "latency_ms": latency_ms,
    }


def _doc_id_from_result(result_pair, fallback_idx: int) -> str:
    doc, _score = _split_result(result_pair)
    metadata = getattr(doc, "metadata", None) or {}
    for key in ("doc_id", "id", "source", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    return f"doc-{fallback_idx}"


def _split_result(result_pair) -> Tuple[Any, Any]:
    if isinstance(result_pair, (list, tuple)) and len(result_pair) == 2:
        return result_pair
    return result_pair, None


def _preview(question: str, limit: int = 80) -> str:
    return question if len(question) <= limit else f"{question[:limit]}…"
