"""Generic dataset runner that powers the ragwatch_client CLI."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ragwatch import RAGMonitor
from ragwatch.logging import ConsoleStepLogger
from ragwatch.utils import env_str

from .datasets.base import DatasetClient, DatasetResources

@dataclass(slots=True)
class DatasetRunConfig:
    dataset_name: Optional[str] = None
    pipeline_name: Optional[str] = None
    log_dir: Optional[str] = None
    interval_seconds: float = 1.0
    max_iterations: Optional[int] = None
    log_steps: bool = True


@dataclass(slots=True)
class PreparedContext:
    monitor: RAGMonitor
    vectorstore: Any
    rag_chain: Any
    top_k: int
    log_steps: bool


def run_eval(dataset: DatasetClient, config: Optional[DatasetRunConfig] = None) -> List[Dict[str, Any]]:
    config = config or DatasetRunConfig()
    dataset_name, pipeline_name = _resolve_names(dataset, config)
    prepared = _prepare_context(dataset, dataset_name, pipeline_name, config.log_dir, config.log_steps)

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
            log_steps=prepared.log_steps,
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
    prepared = _prepare_context(
        dataset,
        dataset_name,
        pipeline_name,
        config.log_dir,
        config.log_steps,
    )

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

    iterations = 0

    for idx, qa in enumerate(materialized, start=1):
        _execute_single_question(
            qa=qa,
            default_session=f"{dataset.slug}-{idx}",
            monitor=prepared.monitor,
            vectorstore=prepared.vectorstore,
            rag=prepared.rag_chain,
            top_k=prepared.top_k,
            log_steps=prepared.log_steps,
        )
        iterations += 1
        if config.max_iterations is not None and iterations >= config.max_iterations:
            break
        time.sleep(config.interval_seconds)


# ----- Internal helpers ------------------------------------------------------------


_STEP_LOGGER = ConsoleStepLogger()


def _log(stage: str, message: str, *args, level: int = logging.INFO, exc_info=False) -> None:
    _STEP_LOGGER.log(stage, message, *args, level=level, exc_info=exc_info)


def _prepare_context(
    dataset: DatasetClient,
    dataset_name: str,
    pipeline_name: str,
    log_dir_override: Optional[str],
 	log_steps: bool,
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
        step_logger=_STEP_LOGGER,
        default_log_steps=log_steps,
    )

    return PreparedContext(
        monitor=monitor,
        vectorstore=resources.vectorstore,
        rag_chain=resources.rag_chain,
        top_k=retriever_top_k,
        log_steps=log_steps,
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
    log_steps: bool,
) -> Dict[str, Any]:
    question = qa["question"]
    session_id = qa.get("id", default_session)

    with monitor.session(question=question, session_id=session_id, log_steps=log_steps) as session:
        t0 = time.time()
        try:
            result = rag.invoke(question)
        except Exception:
            raise

        latency_ms = {"total": (time.time() - t0) * 1000}

        answer = getattr(result, "content", None) or str(result)
        token_usage: Dict[str, int] = {}
        retrieved = vectorstore.similarity_search_with_score(question, k=top_k)
        session.record_retrieval(retrieved, top_k=top_k)

        if qa.get("answer"):
            session.set_gold_answer(qa["answer"])
        session.record_answer(answer, latency_ms=latency_ms, token_usage=token_usage)

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "latency_ms": latency_ms,
    }


