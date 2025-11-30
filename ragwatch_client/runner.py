"""Generic dataset runner that powers the ragwatch_client CLI."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional

from ragwatch import RAGMonitor
from ragwatch.logging import ConsoleStepLogger

from .factory import DatasetClient, DatasetResources

@dataclass(slots=True)
class DatasetRunConfig:
    dataset_name: Optional[str] = None
    version: Optional[str] = None
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

    @classmethod
    def build_from_dataset(
        cls,
        dataset: DatasetClient,
        dataset_name: str,
        version: str,
        log_dir_override: Optional[str],
        log_steps: bool,
    ) -> "PreparedContext":
        _log(
            "prep",
            "Preparing resources for dataset '%s' (version=%s).",
            dataset_name,
            version,
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

        monitor = RAGMonitor(
            dataset_name=dataset_name,
            version=version,
            log_dir=log_dir_override,
            log_env_var=getattr(dataset, "log_env_var", None),
            step_logger=_STEP_LOGGER,
            default_log_steps=log_steps,
        )

        return cls(
            monitor=monitor,
            vectorstore=resources.vectorstore,
            rag_chain=resources.rag_chain,
            top_k=retriever_top_k,
            log_steps=log_steps,
        )


def run_eval(dataset: DatasetClient, config: DatasetRunConfig) -> List[Dict[str, Any]]:
    return _run_dataset(dataset, config, mode="eval")


def run_stream(
    dataset: DatasetClient,
    config: DatasetRunConfig,
    *,
    questions: Optional[Iterable[Dict[str, Any]]] = None,
):
    _run_dataset(dataset, config, mode="stream", questions=questions)


# ----- Internal helpers ------------------------------------------------------------

def _run_dataset(
    dataset: DatasetClient,
    config: DatasetRunConfig,
    *,
    mode: Literal["eval", "stream"],
    questions: Optional[Iterable[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    dataset_name = config.dataset_name or getattr(dataset, "dataset_name", None)
    # TODO: Instead of checking dataset_name here, consider adding a thorough arg validation beforehand
    if dataset_name is None:
        raise ValueError("Dataset is missing 'dataset_name'.")
    prepared = PreparedContext.build_from_dataset(
        dataset,
        dataset_name,
        config.version,
        config.log_dir,
        config.log_steps,
    )

    source = questions if questions is not None else dataset.load_questions()
    qa_pairs = list(source)

    if not qa_pairs:
        if mode == "stream":
            _log("stream", "Dataset %s produced no questions; nothing to stream.", dataset_name)
            return None
        _log("stream", "Dataset %s produced no questions; nothing to evaluate.", dataset_name)
        return []

    if mode == "eval":
        _log(
            "stream",
            "Starting %s evaluation over %d questions.",
            dataset_name,
            len(qa_pairs),
        )
    else:
        _log(
            "stream",
            "Starting %s stream (interval=%.2fs, max_iterations=%s).",
            dataset_name,
            config.interval_seconds,
            config.max_iterations if config.max_iterations is not None else "∞",
        )

    summaries: List[Dict[str, Any]] = []
    iterations = 0

    for idx, qa in enumerate(qa_pairs, start=1):
        summary = _execute_single_question(
            qa=qa,
            default_session=f"{dataset_name}-{idx}",
            monitor=prepared.monitor,
            vectorstore=prepared.vectorstore,
            rag=prepared.rag_chain,
            top_k=prepared.top_k,
            log_steps=prepared.log_steps,
        )
        if mode == "eval":
            summaries.append(summary)
        else:
            iterations += 1
            if config.max_iterations is not None and iterations >= config.max_iterations:
                break
            time.sleep(config.interval_seconds)

    return summaries if mode == "eval" else None


_STEP_LOGGER = ConsoleStepLogger()


def _log(stage: str, message: str, *args, level: int = logging.INFO, exc_info=False) -> None:
    _STEP_LOGGER.log(stage, message, *args, level=level, exc_info=exc_info)


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


