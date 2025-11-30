"""Session-based monitoring helpers for RAGWatch."""
from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager
from typing import Any, Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple, Union

from ..logging import ConsoleStepLogger, RAGWatchLogger
from ..utils import resolve_log_dir

RetrievedDocLike = Dict[str, Any]
RawDocument = Any
RawDocWithScore = Union[Sequence[Any], Tuple[Any, Any]]


class StepLoggerLike(Protocol):
    def log(
        self,
        stage: str,
        message: str,
        *args: Any,
        level: int = ...,
        exc_info: Any | None = ...,
    ) -> None:
        ...


class RAGMonitor:
    """High-level helper that records RAG sessions with minimal boilerplate."""

    def __init__(
        self,
        *,
        dataset_name: str,
        pipeline_name: str,
        log_dir: Optional[str] = None,
        log_env_var: Optional[str] = None,
        logger: Optional[RAGWatchLogger] = None,
        step_logger: Optional[StepLoggerLike] = None,
        default_log_steps: bool = False,
    ) -> None:
        resolved_log_dir = resolve_log_dir(
            override=log_dir,
            dataset_env_var=log_env_var,
        )

        self.logger = logger or RAGWatchLogger(
            log_dir=resolved_log_dir,
            dataset_name=dataset_name,
            pipeline_name=pipeline_name,
        )
        self._step_logger = step_logger
        self._default_log_steps = default_log_steps

    def session(
        self,
        *,
        question: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        log_steps: Optional[bool] = None,
    ) -> "SessionContext":
        should_log_steps = self._default_log_steps if log_steps is None else log_steps
        step_logger = self._resolve_step_logger() if should_log_steps else None
        return SessionContext(
            monitor=self,
            question=question,
            session_id=session_id or self.logger.new_run_id(),
            metadata=metadata or {},
            log_steps=should_log_steps,
            step_logger=step_logger,
        )

    def _resolve_step_logger(self) -> StepLoggerLike:
        if self._step_logger is None:
            self._step_logger = ConsoleStepLogger()
        return self._step_logger


class SessionContext(AbstractContextManager["SessionContext"]):
    """Context manager that buffers RAG activity until the session completes."""

    def __init__(
        self,
        *,
        monitor: RAGMonitor,
        question: str,
        session_id: str,
        metadata: Dict[str, Any],
        log_steps: bool,
        step_logger: Optional[StepLoggerLike],
    ) -> None:
        self.monitor = monitor
        self.question = question
        self.session_id = session_id
        self._metadata = metadata
        self._log_steps = log_steps
        self._step_logger = step_logger
        self._start_time = time.time()
        self._retrieved_docs: List[RetrievedDocLike] = []
        self._answer: Optional[str] = None
        self._latency_ms: Dict[str, float] = {}
        self._token_usage: Dict[str, int] = {}
        self._extra: Dict[str, Any] = {}
        self._finished = False

    def __enter__(self) -> "SessionContext":
        self._log_step(
            "session",
            "Session %s: answering question '%s'",
            self.session_id,
            _preview(self.question),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            self.add_extra(error=str(exc))
            self._log_step(
                "error",
                "Session %s: error after %.0f ms: %s",
                self.session_id,
                (time.time() - self._start_time) * 1000,
                str(exc),
                level=logging.ERROR,
                exc_info=True,
            )
        self.finish()
        # Propagate exceptions to caller
        return False

    # ----- Public recording helpers -------------------------------------------------
    def record_retrieval(
        self,
        docs: Iterable[Union[RawDocument, RawDocWithScore]],
        *,
        top_k: Optional[int] = None,
        scores: Optional[Iterable[float]] = None,
    ) -> None:
        normalized = _normalize_retrieved_docs(docs, scores)
        self._retrieved_docs = normalized
        if not normalized:
            self._log_step(
                "retrieval",
                "Session %s: no supporting documents recorded%s.",
                self.session_id,
                f" from top-{top_k}" if top_k is not None else "",
            )
            return
        preview = ", ".join(doc["doc_id"] for doc in normalized[:3])
        if len(normalized) > 3:
            preview += ", …"
        range_desc = f" from top-{top_k}" if top_k is not None else ""
        self._log_step(
            "retrieval",
            "Session %s: recorded %d docs%s (%s)",
            self.session_id,
            len(normalized),
            range_desc,
            preview,
        )

    def record_answer(
        self,
        answer: str,
        *,
        latency_ms: Optional[Dict[str, float]] = None,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        self._answer = answer
        if latency_ms:
            self._latency_ms.update(latency_ms)
        if token_usage:
            self._token_usage.update(token_usage)
        total = (latency_ms or {}).get("total")
        if total is not None:
            self._log_step(
                "answer",
                "Session %s: answer recorded in %.0f ms.",
                self.session_id,
                total,
            )
        else:
            self._log_step("answer", "Session %s: answer recorded.", self.session_id)

    def add_extra(self, **fields: Any) -> None:
        self._extra.update(fields)

    def set_gold_answer(self, value: str) -> None:
        self.add_extra(gold_answer=value)

    def add_latency(self, label: str, value_ms: float) -> None:
        self._latency_ms[label] = value_ms

    def add_token_usage(self, label: str, value: int) -> None:
        self._token_usage[label] = value

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True

        latency_ms = dict(self._latency_ms)
        if "total" not in latency_ms:
            latency_ms["total"] = (time.time() - self._start_time) * 1000

        answer = self._answer or ""
        extras = {**self._metadata, **self._extra}
        retrieved_docs = self._retrieved_docs

        self.monitor.logger.log_run(
            question=self.question,
            answer=answer,
            retrieved_docs=retrieved_docs,
            latency_ms=latency_ms,
            token_usage=self._token_usage,
            session_id=self.session_id,
            extra=extras,
        )
        self._log_step(
            "write",
            "Session %s: run logged to %s",
            self.session_id,
            self.monitor.logger.writer.path,
        )

    def log_step(
        self,
        stage: str,
        message: str,
        *args: Any,
        level: int = logging.INFO,
        exc_info: Any | None = None,
    ) -> None:
        self._log_step(stage, message, *args, level=level, exc_info=exc_info)

    def _log_step(
        self,
        stage: str,
        message: str,
        *args: Any,
        level: int = logging.INFO,
        exc_info: Any | None = None,
    ) -> None:
        if not self._log_steps or self._step_logger is None:
            return
        self._step_logger.log(stage, message, *args, level=level, exc_info=exc_info)


# ----- Normalization helpers --------------------------------------------------------

def _normalize_retrieved_docs(
    docs: Iterable[Union[RawDocument, RawDocWithScore]],
    scores: Optional[Iterable[float]] = None,
) -> List[RetrievedDocLike]:
    normalized: List[RetrievedDocLike] = []
    score_iter: Optional[Iterator[float]] = iter(scores) if scores is not None else None

    for idx, entry in enumerate(docs, start=1):
        doc, explicit_score = _split_doc_and_score(entry)
        score = explicit_score
        if score is None and score_iter is not None:
            score = next(score_iter, None)

        if _looks_like_retrieved_doc(doc):
            normalized.append(
                {
                    "doc_id": str(doc.get("doc_id")),
                    "score": float(score if score is not None else doc.get("score", 0.0)),
                    "source": doc.get("source") or str(doc.get("doc_id")),
                    "metadata": doc.get("metadata", {}),
                }
            )
            continue

        metadata = _extract_metadata(doc)
        doc_id = _infer_doc_id(metadata, idx)
        source = metadata.get("source") or metadata.get("title") or doc_id
        resolved_score = score
        if resolved_score is None:
            resolved_score = metadata.get("score")
        normalized.append(
            {
                "doc_id": doc_id,
                "score": float(resolved_score if resolved_score is not None else 0.0),
                "source": str(source),
                "metadata": metadata,
            }
        )

    return normalized


def _split_doc_and_score(entry: Union[RawDocument, RawDocWithScore]) -> Tuple[Any, Optional[float]]:
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        doc, score = entry
        try:
            return doc, None if score is None else float(score)
        except (TypeError, ValueError):
            return doc, None
    return entry, None


def _looks_like_retrieved_doc(doc: Any) -> bool:
    return isinstance(doc, dict) and {"doc_id", "metadata"}.issubset(doc.keys())


def _extract_metadata(doc: Any) -> Dict[str, Any]:
    if hasattr(doc, "metadata"):
        data = getattr(doc, "metadata") or {}
        return dict(data)
    if isinstance(doc, dict):
        return {k: v for k, v in doc.items() if k not in {"text", "page_content"}}
    return {}


def _infer_doc_id(metadata: Dict[str, Any], idx: int) -> str:
    for key in ("doc_id", "id", "source", "title"):
        value = metadata.get(key)
        if value:
            return str(value)
    return f"doc-{idx}"


def _preview(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}…"
