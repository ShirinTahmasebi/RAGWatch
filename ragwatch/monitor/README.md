# RAG Monitor & Session Context

This module wraps the low-level `ragwatch.logging` APIs with a single `with monitor.session(...)` block, so RAG pipelines can record every run (question, retrieved docs, answer, latency, token usage, extras) without manual bookkeeping.

## Why `SessionContext`?

- **Explicit lifecycle**: `SessionContext` subclasses `contextlib.AbstractContextManager`, so it provides the standard `__enter__/__exit__` contract. When the `with` block exits—whether the run succeeded or raised—the context manager automatically calls `finish()`.
- **Deferred logging**: All helper calls (`record_retrieval`, `record_answer`, `add_extra`, etc.) only mutate in-memory buffers. Nothing hits disk until `finish()` is invoked inside `__exit__`, which in turn calls `RAGWatchLogger.log_run(...)` to append a JSONL entry.
- **Automatic cleanup**: If an exception bubbles out of the `with` block, `__exit__` captures the error message via `add_extra(error=...)`, emits a step log (when enabled), and still flushes the run so failures are observable.

## Flow inside the `with` block

```
with monitor.session(question=qa["question"], session_id=qa.get("id")) as session:
    session.record_retrieval(retrieved_docs, top_k=5)
    session.set_gold_answer(qa.get("answer", ""))
    session.record_answer(
        llm_answer,
        latency_ms={"total": 750.0},
        token_usage={"prompt": 200, "completion": 95},
    )
# <- exiting the block triggers SessionContext.__exit__ -> finish() -> RAGWatchLogger.log_run
```

Key points:
- `record_retrieval` normalizes any mix of `(doc, score)` tuples, LangChain docs, raw dicts, etc., into `RetrievedDoc` payloads.
- `record_answer` captures latency/token metrics and stores the final answer string.
- You can stash arbitrary metadata via `add_extra(...)` or convenience helpers like `set_gold_answer`.

## `finish()` responsibilities

`finish()` is intentionally small and deterministic:
1. Guard against double-invocation with `_finished`.
2. Ensure a `total` latency exists (derive it from wall-clock if not provided).
3. Merge caller-provided metadata (`metadata` argument) with accumulated extras.
4. Call `RAGWatchLogger.log_run(...)`, which serializes a `RAGRunRecord` and appends it to the dataset/version JSONL file.
5. Emit a `write` step log (when console logging is enabled) so operators know where the record lives.

Because `finish()` lives inside `__exit__`, every path through the `with` block eventually persists a run—no orphaned sessions.

## Step logging vs. JSONL logging

`RAGMonitor.session(..., log_steps=True)` enables optional console narration powered by `ConsoleStepLogger`. These messages (session start, retrieval summary, answer timing, flush path, and error details) are strictly for operator visibility; they do *not* affect the JSONL payload. This separation keeps structured telemetry (via `RAGRunRecord`) independent from human-friendly progress output.

## When to reach for the monitor

Use `RAGMonitor` any time you want to wrap an existing RAG pipeline without refactoring it:
- Evaluate a dataset: the runner loops over questions and lets the monitor capture each run.
- Stream production traffic: open a session per user query, feed in retrieved docs + answer, and let `finish()` ensure the run is persisted even if the LLM call fails.
- Toggle console narration on/off with the `default_log_steps` constructor flag or per session via `log_steps`.

For bespoke instrumentation (custom storage, alternative schemas), you can still instantiate `ragwatch.logging.RAGWatchLogger` directly, but the monitor is the recommended entry point when you want a batteries-included workflow.
