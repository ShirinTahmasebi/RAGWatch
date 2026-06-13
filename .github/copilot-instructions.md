# RAGWatch Project Context

## Project goal

RAGWatch is a research prototype for runtime monitoring and diagnosis of Retrieval-Augmented Generation (RAG) systems.

The long-term goal is to support:

* RAG execution over real QA datasets
* multiple retriever/vector DB backends
* OpenTelemetry tracing
* deterministic and model-based KPI calculation
* DB-based monitoring signals
* experiment execution and result export
* later dashboards/Grafana
* later semantic metrics such as relevance, faithfulness, and drift

This is a research-oriented codebase, so clean architecture, reproducibility, and readable experiments matter more than production complexity.

---

## Current completed components

The project already includes the following completed pieces.

### 1. Core schemas and interfaces

Core objects exist for:

* `Document`
* `QAExample`
* `RetrievedDocument`
* `GenerationResult`
* `RAGRun`
* `RAGDataset`

There are abstract interfaces for:

* dataset loading
* retrieval
* generation
* RAG clients

Do not redesign these unless absolutely necessary.

---

### 2. Dataset support

The project supports:

* toy JSONL datasets
* SQuAD through Hugging Face using `rajpurkar/squad`
* HotpotQA through Hugging Face using `hotpotqa/hotpot_qa`

Dataset adapters return a unified `RAGDataset` object with:

* `corpus`
* `qa_examples`
* metadata

---

### 3. Retriever and vector DB support

The project supports:

* TF-IDF retriever
* Chroma retriever
* Qdrant retriever
* persistent PostgreSQL + pgvector retriever

Chroma and Qdrant can run locally without credentials.

pgvector uses a local PostgreSQL database through Docker Compose.

---

### 4. Persistent pgvector/Postgres backend

The repo includes:

* `docker-compose.yml`
* `infra/postgres/init.sql`
* `.env.template`
* gitignored `.env`
* environment helpers

The local database URL is:

```bash
RAGWATCH_PGVECTOR_URL=postgresql://ragwatch:ragwatch@localhost:5433/ragwatch
```

The host port is `5433`, mapped to container port `5432`.

The database currently has tables:

* `ragwatch_documents`
* `ragwatch_embeddings`

These tables support DB/index-level statistics.

Important: `.env` must not be committed. `.env.template` should be committed.

---

### 5. OpenTelemetry tracing

The project has OpenTelemetry instrumentation.

Current tracing creates spans such as:

```text
ragwatch.rag_run
  ├── ragwatch.retrieval
  └── ragwatch.generation
```

Tracing currently supports console output and optional OTLP configuration.

Important: OpenTelemetry is currently a tracing foundation only. The project does not yet read KPIs from an OpenTelemetry backend, and there is no Grafana dashboard yet.

---

### 6. KPI calculation

The project has a KPI system with:

* central KPI catalog
* RAGRun-based KPI engine
* DB-based KPI engine

Important files:

* `src/ragwatch/metrics/catalog.py`
* `src/ragwatch/metrics/base.py`
* `src/ragwatch/metrics/retrieval.py`
* `src/ragwatch/metrics/generation.py`
* `src/ragwatch/metrics/runtime.py`
* `src/ragwatch/metrics/db_stats.py`
* `src/ragwatch/metrics/report.py`

The KPI catalog defines controlled vocabularies for:

* KPI IDs
* categories
* stages
* sources
* descriptions

Metric classes should not hardcode KPI metadata. They should reference the central catalog.

---

## Current KPI types

### RAGRun-based KPIs

These are calculated from a single `RAGRun`.

They include retrieval metrics such as:

* number of retrieved documents
* retrieval score min/max/mean/std/range
* top-1/top-2 score gap
* context length
* unique retrieved sources
* lexical redundancy

They include generation metrics such as:

* answer length in characters
* answer length in words
* answer/context length ratio

They include runtime metrics such as:

* total latency
* retrieval latency
* generation latency

---

### DB-based KPIs

These are calculated from Postgres/pgvector tables.

They include:

* document count
* embedding count
* collection count
* average document length
* average embedding dimension
* collections
* documents per source

---

## Current important examples

Existing examples include:

```bash
python examples/run_toy_rag.py
python examples/run_toy_rag_with_tracing.py
python examples/run_toy_rag_with_kpis.py
python examples/list_available_kpis.py

python examples/run_squad_tfidf.py
python examples/run_squad_chroma.py
python examples/run_squad_qdrant.py

docker compose up -d postgres
python examples/run_squad_pgvector.py
python examples/query_pgvector_stats.py
python examples/query_pgvector_kpis.py
docker compose down
```

---

## Current test status

Recent validation showed:

* without Postgres: many tests pass, DB tests skipped
* with Postgres: all tests pass

Exact test counts may change because KPI catalog tests are parametrized over KPI IDs.

---

## Important design principles

Follow these principles:

1. Do not add `__all__` to `__init__.py` files.
2. Keep `__init__.py` files minimal.
3. Do not hardcode KPI metadata inside metric classes.
4. Use the KPI catalog for KPI names, categories, stages, sources, and descriptions.
5. Do not commit `.env`.
6. Keep `.env.template` safe to commit.
7. Do not require API keys for the current local research prototype.
8. Do not add Grafana yet.
9. Do not add MCP.
10. Do not add LLM-based judges yet.
11. Do not read KPIs from OpenTelemetry storage yet.
12. Preserve existing public interfaces unless absolutely necessary.
13. Keep tests deterministic.
14. Keep examples small and runnable.

---

## Current development stage

The next step is:

# Batch experiment runner + result export

The goal is to run a RAG client over many QA examples, compute KPIs for each run, and save results to files.

The next step should save:

* RAG runs
* retrieved document summaries
* KPI reports
* optional DB KPI snapshot

The first version should export to:

* JSONL
* CSV

Do not add Grafana yet.
Do not add semantic LLM judges yet.
Do not add drift detection yet.
Do not add Postgres experiment storage yet unless explicitly requested later.

---

## Planned next steps after batch runner

After the batch experiment runner, planned future steps are:

1. Postgres storage for experiment runs and KPI values
2. comparison scripts for TF-IDF vs Chroma vs Qdrant vs pgvector
3. paper-quality plots
4. semantic/open-source model-based KPIs
5. drift simulation and drift metrics
6. dashboard or Grafana integration
7. optional OpenTelemetry backend integration



## Coding and architecture rules

Follow these rules throughout the RAGWatch codebase.

### 1. Keep package initialization minimal

* Keep all `__init__.py` files minimal.
* Do not add `__all__`.
* Do not use package-level export magic.
* Prefer explicit imports from concrete modules.

Good:

```python
from ragwatch.metrics.report import KPIEngine
```

Avoid:

```python
from ragwatch.metrics import *
```

---

### 2. Keep the system structured and decoupled

Each module should have one clear responsibility.

Examples:

* dataset adapters should only load and normalize datasets
* retrievers should only index and retrieve documents
* generators should only generate answers
* instrumentation should only trace/log runtime behavior
* metrics should only calculate KPI values
* experiment runners should orchestrate repeated runs
* exporters should only save outputs

Avoid mixing responsibilities. For example, the retriever should not compute KPI reports, and the KPI engine should not know how to load datasets.

---

### 3. Avoid overengineering

Keep the code extensible, but do not introduce unnecessary abstractions.

Avoid adding:

* plugin systems
* dependency injection frameworks
* complex registries unless needed
* async/multiprocessing unless explicitly requested
* Alembic migrations unless explicitly requested
* dashboard infrastructure unless explicitly requested

Prefer simple, readable classes and functions.

---

### 4. Centralize important constants and metadata

Do not define important names, categories, stages, sources, or environment variable keys on the fly inside random files.

Use central files for shared constants.

Examples:

* KPI names, categories, stages, sources, and descriptions belong in `src/ragwatch/metrics/catalog.py`
* environment variable names should belong in config-related utilities
* span names and OpenTelemetry attribute names should belong in instrumentation schema files
* database table names should be centralized in storage or database utility modules if reused

Avoid repeating strings like:

```python
"retrieval_quality"
"generation"
"RAGWATCH_PGVECTOR_URL"
"ragwatch_documents"
```

across many files.

If a string is part of the public architecture, define it once and reuse it.

---

### 5. Preserve existing public interfaces

Do not rewrite existing schemas or interfaces unless there is a strong reason.

Important existing objects include:

* `Document`
* `QAExample`
* `RetrievedDocument`
* `GenerationResult`
* `RAGRun`
* `RAGDataset`
* `BaseRetriever`
* `BaseGenerator`
* `BaseRAGClient`
* `KPIResult`
* `KPIReport`

If a change to these objects is necessary, make it backward-compatible when possible and update tests/examples.

---

### 6. Prefer explicit, typed dataclasses for structured data

Use dataclasses for structured research objects such as:

* experiment configs
* experiment results
* KPI reports
* export records

Use type hints everywhere.

Avoid passing around unstructured dictionaries unless the data is genuinely flexible metadata.

---

### 7. Keep examples small and runnable

Examples should be easy to run and should not require large downloads by default.

Use small values such as:

```python
max_examples = 20
```

or smaller when appropriate.

Examples should fail gracefully when optional infrastructure is missing.

For example, pgvector examples should explain how to run:

```bash
cp .env.template .env
docker compose up -d postgres
```

---

### 8. Keep tests deterministic

Tests should not require external APIs or API keys.

Default tests should not require:

* Hugging Face downloads
* running Postgres
* Chroma server
* Qdrant server
* OpenTelemetry collector
* OpenAI/Claude/Gemini keys

Integration tests that require Postgres should be skipped unless `RAGWATCH_PGVECTOR_URL` is available.

---

### 9. Do not commit local secrets or runtime files

* `.env` must remain gitignored.
* `.env.template` should be committed.
* Do not print secrets in logs.
* Do not hardcode real credentials in Python files.

Local development credentials for Docker are acceptable in documentation and `.env.template`.

---

### 10. Keep research reproducibility in mind

When adding experiment-related code, make sure outputs include enough metadata to reproduce results.

Useful metadata includes:

* experiment name
* dataset name
* retriever name
* generator name
* vector DB backend
* collection name
* embedding model
* top-k
* max examples
* timestamp
* KPI names and values

---

### 11. Do not add future components too early

Unless explicitly requested, do not add:

* Grafana
* MCP
* LLM-as-judge metrics
* open-source semantic judges
* drift detection
* OpenTelemetry backend reading
* Postgres experiment storage
* dashboard code

These are planned future steps, but they should not be implemented prematurely.

---

### 12. Before making changes, inspect the existing code

Before adding or modifying code:

1. Read the relevant existing files.
2. Reuse existing schemas and helpers.
3. Check current tests and examples.
4. Avoid duplicating functionality that already exists.

If something already exists, extend it instead of recreating it.
