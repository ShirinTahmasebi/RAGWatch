# RAGWatch

**Runtime monitoring and diagnosis framework for Retrieval-Augmented Generation systems.**

RAGWatch is a research prototype for observing, measuring, and diagnosing RAG pipelines. It provides a clean, extensible foundation with multiple retriever backends, dataset adapters, and a composable pipeline architecture.

## What's included (v0.6)

- **Core data models** — `Document`, `QAExample`, `RetrievedDocument`, `GenerationResult`, `RAGRun`, `RAGDataset`
- **Abstract interfaces** — `BaseDatasetLoader`, `BaseRetriever`, `BaseGenerator`, `BaseRAGClient`, `BaseQAAdapter`, `BaseEmbeddingModel`
- **Dataset adapters** — SQuAD, HotpotQA (via Hugging Face `datasets`)
- **JSONL dataset loader** — loads corpus and QA datasets from local JSONL files
- **Retrievers**:
  - TF-IDF (scikit-learn, no extra dependencies)
  - ChromaDB (in-memory vector store)
  - Qdrant (in-memory or remote)
  - pgvector (PostgreSQL)
- **Embedding models**:
  - SentenceTransformer (`all-MiniLM-L6-v2`)
  - Deterministic (hash-based, for testing)
- **Heuristic generator** — placeholder generator (no LLM)
- **Basic RAG client** — orchestrates retrieval + generation with timing
- **OpenTelemetry instrumentation** — structured tracing of RAG runs, retrieval, and generation
- **Deterministic KPI engine** — retrieval quality, generation quality, and runtime metrics from `RAGRun`
- **Batch experiment runner** — run pipelines over multiple QA examples with KPI export to JSONL/CSV
- **Examples** — toy pipeline, SQuAD with TF-IDF/Chroma/Qdrant, tracing demos, KPI reports, batch experiments

## Installation

### Base (TF-IDF only)

```bash
pip install -e ".[dev]"
```

### With Hugging Face dataset support

```bash
pip install -e ".[dev,datasets]"
```

### With vector database support

```bash
pip install -e ".[dev,datasets,vectordb]"
```

### With OpenTelemetry tracing

```bash
pip install -e ".[dev,otel]"
```

### Everything

```bash
pip install -e ".[dev,datasets,vectordb,otel]"
```

## Running examples

### Toy example (no extra dependencies)

```bash
python examples/run_toy_rag.py
```

### SQuAD with TF-IDF

```bash
pip install -e ".[datasets]"
python examples/run_squad_tfidf.py
```

### SQuAD with ChromaDB

```bash
pip install -e ".[datasets,vectordb]"
python examples/run_squad_chroma.py
```

### SQuAD with Qdrant

```bash
pip install -e ".[datasets,vectordb]"
python examples/run_squad_qdrant.py
```

### Toy example with OpenTelemetry tracing

```bash
pip install -e ".[otel]"
python examples/run_toy_rag_with_tracing.py
```

### SQuAD + Chroma with tracing

```bash
pip install -e ".[datasets,vectordb,otel]"
python examples/run_squad_chroma_with_tracing.py
```

### SQuAD with pgvector (requires Docker)

```bash
pip install -e ".[datasets,vectordb]"
cp .env.template .env
docker compose up -d postgres
python examples/run_squad_pgvector.py
python examples/query_pgvector_stats.py
```

## Running tests

```bash
pytest
```

Tests for optional backends (Chroma, Qdrant) are automatically skipped if the dependencies are not installed. pgvector tests require `RAGWATCH_PGVECTOR_URL` to be set.

## Project structure

```
src/ragwatch/
  core/              # Data models (schema.py) and abstract interfaces (interfaces.py)
  datasets/          # Dataset loaders and adapters
    adapters/        # SQuAD, HotpotQA adapters (Hugging Face)
    jsonl_loader.py  # Local JSONL file loader
    base.py          # BaseQAAdapter interface
    hf_qa_loader.py  # Hugging Face loading utility
  embeddings/        # Embedding model implementations
    base.py          # BaseEmbeddingModel interface
    deterministic.py # Hash-based test embeddings
    sentence_transformer.py
  retrievers/        # Retriever implementations
    tfidf_retriever.py
    vector/          # Vector DB backends
      base.py        # BaseVectorRetriever
      chroma_retriever.py
      qdrant_retriever.py
      pgvector_retriever.py
  generators/        # Generator implementations
  clients/           # RAG client orchestrators
  instrumentation/   # OpenTelemetry tracing
    tracer.py        # RAGWatchTracer setup
    instrumented_client.py  # Traced RAG client
    span_schema.py   # Span/attribute constants
  metrics/           # Deterministic KPI calculation
    catalog.py       # Central KPI registry (all metadata in one place)
    schema.py        # KPIResult, KPIReport data models
    base.py          # BaseMetric interface
    retrieval.py     # Retrieval quality metrics
    generation.py    # Generation quality metrics
    runtime.py       # Latency metrics
    report.py        # KPIEngine and serialization helpers
  experiments/       # Batch experiment runner and export
    schema.py        # ExperimentConfig, ExperimentResult
    runner.py        # ExperimentRunner
    exporters.py     # JSONL and CSV export
  config/            # Configuration utilities
    env.py           # Environment variable loading (.env support)
  storage/           # Database storage helpers
    postgres.py      # PostgreSQL connection and table init
  utils/             # Shared utilities (timing)

examples/            # Runnable example scripts and data
tests/               # pytest tests
```

## OpenTelemetry instrumentation

RAGWatch includes optional OpenTelemetry tracing that instruments the RAG pipeline with structured spans for retrieval, generation, and overall run execution.

### Why tracing?

Tracing lets you observe exactly what happens inside a RAG pipeline: how long retrieval takes, which documents are returned, and how generation behaves — all without manually adding logging code.

### Usage

```python
from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.instrumentation.tracer import RAGWatchTracer

tracer = RAGWatchTracer(service_name="my-rag-app", exporter_type="console")
client = BasicRAGClient(retriever=retriever, generator=generator, tracer=tracer)

run = client.run("What is the capital of France?", top_k=5)
# Spans are automatically emitted to console
```

Or use the factory directly:

```python
from ragwatch.instrumentation.instrumented_client import instrument_client

client = instrument_client(retriever=retriever, generator=generator, tracer=tracer)
run = client.run("What is the capital of France?", top_k=5)
```

### Exporter types

| Type | Description |
|------|-------------|
| `"console"` | Prints spans to stdout (local debugging) |
| `"otlp"` | Sends spans to an OTLP collector (e.g. Jaeger, Grafana Tempo) |
| `"none"` | Creates spans but does not export them (testing) |

### Span structure

```
ragwatch.rag_run
├── ragwatch.retrieval
└── ragwatch.generation
```

Each span includes attributes like `ragwatch.run_id`, `ragwatch.query`, latencies, document IDs, and scores.

### Current limitations

- No KPI calculation from traces yet
- No Grafana dashboard integration yet
- No OTLP collector required for console tracing
- No LLM token/cost metrics yet (heuristic generator only)

## Persistent pgvector/Postgres backend

RAGWatch uses PostgreSQL with pgvector as its primary persistent storage backend. Documents and embeddings are stored in structured tables (`ragwatch_documents`, `ragwatch_embeddings`), enabling SQL-based analysis and DB-level KPIs.

### Why pgvector?

- Persistent storage that survives process restarts
- SQL access to documents, embeddings, and metadata
- Enables future DB-based KPIs (document counts, embedding stats, collection analysis)
- Standard PostgreSQL ecosystem (backups, monitoring, extensions)

### Setup

```bash
# Start local PostgreSQL with pgvector
docker compose up -d postgres

# Set the database URL (or use .env)
cp .env.template .env
```

The default local development URL is:
```
postgresql://ragwatch:ragwatch@localhost:5433/ragwatch
```

**Note:** These credentials are for local development only.

### Database tables

| Table | Purpose |
|-------|---------|
| `ragwatch_documents` | Corpus documents with text, metadata, dataset_name, source |
| `ragwatch_embeddings` | Vector embeddings linked to documents, organized by collection |

### Run examples

```bash
python examples/run_squad_pgvector.py
python examples/query_pgvector_stats.py
```

### Environment variables

- `.env.template` — safe to commit, contains variable names with local defaults
- `.env` — local file (gitignored), loaded automatically by examples

### Current limitations

- No HNSW/IVFFlat vector index yet (uses exact search, fine for research scale)
- No Alembic migrations (tables are created on first use)
- DB-based KPI queries not integrated into KPIEngine yet

## Deterministic KPI calculation

RAGWatch includes a deterministic KPI engine that computes cheap, non-semantic monitoring metrics directly from a `RAGRun` object. No external services, API keys, or LLM calls required.

### KPI catalog

All KPI metadata (name, category, stage, source, description) is centralized in `src/ragwatch/metrics/catalog.py`. Metric classes only implement computation logic — they reference the catalog for their identity.

List all available KPIs:

```bash
python examples/list_available_kpis.py
```

Use the catalog programmatically:

```python
from ragwatch.metrics.catalog import list_kpis, list_kpis_by_category, KPICategory

# All KPIs
for defn in list_kpis():
    print(f"{defn.id}: {defn.description}")

# Filter by category
for defn in list_kpis_by_category(KPICategory.RETRIEVAL_QUALITY):
    print(defn.id)
```

### What the KPI engine does

Given a completed `RAGRun`, the `KPIEngine` computes a set of metrics and returns a structured `KPIReport` containing named results with values, categories, and descriptions.

### Supported metrics

| Metric | Category | Stage |
|--------|----------|-------|
| `num_retrieved_documents` | retrieval_quality | retrieval |
| `retrieval_score_min` | retrieval_quality | retrieval |
| `retrieval_score_max` | retrieval_quality | retrieval |
| `retrieval_score_mean` | retrieval_quality | retrieval |
| `retrieval_score_std` | retrieval_quality | retrieval |
| `retrieval_score_range` | retrieval_quality | retrieval |
| `retrieval_score_gap_top1_top2` | retrieval_quality | retrieval |
| `context_length_chars` | retrieval_quality | retrieval |
| `unique_retrieved_sources` | retrieval_quality | retrieval |
| `retrieval_redundancy` | retrieval_quality | retrieval |
| `answer_length_chars` | generation_quality | generation |
| `answer_length_words` | generation_quality | generation |
| `answer_to_context_length_ratio` | generation_quality | generation |
| `total_latency_ms` | runtime | end_to_end |
| `retrieval_latency_ms` | runtime | retrieval |
| `generation_latency_ms` | runtime | generation |

### Usage

```python
from ragwatch.metrics.report import KPIEngine, to_flat_dict

engine = KPIEngine()
report = engine.compute(rag_run)

for kpi in report.results:
    print(f"{kpi.name} = {kpi.value}")

# Flat dict for CSV export
flat = to_flat_dict(report)
```

### Run the example

```bash
python examples/run_toy_rag_with_kpis.py
```

### Current limitations

- Metrics are deterministic and non-semantic (no faithfulness or relevance judgment)
- No LLM-based evaluation judges yet
- No drift detection yet
- No Grafana dashboard yet
- No OpenTelemetry backend reading yet

### DB-based KPIs

RAGWatch also computes database-level metrics from the pgvector/Postgres tables using `DBKPIEngine`:

| Metric | Category | Source |
|--------|----------|--------|
| `db_document_count` | db_index_stats | postgres |
| `db_embedding_count` | db_index_stats | pgvector |
| `db_collection_count` | db_index_stats | pgvector |
| `db_avg_document_length` | db_index_stats | postgres |
| `db_avg_embedding_dim` | db_index_stats | pgvector |
| `db_collections` | db_index_stats | pgvector |
| `db_documents_per_source` | db_index_stats | postgres |

```python
from ragwatch.metrics.report import DBKPIEngine

engine = DBKPIEngine()
report = engine.compute()  # reads RAGWATCH_PGVECTOR_URL from .env

for kpi in report.results:
    print(f"{kpi.name} = {kpi.value}")
```

Run the example:

```bash
cp .env.template .env
docker compose up -d postgres
python examples/run_squad_pgvector.py   # index some documents first
python examples/query_pgvector_kpis.py
```

## Batch experiments and export

RAGWatch includes a batch experiment runner that executes a RAG pipeline over multiple QA examples, computes KPIs for each run, optionally captures a DB KPI snapshot, and exports structured results for analysis.

### What it does

1. Runs a `BaseRAGClient` over a list of `QAExample`s using `config.top_k`
2. Computes a `KPIReport` for each successful run
3. Records per-example errors without crashing the experiment
4. Optionally computes one DB KPI snapshot (`compute_db_kpis=True`)
5. Exports results to JSONL, CSV, and JSON for research use

Each example produces one `ExperimentExampleResult` (with `run`, `kpi_report`, and `error`), and the runner returns an `ExperimentResult` exposing `num_examples`, `num_successful`, and `num_failed`.

### Usage

```python
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import ExperimentConfig
from ragwatch.experiments.exporters import export_experiment

config = ExperimentConfig(
    experiment_name="my_experiment",
    dataset_name="squad_validation",
    retriever_name="tfidf",
    generator_name="heuristic",
    top_k=3,
    max_examples=20,
)

runner = ExperimentRunner(client=client, kpi_engine=kpi_engine)
result = runner.run(qa_examples=examples, config=config)

# Writes runs.jsonl, kpis.jsonl, kpis.csv, db_kpis.json, summary.json
export_experiment(result, "outputs/experiments/my_experiment")
```

Individual exporters (`export_runs_jsonl`, `export_kpis_jsonl`, `export_kpis_csv`, `export_db_kpis_json`, `export_experiment_summary_json`) are also available if you only need specific files.

### Run the example

```bash
pip install -e ".[datasets]"
python examples/run_squad_tfidf_experiment.py
```

This writes to `outputs/experiments/squad_tfidf/`.

To include a DB KPI snapshot (requires PostgreSQL):

```bash
pip install -e ".[datasets,vectordb]"
cp .env.template .env
docker compose up -d postgres
python examples/run_squad_pgvector_experiment.py
docker compose down
```

The pgvector example fails gracefully if `.env` / PostgreSQL is not available.

### Output files

| File | Content |
|------|---------|
| `runs.jsonl` | One line per example (config, example_id, question, gold answers, run_id, answer, retrieved doc IDs/scores/ranks/sources, latency); failed examples record their error |
| `kpis.jsonl` | One line per successful example (experiment columns + flat KPI metrics) |
| `kpis.csv` | One row per successful example (experiment config + example_id + query + run_id + one column per KPI name) |
| `db_kpis.json` | DB KPI snapshot if computed, otherwise `{"db_kpi_report": null}` |
| `summary.json` | Experiment config, counts, timestamp, and simple aggregate metrics |

Full retrieved document text is intentionally excluded from `runs.jsonl` to keep files small.

### What these files are useful for

- Paper plots and experiment comparison tables
- Later dashboards (Grafana, Prometheus, ClickHouse)
- Later Postgres experiment storage
- Later drift analysis over time
- Archiving experiment baselines

> **Note:** Postgres-backed experiment storage is not implemented yet — results are exported to files only.

## What's NOT included yet (intentionally)

- LLM-based generators or evaluators
- Semantic faithfulness/relevance judges
- Drift detection
- Grafana dashboards
- MCP (Model Context Protocol) integration
- Reading KPIs from OpenTelemetry trace storage

These will be added in future steps.

## What's next

- LLM-based generation (OpenAI, Anthropic)
- Semantic evaluation judges (faithfulness, relevance)
- Drift detection over time
- Grafana dashboard integration
- Experiment comparison tooling
