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

## OpenTelemetry and Grafana observability

RAGWatch ships an optional observability backend (OpenTelemetry Collector, Prometheus, Tempo, and Grafana) as part of the **root `docker-compose.yml`**. The services run under the `observability` Docker Compose profile, so the default stack (Postgres/pgvector) is unaffected.

### Start the default services

```bash
docker compose up -d
```

### Start the observability stack

```bash
docker compose --profile observability up -d
```

Then open:

| Service | URL |
|---------|-----|
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |
| Tempo | http://localhost:3200 |

Grafana is pre-provisioned with Prometheus and Tempo datasources.

### Run RAGWatch with OTLP enabled

```bash
RAGWATCH_OTEL_ENABLED=true \
RAGWATCH_OTEL_EXPORTER=otlp \
RAGWATCH_OTEL_ENDPOINT=http://localhost:4317 \
python examples/run_retriever_comparison.py
```

The OTel Collector receives OTLP on gRPC `4317` and HTTP `4318`, forwards traces to Tempo, and exposes metrics for Prometheus.

### Offline vs. runtime observability

- **Streamlit / CSV / plots** are for offline research inspection of completed experiments.
- **Grafana / OpenTelemetry** are for near real-time runtime observability of running experiments.

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

## Dataset registry and dataset-agnostic experiments

RAGWatch experiments can run on different QA/RAG datasets through a common registry, so the same scripts work across datasets without code changes.

### Common dataset schema

Every dataset is normalized into a single `RAGDataset` object (`src/ragwatch/core/schema.py`) with:

- `corpus` — a list of `Document`s
- `qa_examples` — a list of `QAExample`s
- `metadata`

Dataset adapters (under `src/ragwatch/datasets/adapters/`) are responsible for turning a raw source into this unified shape. The registry (`src/ragwatch/datasets/registry.py`) maps short dataset names to those adapters and constructs them lazily, so importing the registry never triggers heavy imports or downloads.

### Supported datasets

| Name | Description | Source |
|------|-------------|--------|
| `squad` | SQuAD v1.1 reading-comprehension QA | `rajpurkar/squad` |
| `hotpotqa` | HotpotQA multi-hop QA (fullwiki) | `hotpotqa/hotpot_qa` |

List them programmatically with `get_available_dataset_names()`.

### Default workflow

```bash
pip install -e ".[datasets,semantic,plots,dashboard]"

# Retriever comparison on different datasets
RAGWATCH_DATASET=squad    python examples/run_retriever_comparison.py
RAGWATCH_DATASET=hotpotqa python examples/run_retriever_comparison.py

# Rule-based drift experiment on different datasets
RAGWATCH_DATASET=squad    python examples/run_drift_experiment.py
RAGWATCH_DATASET=hotpotqa python examples/run_drift_experiment.py
```

`examples/run_retriever_comparison.py` and `examples/run_drift_experiment.py` are the preferred dataset-agnostic research entry points. The dataset defaults to `squad` when `RAGWATCH_DATASET` is unset.

### Dataset configuration (environment variables)

| Variable | Meaning | Default |
|----------|---------|---------|
| `RAGWATCH_DATASET` | Registry dataset name | `squad` |
| `RAGWATCH_DATASET_SPLIT` | Split to load | the registry entry's default (e.g. `validation`) |
| `RAGWATCH_DATASET_MAX_EXAMPLES` | QA examples to run | 120 (comparison) / 20 (drift) |
| `RAGWATCH_DATASET_LOAD_MAX_EXAMPLES` | Rows to load to build the corpus | 200 (drift) |

### Output directories

Output directories are dataset- and mode-aware, so different datasets and semantic modes never overwrite each other:

```
outputs/comparisons/squad_retrievers_semantic_local/
outputs/comparisons/hotpotqa_retrievers_semantic_local/
outputs/drift/squad_tfidf_semantic_local/
outputs/drift/hotpotqa_tfidf_semantic_local/
```

The mode segment is `semantic_<provider>` by default (for example `semantic_local`) or `base` when semantic KPIs are disabled.

### Backward-compatible SQuAD scripts

The historical SQuAD scripts still work and are now thin wrappers that default the dataset to SQuAD:

| Wrapper | Equivalent to |
|---------|---------------|
| `python examples/run_squad_retriever_comparison.py` | `RAGWATCH_DATASET=squad python examples/run_retriever_comparison.py` |
| `python examples/run_squad_drift_experiment.py` | `RAGWATCH_DATASET=squad python examples/run_drift_experiment.py` |
| `python examples/generate_squad_comparison_plots.py` | `RAGWATCH_DATASET=squad python examples/generate_comparison_plots.py` |
| `python examples/generate_squad_drift_plots.py` | `RAGWATCH_DATASET=squad python examples/generate_drift_plots.py` |

### Adding a new dataset

1. Write an adapter that returns a `RAGDataset`:

   ```python
   # src/ragwatch/datasets/adapters/my_adapter.py
   from ragwatch.core.schema import Document, QAExample, RAGDataset


   class MyAdapter:
       name = "mydataset"

       def load(self, split=None, max_examples=None) -> RAGDataset:
           corpus = [Document(doc_id="d0", text="...")]
           qa = [QAExample(example_id="q0", question="...", answers=["..."])]
           return RAGDataset(name=self.name, corpus=corpus, qa_examples=qa)
   ```

2. Register it in `src/ragwatch/datasets/registry.py` by adding a `DatasetRegistryEntry` to `_REGISTRY`.

3. Run any generic script with `RAGWATCH_DATASET=mydataset`.

## Retriever comparison experiments

RAGWatch can run the same QA dataset across multiple retriever backends and top-k settings, then export combined comparison files for paper-style analysis.

### Why comparison experiments?

Comparing retrievers (TF-IDF vs Chroma vs Qdrant vs pgvector) and top-k settings side by side is a core research task. The comparison runner reuses the existing `ExperimentRunner` for each setting and aggregates the results into shared tables, so you don't have to stitch experiments together by hand.

### How it works

`ComparisonRunner` takes a list of `RetrieverExperimentSpec`s (each wrapping a `BaseRAGClient`) and a `ComparisonConfig` with `top_k_values`. For each retriever × top-k it runs one experiment and stores the `ExperimentResult` under a readable key like `tfidf_top3`, `chroma_top5`. A failure in one setting does not abort the whole comparison.

### Run the example

```bash
pip install -e ".[datasets]"            # TF-IDF only
pip install -e ".[datasets,vectordb]"   # also include Chroma + Qdrant + pgvector
python examples/run_retriever_comparison.py                 # defaults to squad
RAGWATCH_DATASET=hotpotqa python examples/run_retriever_comparison.py
```

The SQuAD-specific wrapper `python examples/run_squad_retriever_comparison.py` remains available and is equivalent to `RAGWATCH_DATASET=squad python examples/run_retriever_comparison.py`.

The default comparison includes:

```
TF-IDF
Chroma
Qdrant
```

TF-IDF always runs; Chroma and Qdrant are included automatically when their optional dependencies are installed. **pgvector is included automatically when Postgres is configured and reachable** (via `RAGWATCH_PGVECTOR_URL`); otherwise it is skipped gracefully with a clear message and the comparison still completes with the other retrievers.

To include pgvector:

```bash
cp .env.template .env
docker compose up -d postgres
python examples/run_retriever_comparison.py
```

When pgvector is included, the dashboard and plotting outputs automatically show `pgvector` as another retriever (they read `combined_kpis.csv` and `comparison_summary.csv`), and the `pgvector_top*` experiment folders contain a non-null `db_kpis.json` DB KPI snapshot. Non-pgvector retrievers keep a null `db_kpis.json`. pgvector stays inside the same mode-specific directory; including it does not create a separate output directory.

### Mode-aware output directories

Semantic KPIs are enabled by default in the main research experiment scripts. The comparison example chooses its output directory based on the run mode, so a semantic run and an opt-out run never overwrite each other:

| Command | Output directory |
|---------|------------------|
| `python examples/run_retriever_comparison.py` | `outputs/comparisons/squad_retrievers_semantic_local/` |
| `RAGWATCH_DATASET=hotpotqa python examples/run_retriever_comparison.py` | `outputs/comparisons/hotpotqa_retrievers_semantic_local/` |
| `RAGWATCH_SEMANTIC_PROVIDER=openai python examples/run_retriever_comparison.py` | `outputs/comparisons/squad_retrievers_semantic_openai/` |
| `RAGWATCH_SEMANTIC_PROVIDER=azure_openai python examples/run_retriever_comparison.py` | `outputs/comparisons/squad_retrievers_semantic_azure_openai/` |
| `RAGWATCH_DISABLE_SEMANTIC_KPIS=true python examples/run_retriever_comparison.py` | `outputs/comparisons/squad_retrievers_base/` |

The `comparison_name` inside the exported CSV/JSON matches the directory (for example `squad_retrievers_semantic_local`). When semantic KPIs are disabled, the run uses the base directory and omits the semantic columns. If a non-local provider is requested but its configuration is missing, the script prints a clear message and exits with a non-zero status instead of silently falling back.

Set `RAGWATCH_COMPARISON_OUTPUT_DIR` to use a custom output directory verbatim:

```bash
RAGWATCH_COMPARISON_OUTPUT_DIR=outputs/comparisons/custom_test \
python examples/run_retriever_comparison.py
```

### Output files

```
outputs/comparisons/squad_retrievers_semantic_local/
  combined_kpis.csv          # one row per successful run across all experiments
  comparison_summary.csv     # one row per retriever/top-k setting with averaged KPIs
  comparison_summary.json    # comparison metadata + per-experiment aggregates
  experiments/
    tfidf_top3/              # full per-experiment export (runs.jsonl, kpis.jsonl, ...)
    tfidf_top5/
    chroma_top3/
    ...
```

Each per-experiment folder is written with the standard experiment exporters (`runs.jsonl`, `kpis.jsonl`, `kpis.csv`, `db_kpis.json`, `summary.json`).

### What these files are useful for

- Paper comparison tables
- Paper plots
- A later Streamlit dashboard
- Later Postgres experiment storage

## Research dashboard

RAGWatch includes a simple, local Streamlit dashboard that reads exported comparison files and visualizes the results. It is a research tool for exploring KPIs interactively — **not** Grafana, and it requires no database, OpenTelemetry backend, or API keys.

The dashboard has two tabs, selectable at the top of the page:

- **Retriever comparison** — explores `outputs/comparisons/...` retriever comparison runs
- **Drift experiments** — explores `outputs/drift/...` rule-based drift runs (see [Rule-based drift experiments](#rule-based-drift-experiments))

### What it does

#### Retriever comparison tab

- **Overview** — comparison/dataset name, number of settings, successful/failed run counts, available retrievers and top-k values
- **Retriever comparison** — the `comparison_summary.csv` table plus bar charts of average latency, retrieval score, redundancy, and answer length by retriever/top-k
- **KPI explorer** — pick any numeric KPI from `combined_kpis.csv` and view its distribution and per-retriever breakdown
- **Query / run inspector** — drill into a single experiment + example to see the query, answer, retrieved doc IDs/scores/ranks, and KPI values
- **Failure inspection** — a table of failed examples (experiment, example_id, question, error), or a confirmation that none failed

#### Drift experiments tab

- **Scenario overview** — the `drift_summary.csv` table, scenario/perturbation counts, and bar charts of average retrieval score, redundancy, answer length, and latency per scenario
- **Clean vs drifted scenario** — pick a baseline (defaults to `clean`) and a drift scenario and compare their summary KPIs side by side
- **Manifest inspection** — key drift manifest fields (perturbation type, severity, seed, added/removed/modified doc and query IDs) plus the full manifest JSON
- **Clean vs drifted example inspection** — pick an `example_id` and see the question, answer, and retrieved docs side by side for clean vs drifted runs; for query-shift scenarios the original question is shown when present

### Setup

```bash
pip install -e ".[dashboard]"
```

### First generate outputs

```bash
# Retriever comparison tab
python examples/run_squad_retriever_comparison.py

# Drift experiments tab
python examples/run_squad_drift_experiment.py
```

### Run the dashboard

```bash
streamlit run src/ragwatch/dashboard/app.py
# or
python examples/run_dashboard.py
```

By default the comparison tab reads `outputs/comparisons/squad_retrievers_semantic_local` and the drift tab reads `outputs/drift/squad_tfidf_semantic_local` (the semantic-by-default outputs of the main scripts). If `outputs/comparisons/` or `outputs/drift/` contain directories, the sidebar offers them in a selectbox, and you can always paste a custom path. Each tab shows a hint to generate its outputs if they are missing.

### Files it reads

| File | Used for |
|------|----------|
| `combined_kpis.csv` | KPI explorer, run inspector |
| `comparison_summary.csv` | Retriever comparison table and charts |
| `comparison_summary.json` | Overview metadata |
| `experiments/*/runs.jsonl` | Answers, retrieved docs, and failure inspection |
| `drift_summary.csv` | Drift scenario overview and clean vs drifted comparison |
| `<scenario>/runs.jsonl` | Clean vs drifted example inspection |
| `<scenario>/drifted_dataset/manifest.json` | Drift manifest inspection |
| `<scenario>/drifted_dataset/qa_examples.jsonl` | Original questions for query-shift scenarios |

This is a local research dashboard, not Grafana.


## Paper-quality plots

Separate from the interactive dashboard, RAGWatch can generate **static, paper-ready figures** from the same comparison outputs using matplotlib. These are reproducible PNGs suitable for papers, reports, slides, and debugging retriever behavior.

### Setup

```bash
pip install -e ".[plots]"
```

### First generate comparison outputs

```bash
python examples/run_squad_retriever_comparison.py
```

### Then generate plots

```bash
python examples/generate_squad_comparison_plots.py
```

By default it reads `outputs/comparisons/squad_retrievers_semantic_local/` and saves figures to `outputs/figures/squad_retrievers_semantic_local/`. The figures directory mirrors the comparison directory name so semantic and opt-out figures do not overwrite each other.

To plot an opt-out (base) run, point it at that comparison directory:

```bash
RAGWATCH_COMPARISON_OUTPUT_DIR=outputs/comparisons/squad_retrievers_base \
python examples/generate_squad_comparison_plots.py
# -> writes outputs/figures/squad_retrievers_base/
```

Use `RAGWATCH_FIGURE_OUTPUT_DIR` to override the figures directory verbatim:

```bash
RAGWATCH_COMPARISON_OUTPUT_DIR=outputs/comparisons/squad_retrievers_semantic_local \
RAGWATCH_FIGURE_OUTPUT_DIR=outputs/figures/custom_semantic \
python examples/generate_squad_comparison_plots.py
```

Figures are saved to the resolved figures directory (default `outputs/figures/squad_retrievers_semantic_local/`):

| Figure | Shows |
|--------|-------|
| `latency_by_retriever.png` | Avg total latency per retriever/top-k |
| `retrieval_score_by_retriever.png` | Avg mean retrieval score |
| `retrieval_redundancy_by_retriever.png` | Avg retrieval redundancy |
| `answer_length_by_retriever.png` | Avg answer length (words) |
| `top_k_latency_sensitivity.png` | Latency across top-k, by retriever |
| `top_k_retrieval_score_sensitivity.png` | Retrieval score across top-k, by retriever |

When the comparison directory contains semantic KPI columns (the default), three additional semantic figures are also produced (query/context, answer/context, and answer/query similarity by retriever).

These figures are useful for papers, reports, slides, and debugging retriever behavior. If the comparison outputs are missing, the script tells you to run `python examples/run_squad_retriever_comparison.py` first.

## Semantic KPI metrics

RAGWatch can optionally compute **embedding-based semantic KPIs** that measure how semantically related the query, retrieved context, and generated answer are. These complement the deterministic KPIs and are a first relevance layer. LLM-as-judge faithfulness is **not** implemented yet.

Semantic KPIs (category `semantic_quality`):

| KPI | Measures |
|-----|----------|
| `query_context_similarity_mean` | Mean cosine similarity of query to each retrieved doc |
| `query_context_similarity_max` | Max cosine similarity of query to any retrieved doc |
| `answer_context_similarity_mean` | Mean cosine similarity of answer to each retrieved doc |
| `answer_context_similarity_max` | Max cosine similarity of answer to any retrieved doc |
| `answer_query_similarity` | Cosine similarity of answer to query |

They are **optional** and never required by the default `KPIEngine`. They are computed by a separate `SemanticKPIEngine` that uses a configurable embedding provider.

### Supported providers

- **local** — sentence-transformers (no API key)
- **openai** — direct OpenAI embeddings API
- **azure_openai** — Azure OpenAI embeddings deployment

### Install

```bash
pip install -e ".[semantic]"
```

### Local example (no API key)

```bash
RAGWATCH_SEMANTIC_PROVIDER=local
python examples/run_toy_rag_with_semantic_kpis.py
```

### Direct OpenAI example

```bash
RAGWATCH_SEMANTIC_PROVIDER=openai
RAGWATCH_OPENAI_API_KEY=...
RAGWATCH_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
python examples/run_toy_rag_with_semantic_kpis.py
```

### Azure OpenAI example

```bash
RAGWATCH_SEMANTIC_PROVIDER=azure_openai
RAGWATCH_AZURE_OPENAI_API_KEY=...
RAGWATCH_AZURE_OPENAI_ENDPOINT=...
RAGWATCH_AZURE_OPENAI_API_VERSION=2024-02-01
RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT=...
python examples/run_toy_rag_with_semantic_kpis.py
```

Note: for direct OpenAI, `RAGWATCH_OPENAI_EMBEDDING_MODEL` is a **model name**; for Azure OpenAI, `RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT` is a **deployment name** (it is passed where OpenAI expects the model argument).

### Semantic KPIs in batch and comparison experiments

Semantic KPIs can also be computed inside the batch `ExperimentRunner` and the retriever `ComparisonRunner`, not just the standalone examples. **Semantic KPIs are enabled by default in the main research experiment scripts** (`run_squad_retriever_comparison.py` and `run_squad_drift_experiment.py`), using the local sentence-transformers provider, which needs no API key. The library internals remain opt-in (the default `KPIEngine` never requires semantic dependencies); only the research scripts default semantic on.

Default workflow (local, no API key):

```bash
pip install -e ".[semantic,dashboard,plots]"

python examples/run_squad_retriever_comparison.py
python examples/generate_squad_comparison_plots.py
python examples/run_squad_drift_experiment.py
python examples/generate_squad_drift_plots.py
python examples/list_available_kpis.py

streamlit run src/ragwatch/dashboard/app.py
```

You do **not** need to set any environment variable to get semantic KPIs. The comparison run writes to `outputs/comparisons/squad_retrievers_semantic_local/` and the drift run writes to `outputs/drift/squad_tfidf_semantic_local/`.

Programmatically, attach a `SemanticKPIEngine` and enable the flag:

```python
runner = ExperimentRunner(client=client, semantic_kpi_engine=SemanticKPIEngine(provider))
result = runner.run(qa_examples, config, compute_semantic_kpis=True)
```

For comparisons, each `RetrieverExperimentSpec` can independently set `semantic_kpi_engine` and `compute_semantic_kpis=True`.

Semantic KPI columns (for example `query_context_similarity_mean`, `answer_context_similarity_mean`, `answer_query_similarity`) appear in the exported `kpis.csv`, `kpis.jsonl`, and `combined_kpis.csv`, and their averages (`avg_query_context_similarity_mean`, `avg_answer_context_similarity_mean`, `avg_answer_query_similarity`) appear in `comparison_summary.csv` and the drift `drift_summary.csv`. The dashboard KPI explorer and the comparison/drift plots pick these up automatically when present.

Each semantic provider writes to its own mode-specific directory (`outputs/comparisons/squad_retrievers_semantic_local/`, `..._semantic_openai/`, `..._semantic_azure_openai/`), so different providers never overwrite each other. See [Mode-aware output directories](#mode-aware-output-directories).

### Switching provider

Set `RAGWATCH_SEMANTIC_PROVIDER` to switch from the default local provider to a paid API. The configuration must be complete or the script exits with a clear error (it does not silently fall back).

**OpenAI comparison example:**

```bash
RAGWATCH_SEMANTIC_PROVIDER=openai \
RAGWATCH_OPENAI_API_KEY=... \
RAGWATCH_OPENAI_EMBEDDING_MODEL=text-embedding-3-small \
python examples/run_squad_retriever_comparison.py
```

**Azure OpenAI comparison example:**

```bash
RAGWATCH_SEMANTIC_PROVIDER=azure_openai \
RAGWATCH_AZURE_OPENAI_API_KEY=... \
RAGWATCH_AZURE_OPENAI_ENDPOINT=... \
RAGWATCH_AZURE_OPENAI_API_VERSION=2024-02-01 \
RAGWATCH_AZURE_OPENAI_EMBEDDING_DEPLOYMENT=... \
python examples/run_squad_retriever_comparison.py
```

OpenAI and Azure OpenAI embeddings may incur cost. Start with small datasets (lower `max_examples`) before running larger comparisons.

### Opting out of semantic KPIs

To run the main scripts deterministic-only (no semantic dependencies, models, or keys), set the opt-out flag:

```bash
RAGWATCH_DISABLE_SEMANTIC_KPIS=true python examples/run_squad_retriever_comparison.py
RAGWATCH_DISABLE_SEMANTIC_KPIS=true python examples/run_squad_drift_experiment.py
```

Opt-out runs write to the base directories (`outputs/comparisons/squad_retrievers_base/` and `outputs/drift/squad_tfidf/`) and omit the semantic columns.

**Reminders:**

- Do not commit `.env` (it stays gitignored); `.env.template` documents the variables.
- Use small example sizes when using paid APIs.
- LLM-as-judge faithfulness is not implemented yet.
- The library internals keep semantic KPIs optional; only the research scripts default them on, and `RAGWATCH_DISABLE_SEMANTIC_KPIS=true` turns them off.

## Rule-based drift experiments

RAGWatch can build **deterministic, rule-based drift scenarios** that simulate controlled RAG degradation, then run clean vs perturbed data and compare KPIs. These perturbations are reproducible rules — **not** LLM-generated drift and **not** LLM-as-judge metrics.

### Terminology

- **Perturbation** — the actual rule-based modification applied to data.
- **Drift scenario** — an experimental condition built from one or more perturbations.
- **Drift experiment** — running RAGWatch on clean vs perturbed data and comparing KPIs.

### Production-relevant perturbations

| Perturbation | Production interpretation |
|--------------|---------------------------|
| **Corpus contamination / hard-negative injection** | Stale pages, wrong tenant/domain docs, near-topic chunks, or duplicate noisy pages enter the index. |
| **Source outage / partial index failure** | A connector, source, ACL filter, ingestion job, or index rebuild silently drops part of the corpus. |
| **Parser noise / ingestion degradation** | PDF/OCR/HTML parsing, chunking, boilerplate removal, or encoding quality gets worse. |
| **Query distribution shift** | Users start asking with different wording, typos, conversational phrasing, or irrelevant extra tokens. |

All transforms live in [src/ragwatch/drift/transforms.py](src/ragwatch/drift/transforms.py), never mutate the input dataset, return a new `RAGDataset`, are deterministic for a given seed, and record useful metadata. Scenario builders live in [src/ragwatch/drift/scenarios.py](src/ragwatch/drift/scenarios.py).

### Default scenarios

`create_drift_scenarios(dataset)` returns these conditions:

```
clean
corpus_contamination_10
corpus_contamination_30
source_outage_10
source_outage_30
parser_noise_10
query_shift_10
query_shift_30
```

### Run the example

```bash
pip install -e ".[datasets,semantic]"
python examples/run_squad_drift_experiment.py
```

The retriever is **TF-IDF**, and semantic KPIs are computed by default using the local sentence-transformers provider (no API key). To run deterministic-only without semantic dependencies, opt out:

```bash
RAGWATCH_DISABLE_SEMANTIC_KPIS=true python examples/run_squad_drift_experiment.py
```

### Output structure

```
outputs/drift/squad_tfidf_semantic_local/
  clean/                       # full per-scenario experiment export
    drifted_dataset/           # persisted snapshot of the drifted dataset
      corpus.jsonl
      qa_examples.jsonl
      manifest.json
  corpus_contamination_10/
  corpus_contamination_30/
  source_outage_10/
  source_outage_30/
  parser_noise_10/
  query_shift_10/
  query_shift_30/
  drift_summary.csv            # one row per scenario
```

The output directory is mode-aware: the default is `outputs/drift/squad_tfidf_semantic_local/` (with `..._semantic_openai/` and `..._semantic_azure_openai/` for other providers), and opt-out runs use `outputs/drift/squad_tfidf/`.

Each per-scenario folder is written with the standard experiment exporters (`runs.jsonl`, `kpis.jsonl`, `kpis.csv`, `db_kpis.json`, `summary.json`). The combined `drift_summary.csv` has one row per scenario with columns:

```
scenario_name, perturbation_type, severity, production_interpretation,
num_documents, num_examples, num_successful, num_failed,
avg_total_latency_ms, avg_retrieval_score_mean,
avg_retrieval_redundancy, avg_answer_length_words
```

When semantic KPIs are enabled (the default), three semantic average columns are appended: `avg_query_context_similarity_mean`, `avg_answer_context_similarity_mean`, `avg_answer_query_similarity`.

### Persisted dataset snapshots

Every scenario folder also contains a `drifted_dataset/` snapshot so you can see and reproduce exactly what changed:

```
drifted_dataset/
  corpus.jsonl        # one JSON object per document (doc_id, text, metadata)
  qa_examples.jsonl   # one JSON object per QA example (example_id, question, answers, metadata)
  manifest.json       # scenario config, counts, and change details
```

The snapshot preserves all drift metadata (synthetic/contaminated docs, degraded text, original questions for shifted queries). The `manifest.json` records `original_name`, `scenario_name`, `perturbation_type`, `severity`, `random_seed`, `production_interpretation`, document/example counts, and — when applicable — added/removed/modified doc IDs and modified query IDs. This is useful to:

- inspect exactly what changed in each scenario
- reproduce drift scenarios deterministically
- debug why KPIs changed
- support paper/review reproducibility

See [src/ragwatch/drift/persistence.py](src/ragwatch/drift/persistence.py).

### Inspect drift results in the dashboard

After running the drift example, the **Drift experiments** tab of the [research dashboard](#research-dashboard) can inspect these outputs interactively: compare clean vs drifted scenarios, see per-scenario KPI differences, view the scenario manifest, and step through individual questions to see how answers and retrieved documents changed (including the original question for query-shift scenarios).

```bash
pip install -e ".[dashboard]"
python examples/run_squad_drift_experiment.py
streamlit run src/ragwatch/dashboard/app.py
```

### Custom output directory

```bash
RAGWATCH_DRIFT_OUTPUT_DIR=outputs/drift/custom_squad_tfidf \
python examples/run_squad_drift_experiment.py
```

### Static drift plots

Separate from the interactive dashboard, RAGWatch can render **static, paper-ready figures** from `drift_summary.csv` using matplotlib (no seaborn, Plotly, or Streamlit). These are reproducible PNGs for papers, reports, slides, and debugging drift behavior.

```bash
pip install -e ".[plots]"
python examples/run_squad_drift_experiment.py
python examples/generate_squad_drift_plots.py
```

By default the figures are written next to the drift outputs:

```
outputs/drift/squad_tfidf_semantic_local/figures/
  retrieval_score_by_drift_scenario.png
  retrieval_redundancy_by_drift_scenario.png
  answer_length_by_drift_scenario.png
  latency_by_drift_scenario.png
```

Each figure is a bar chart of one averaged KPI per drift scenario (scenario names are rotated for readability). Because semantic KPIs are on by default, `drift_summary.csv` contains semantic drift columns (`avg_query_context_similarity_mean`, `avg_answer_context_similarity_mean`, `avg_answer_query_similarity`), so three additional semantic plots are generated automatically:

```
outputs/drift/squad_tfidf_semantic_local/figures/
  query_context_similarity_by_drift_scenario.png
  answer_context_similarity_by_drift_scenario.png
  answer_query_similarity_by_drift_scenario.png
```

If those columns are absent, the semantic plots are skipped without failing.

Custom directories are supported. Read from a different drift directory:

```bash
RAGWATCH_DRIFT_OUTPUT_DIR=outputs/drift/custom_squad_tfidf \
python examples/generate_squad_drift_plots.py
```

Write figures to a custom directory (used verbatim) instead of `<drift dir>/figures/`:

```bash
RAGWATCH_DRIFT_OUTPUT_DIR=outputs/drift/custom_squad_tfidf \
RAGWATCH_DRIFT_FIGURE_OUTPUT_DIR=outputs/drift/custom_squad_tfidf/figures_custom \
python examples/generate_squad_drift_plots.py
```

If `drift_summary.csv` is missing, the script prints a hint to run `python examples/run_squad_drift_experiment.py` first.

### Later drift layers

This is the first drift layer. Later steps can add: semantic drift summaries, Chroma/Qdrant/pgvector drift, and LLM-generated perturbations as an optional future extension.

## What's NOT included yet (intentionally)



- LLM-based generators or evaluators
- LLM-as-judge faithfulness metrics
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
