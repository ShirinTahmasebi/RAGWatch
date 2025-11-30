# RAGWatch

Core logging utilities plus a HotpotQA-focused RAG prototype.

## Quick start

1. **Clone & create env**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **Configure secrets**
   - Copy `.env.template` to `.env` and set `OPENAI_API_KEY`. That file is ignored by git.
   - The code loads `.env` via `python-dotenv`, so you do not need to export it manually.
3. **Run tests**
   ```bash
   source .venv/bin/activate
   python -m unittest tests.test_schema tests.test_writers tests.test_logger tests.test_data_prep tests.test_retriever -v
   ```

## Project structure (implemented so far)

```
RAGWatch/
├── ragwatch/                 # core logging & monitoring package
│   ├── utils/                # env helpers, writers, shared helpers
│   │   ├── env_manager.py
│   │   └── writers.py
│   ├── logging/              # JSONL-backed logger
│   │   └── logger.py
│   ├── models/               # Pydantic models for logged artifacts
│   │   ├── retrieved_doc.py
│   │   └── run_record.py
│   └── monitor/              # Session-based monitor context
│       └── monitor.py
├── ragwatch_client/          # dataset-agnostic runner + CLI
│   ├── __main__.py           # Typer CLI (python -m ragwatch_client <dataset> ...)
│   ├── runner.py             # shared evaluation/stream harness
│   ├── datasets/
│   │   ├── base.py           # Dataset protocol + wiring helpers/factory
│   │   └── hotpotqa/
│   │       └── data.py       # dataset loaders/dummy fallbacks
│   ├── pipelines/
│   │   ├── base.py           # RAG pipeline builder interface
│   │   └── basic.py          # Shared LangChain-based implementation
│   └── vectorstores/
│       ├── base.py           # Vectorstore adapter interface
│       └── faiss.py          # FAISS-backed adapter implementation
├── tests/
│   ├── test_schema.py
│   ├── test_writers.py
│   ├── test_logger.py
│   ├── test_data_prep.py
│   └── test_retriever.py
└── requirements.txt
```

## Environment configuration

Copy `.env.template` to `.env` and fill in the required values:

- `OPENAI_API_KEY`: passed to LangChain’s OpenAI clients.
- `RAGWATCH_LOG_DIR`: base directory for JSONL logs (e.g., `logs`).
- `RAGWATCH_INDEX_DIR`: base directory where dataset vectorstores are stored (each dataset writes to its own subfolder).
- `RAGWATCH_VERSION`: semantic/version label recorded with every run (e.g., `v1`).
- `RAGWATCH_HOTPOTQA_SPLIT`: HotpotQA split to load via Hugging Face (default `validation`).
- `RAGWATCH_HOTPOTQA_SAMPLE_SIZE`: Number of rows/docs to sample from the split when building the corpus (default `25`).
- `RAGWATCH_HOTPOTQA_USE_DUMMY_DATA`: Set to `true` to skip downloading HotpotQA and fall back to the tiny built-in dummy set (useful for tests/offline).

All modules resolve paths via `ragwatch.utils.env_manager`, so nothing in the codebase hardcodes directories anymore.

## Core modules

### Logging & monitoring (`ragwatch`)

Core components now live in focused subpackages:
- `ragwatch.models`: houses `RetrievedDoc` and `RAGRunRecord`—Pydantic models that capture every aspect of a RAG execution (dataset/version metadata, QA pairs, retrieved docs, latency metrics, token usage, and arbitrary extras).
- `ragwatch.utils`: contains `env_manager` (centralized `.env` loader + helpers) and `writers` (the `JSONLWriter` that appends serialized `RAGRunRecord` entries to disk, creating directories as needed).
- `ragwatch.logging`: exposes the low-level `RAGWatchLogger` plus `ConsoleStepLogger`, wiring the models + writer for JSONL telemetry and optionally printing colorized console updates.
- `ragwatch.monitor`: implements `RAGMonitor` / `SessionContext`, a thin wrapper that lets you instrument existing RAG pipelines with a `with monitor.session(...)` block, optionally enabling `log_steps` for automatic console narration of each run.

#### Example usage

```python
from ragwatch import RAGMonitor

monitor = RAGMonitor(
   log_dir="logs/hotpotqa",
   dataset_name="hotpotqa",
   version="demo",
)

with monitor.session(question=qa["question"], session_id=qa["id"], log_steps=True) as session:
   docs_with_scores = retriever.similarity_search_with_score(qa["question"], k=5)
   session.record_retrieval(docs_with_scores)

   answer = rag_chain.invoke(qa["question"])
   session.record_answer(
      getattr(answer, "content", None) or str(answer),
      latency_ms={"total": 850.0},
      token_usage={"prompt": 210, "completion": 96},
   )
   session.set_gold_answer(qa.get("answer", ""))

print("Runs recorded in logs/hotpotqa_demo.jsonl")

# Share the same ConsoleStepLogger instance across monitors if you
# want consistent CLI output:
# from ragwatch.logging import ConsoleStepLogger
# step_logger = ConsoleStepLogger()
# monitor = RAGMonitor(..., step_logger=step_logger, default_log_steps=True)
```

If you need full control, you can still instantiate `RAGWatchLogger` directly; the monitor simply wraps it with a friendlier API.

#### Module tests

```bash
python -m unittest tests.test_schema tests.test_writers tests.test_logger tests.test_monitor
```

### HotpotQA dataset (`ragwatch_client.datasets.hotpotqa`)

- **Data-only class**: `HotpotQADataSource` implements the `CorpusDataset` protocol so it only concerns itself with `load_questions()` and `build_document_corpus()` (still powered by `ragwatch_client.datasets.hotpotqa.data`).
- **Vectorstore adapter**: `ragwatch_client.vectorstores.faiss.FaissVectorStoreAdapter` encapsulates FAISS builds/loads using the shared `RAGWATCH_INDEX_DIR`, and supports dependency-injected embeddings/paths for tests.
- **RAG pipeline**: `ragwatch_client.pipelines.basic.BasicRAGPipelineBuilder` wires any retriever into a LangChain prompt + `ChatOpenAI(model="gpt-4o-mini", temperature=0.1)` call.
- **Factory wiring** (`ragwatch_client.factory`): `build_hotpotqa_dataset()` now lives in the factory registry and is built via `RAGDatasetBuilder`, so all wiring of corpus/vectorstore/pipeline pieces happens outside the dataset module. Swapping adapters or prompts is an exercise in editing that builder recipe.
- **Generic runner & CLI**: `ragwatch_client.runner` remains dataset-agnostic while the Typer CLI instantiates the builder registered under `ragwatch_client.factory.DATASETS`.

### Adding another dataset quickly

1. Create `ragwatch_client/datasets/<name>/` and implement a `CorpusDataset`-compatible data source (usually a small class that calls into your `data.py`). No vectorstore/pipeline code should live here.
2. Choose a vectorstore adapter (`FaissVectorStoreAdapter` or your own subclass of `VectorStoreAdapter`).
3. Pick or implement a `RAGPipelineBuilder` (e.g., `BasicRAGPipelineBuilder`).
4. In `ragwatch_client/factory/registry.py` (or a helper it imports), use `RAGDatasetBuilder` to compose your corpus/vectorstore/pipeline combo and expose a `build_<name>_dataset()` function that returns a `DatasetClient`.
5. Register the builder in `ragwatch_client/factory.DATASETS` so the CLI can discover it.
6. Add env entries (log/index dirs, split/sample overrides) to `.env.template` as needed.

The dataset itself stays cleanly focused on documents/questions, while the vectorstore + pipeline plumbing is swapped in via the factory.

## Usage snippet (HotpotQA prototype)

```python
from ragwatch_client.factory import build_hotpotqa_dataset

dataset = build_hotpotqa_dataset()
resources = dataset.prepare_resources()  # returns RAGRuntimeResources
answer = resources.rag_chain.invoke("Who is Barack Obama?")
print(answer)
```

To instrument the above chain with rich telemetry, run the evaluation harness:

```bash
python -m ragwatch_client hotpotqa eval
```

It ensures the FAISS index exists, invokes the RAG chain over the dummy dataset, and writes JSONL logs under the directory specified by `RAGWATCH_LOG_DIR`. Override the defaults by passing `--log-dir`; the run `version` is sourced from the `RAGWATCH_VERSION` value in your `.env` file.

For continuous monitoring, run:

```bash
python -m ragwatch_client hotpotqa stream --interval 1.0 --max-iterations 10
```

This repeatedly samples questions (cycling through the dummy set for now), waits the requested interval between calls, and logs each result so you can watch performance trends over time.

## Next steps
- Replace the placeholder HotpotQA loaders with real data ingestion + chunking.
- Expand logging metadata (e.g., model version, guardrail flags) and add monitoring dashboards.
- Integrate evaluation harnesses to compare runs across datasets.
