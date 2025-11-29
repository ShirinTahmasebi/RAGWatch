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
├── ragwatch/                 # core logging package
│   ├── schema.py             # Pydantic models: RetrievedDoc, RunRecord
│   ├── writers.py            # JSONLWriter appends runs to disk
│   ├── logger.py             # RAGWatchLogger high-level API
│   └── settings.py           # Shared environment helpers (paths, secrets)
├── ragwatch_hotpotqa/
│   ├── data_prep.py          # placeholder loaders for questions/docs
│   ├── retriever.py          # FAISS build/load/ensure helpers
│   ├── build_rag.py          # LangChain-based RAG chain (ChatOpenAI + retriever)
│   ├── run_eval.py           # Evaluation harness that logs via RAGWatch
│   └── __main__.py           # Typer CLI entry point (python -m ragwatch_hotpotqa)
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
- `RAGWATCH_HOTPOTQA_LOG_DIR`: HotpotQA-specific log directory (e.g., `logs/hotpotqa`).
- `RAGWATCH_HOTPOTQA_INDEX_DIR`: location on disk for the FAISS index (e.g., `ragwatch_hotpotqa/index`).
- `RAGWATCH_HOTPOTQA_SPLIT`: HotpotQA split to load via Hugging Face (default `validation`).
- `RAGWATCH_HOTPOTQA_SAMPLE_SIZE`: Number of rows/docs to sample from the split when building the corpus (default `25`).
- `RAGWATCH_HOTPOTQA_USE_STUBS`: Set to `true` to skip downloading HotpotQA and fall back to the tiny built-in stubs (useful for tests/offline).

All modules resolve paths via `ragwatch.settings`, so nothing in the codebase hardcodes directories anymore.

## Core modules

### Logging (`ragwatch`)

Core components:
- `schema.py`: defines `RetrievedDoc` and `RunRecord` Pydantic models for every RAG execution (dataset/pipeline names, QA pairs, retrieved docs, latency metrics, token usage, and arbitrary `extra` flags).
- `writers.py`: provides `JSONLWriter`, which ensures the log directory exists and appends serialized `RunRecord` entries as newline-delimited JSON.
- `logger.py`: exposes `RAGWatchLogger`, wiring schema + writer, stamping UUIDs/timestamps, and streaming runs into dataset-specific JSONL files (e.g., `logs/hotpotqa_v1.jsonl`).

#### Example usage

```python
from ragwatch import RAGWatchLogger

logger = RAGWatchLogger(
   log_dir="logs",
   dataset_name="hotpotqa",
   pipeline_name="demo",
)

logger.log_run(
   question="What is retrieval-augmented generation?",
   answer="It combines search over a corpus with LLM completion.",
   retrieved_docs=[
      {"doc_id": "doc-1", "score": 0.88, "source": "stub", "metadata": {}},
   ],
   latency_ms={"retrieval": 42.1, "generation": 210.5},
   token_usage={"prompt": 250, "completion": 120},
   session_id="notebook-001",
   extra={"escalated": False},
)

print("Run recorded in logs/hotpotqa_demo.jsonl")
```

#### Module tests

```bash
python -m unittest tests.test_schema tests.test_writers tests.test_logger
```

### HotpotQA pipeline (`ragwatch_hotpotqa`)

- **Data prep**: `load_questions()` / `build_document_corpus()` pull from the Hugging Face `hotpot_qa` dataset (with stub fallback for offline/test scenarios) and respect the env-configured split/sample sizes.
- **Retriever**: `build_retriever`, `load_retriever`, `load_vectorstore`, and `ensure_retriever` manage the FAISS index stored at `RAGWATCH_HOTPOTQA_INDEX_DIR`. They accept dependency-injected embeddings/doc sources for testing, while `ragwatch.settings` handles `.env` loading.
- **RAG chain**: `build_rag_chain()` wires the retriever output through a simple LangChain prompt and `ChatOpenAI(model="gpt-4o-mini", temperature=0.1)`. It formats retrieved docs into a context block before sending the request to the LLM.
- **Evaluation harness**: `run_eval.py` loops through QA pairs, invokes the chain, and logs outcomes via `RAGWatchLogger`, including the retrieved documents + similarity scores for each question. It also exposes `run_hotpotqa_stream` for continuous monitoring.
- **CLI**: `python -m ragwatch_hotpotqa eval` executes a single pass over the dataset, while `python -m ragwatch_hotpotqa stream --interval 1.0` keeps answering questions in a loop (use `--max-iterations` to stop automatically).

## Usage snippet (HotpotQA prototype)

```python
from ragwatch_hotpotqa.retriever import ensure_retriever
from ragwatch_hotpotqa.build_rag import build_rag_chain

# Ensure FAISS index exists (builds from stub docs for now)
ensure_retriever()

# Build the LangChain-style RAG pipeline
chain = build_rag_chain()
answer = chain.invoke("Who is Barack Obama?")
print(answer)
```

To instrument the above chain with rich telemetry, run the evaluation harness:

```bash
python -m ragwatch_hotpotqa eval
```

It ensures the FAISS index exists, invokes the RAG chain over the stub dataset, and writes JSONL logs under the directory specified by `RAGWATCH_HOTPOTQA_LOG_DIR`. Override the defaults by passing `--log-dir`, `--dataset-name`, or `--pipeline-name` flags.

For continuous monitoring, run:

```bash
python -m ragwatch_hotpotqa stream --interval 1.0 --max-iterations 10
```

This repeatedly samples questions (cycling through the stub set for now), waits the requested interval between calls, and logs each result so you can watch performance trends over time.

## Next steps
- Replace the placeholder HotpotQA loaders with real data ingestion + chunking.
- Expand logging metadata (e.g., model version, guardrail flags) and add monitoring dashboards.
- Integrate evaluation harnesses to compare runs across datasets.
