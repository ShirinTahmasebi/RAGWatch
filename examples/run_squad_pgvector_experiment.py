"""Run a batch experiment on SQuAD with the pgvector retriever + DB KPI snapshot.

Requirements:
    pip install -e ".[datasets,vectordb]"
    cp .env.template .env
    docker compose up -d postgres

Usage:
    python examples/run_squad_pgvector_experiment.py

Outputs (under outputs/experiments/squad_pgvector/):
    runs.jsonl
    kpis.jsonl
    kpis.csv
    db_kpis.json
    summary.json

Fails gracefully if .env / PostgreSQL is not available.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.config.env import get_env, load_env

# Load .env before anything else
load_env()


def main() -> None:
    db_url = get_env("RAGWATCH_PGVECTOR_URL")
    if not db_url:
        print("ERROR: RAGWATCH_PGVECTOR_URL is not set.")
        print()
        print("To run this example:")
        print("  1. cp .env.template .env")
        print("  2. docker compose up -d postgres")
        print("  3. python examples/run_squad_pgvector_experiment.py")
        sys.exit(1)

    from ragwatch.clients.basic_rag_client import BasicRAGClient
    from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
    from ragwatch.embeddings.sentence_transformer import (
        SentenceTransformerEmbeddingModel,
    )
    from ragwatch.experiments.exporters import export_experiment
    from ragwatch.experiments.runner import ExperimentRunner
    from ragwatch.experiments.schema import ExperimentConfig
    from ragwatch.generators.heuristic_generator import HeuristicGenerator
    from ragwatch.metrics.report import KPIEngine
    from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

    config = ExperimentConfig(
        experiment_name="squad_pgvector_baseline",
        dataset_name="squad_validation",
        retriever_name="pgvector",
        generator_name="heuristic",
        top_k=3,
        max_examples=20,
        metadata={"collection_name": "ragwatch_squad_experiment"},
    )

    # --- Load dataset ---
    print(f"Loading SQuAD (validation, max {config.max_examples} examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=config.max_examples)

    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # --- Build pipeline ---
    embedding_model = SentenceTransformerEmbeddingModel()
    retriever = PgVectorRetriever(
        embedding_model=embedding_model,
        collection_name="ragwatch_squad_experiment",
        connection_string=db_url,
        dataset_name=config.dataset_name,
        reset_collection=True,
    )

    print("Indexing corpus into pgvector...")
    retriever.index(dataset.corpus)
    print("Indexing complete.\n")

    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    # --- Run experiment with a DB KPI snapshot ---
    runner = ExperimentRunner(client=client, kpi_engine=KPIEngine())
    print("Running experiment...")
    result = runner.run(
        qa_examples=dataset.qa_examples,
        config=config,
        compute_db_kpis=True,
    )

    # --- Export results ---
    output_dir = Path("outputs/experiments/squad_pgvector")
    export_experiment(result, output_dir)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"  Experiment:     {config.experiment_name}")
    print(f"  Num examples:   {result.num_examples}")
    print(f"  Successful:     {result.num_successful}")
    print(f"  Failed:         {result.num_failed}")
    print(f"  Output dir:     {output_dir}")
    print()

    if result.db_kpi_report is not None:
        print("DB KPIs:")
        for kpi in result.db_kpi_report.results:
            print(f"  {kpi.name:30s} = {kpi.value}")
        print()

    summary = json.loads((output_dir / "summary.json").read_text())
    aggregates = summary.get("aggregate_metrics", {})
    if aggregates:
        print("Aggregate metrics:")
        for name, value in aggregates.items():
            print(f"  {name}: {value:.4f}")
    print()

    print("Output files:")
    for name in ("runs.jsonl", "kpis.jsonl", "kpis.csv", "db_kpis.json", "summary.json"):
        print(f"  {output_dir / name}")
    print()


if __name__ == "__main__":
    main()
