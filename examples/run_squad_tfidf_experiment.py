"""Run a batch experiment on SQuAD with the TF-IDF retriever.

Requirements:
    pip install -e ".[datasets]"

Usage:
    python examples/run_squad_tfidf_experiment.py

Outputs (under outputs/experiments/squad_tfidf/):
    runs.jsonl
    kpis.jsonl
    kpis.csv
    db_kpis.json
    summary.json
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
from ragwatch.experiments.exporters import export_experiment
from ragwatch.experiments.runner import ExperimentRunner
from ragwatch.experiments.schema import ExperimentConfig
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.report import KPIEngine
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever


def main() -> None:
    # --- Configuration ---
    config = ExperimentConfig(
        experiment_name="squad_tfidf_baseline",
        dataset_name="squad_validation",
        retriever_name="tfidf",
        generator_name="heuristic",
        top_k=3,
        max_examples=20,
    )

    # --- Load dataset ---
    print(f"Loading SQuAD (validation, max {config.max_examples} examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=config.max_examples)

    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # --- Build pipeline ---
    retriever = TfidfRetriever()
    retriever.index(dataset.corpus)
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    # --- Run experiment ---
    runner = ExperimentRunner(client=client, kpi_engine=KPIEngine())
    print("Running experiment...")
    result = runner.run(qa_examples=dataset.qa_examples, config=config)

    # --- Export results ---
    output_dir = Path("outputs/experiments/squad_tfidf")
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
