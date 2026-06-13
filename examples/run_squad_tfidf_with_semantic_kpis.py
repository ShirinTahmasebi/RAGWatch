"""Run SQuAD + TF-IDF with default KPIs plus optional semantic KPIs.

Requirements:
    pip install -e ".[datasets,semantic]"

Configure the provider via environment (defaults to local):
    RAGWATCH_SEMANTIC_PROVIDER=local | openai | azure_openai

Usage:
    python examples/run_squad_tfidf_with_semantic_kpis.py

A small number of examples is used by default to keep paid-API usage low.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.config.env import load_env
from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.report import KPIEngine
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever
from ragwatch.semantic.providers import (
    SemanticProviderConfigError,
    create_semantic_embedding_provider_from_env,
)

MAX_EXAMPLES = 5  # keep small: semantic KPIs may use paid APIs

SEMANTIC_HINT = (
    "Semantic KPIs are optional. Install dependencies with:\n"
    '  pip install -e ".[datasets,semantic]"\n'
    "Then configure .env if using OpenAI or Azure OpenAI."
)


def main() -> None:
    load_env()

    print(f"Loading SQuAD (validation, max {MAX_EXAMPLES} examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=MAX_EXAMPLES)
    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    retriever = TfidfRetriever()
    retriever.index(dataset.corpus)
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    kpi_engine = KPIEngine()

    try:
        provider = create_semantic_embedding_provider_from_env()
        semantic_engine: SemanticKPIEngine | None = SemanticKPIEngine(provider)
    except SemanticProviderConfigError as exc:
        print(f"Skipping semantic KPIs: {exc}\n")
        print(SEMANTIC_HINT)
        semantic_engine = None

    for example in dataset.qa_examples:
        run = client.run(query=example.question, top_k=3)
        report = kpi_engine.compute(run)

        print(f"Question: {run.query}")
        print(f"Answer:   {run.generation.answer}")
        print("\nDefault KPIs:")
        print("-" * 60)
        for kpi in report.results:
            print(f"  {kpi.name:40s} = {kpi.value}")

        if semantic_engine is not None:
            semantic_report = semantic_engine.compute(run)
            print("\nSemantic KPIs:")
            print("-" * 60)
            for kpi in semantic_report.results:
                print(f"  {kpi.name:40s} = {kpi.value}")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
