"""Run the toy RAG pipeline with default KPIs plus optional semantic KPIs.

Requirements (semantic KPIs are optional):
    pip install -e ".[semantic]"

Configure the provider via environment (defaults to local):
    RAGWATCH_SEMANTIC_PROVIDER=local | openai | azure_openai

Usage:
    python examples/run_toy_rag_with_semantic_kpis.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.config.env import load_env
from ragwatch.datasets.jsonl_loader import JsonlLoader
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.report import KPIEngine, merge_kpi_reports
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever
from ragwatch.semantic.providers import (
    SemanticProviderConfigError,
    create_semantic_embedding_provider_from_env,
)

SEMANTIC_HINT = (
    "Semantic KPIs are optional. Install dependencies with:\n"
    '  pip install -e ".[semantic]"\n'
    "Then configure .env if using OpenAI or Azure OpenAI."
)


def main() -> None:
    load_env()
    data_dir = Path(__file__).resolve().parent / "data"

    loader = JsonlLoader(
        corpus_path=data_dir / "toy_corpus.jsonl",
        qa_path=data_dir / "toy_qa.jsonl",
    )
    corpus = loader.load_corpus()
    qa_examples = loader.load_qa_examples()
    print(f"Loaded {len(corpus)} documents and {len(qa_examples)} QA examples.\n")

    retriever = TfidfRetriever()
    retriever.index(corpus)
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    kpi_engine = KPIEngine()

    # Semantic engine is optional; fail gracefully if config/deps are missing.
    try:
        provider = create_semantic_embedding_provider_from_env()
        semantic_engine: SemanticKPIEngine | None = SemanticKPIEngine(provider)
    except SemanticProviderConfigError as exc:
        print(f"Skipping semantic KPIs: {exc}\n")
        print(SEMANTIC_HINT)
        semantic_engine = None

    for example in qa_examples:
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
            # Reports can be merged into one combined report if desired.
            merge_kpi_reports(report, semantic_report)

        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
