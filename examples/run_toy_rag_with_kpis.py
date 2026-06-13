"""Run the toy RAG pipeline and compute deterministic KPIs.

Usage:
    python examples/run_toy_rag_with_kpis.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.datasets.jsonl_loader import JsonlLoader
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.report import KPIEngine, to_flat_dict
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever


def main() -> None:
    data_dir = Path(__file__).resolve().parent / "data"

    # Load data
    loader = JsonlLoader(
        corpus_path=data_dir / "toy_corpus.jsonl",
        qa_path=data_dir / "toy_qa.jsonl",
    )
    corpus = loader.load_corpus()
    qa_examples = loader.load_qa_examples()

    print(f"Loaded {len(corpus)} documents and {len(qa_examples)} QA examples.\n")

    # Build pipeline
    retriever = TfidfRetriever()
    retriever.index(corpus)
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    # KPI engine with default metrics
    engine = KPIEngine()

    for example in qa_examples:
        result = client.run(query=example.question, top_k=3)
        report = engine.compute(result)

        print(f"Question: {result.query}")
        print(f"Answer:   {result.generation.answer}")
        print(f"Docs:     {[rd.document.doc_id for rd in result.retrieved_documents]}")
        print()
        print("KPI Results:")
        print("-" * 60)
        for kpi in report.results:
            print(f"  {kpi.name:40s} = {kpi.value}")
        print()

        # Also show flat dict format
        flat = to_flat_dict(report)
        print("Flat dict (for CSV):")
        print(json.dumps(flat, indent=2, default=str))
        print("=" * 60)
        print()


if __name__ == "__main__":
    main()
