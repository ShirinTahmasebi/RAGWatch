"""Run the toy RAG pipeline with OpenTelemetry console tracing.

Requirements:
    pip install -e ".[otel]"

Usage:
    python examples/run_toy_rag_with_tracing.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.datasets.jsonl_loader import JsonlLoader
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.instrumentation.tracer import RAGWatchTracer
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

    # Build retriever and index
    retriever = TfidfRetriever()
    retriever.index(corpus)

    # Create RAG client with tracing
    generator = HeuristicGenerator()
    tracer = RAGWatchTracer(service_name="ragwatch-toy-demo", exporter_type="console")
    client = BasicRAGClient(retriever=retriever, generator=generator, tracer=tracer)

    # Run on first 2 QA examples to keep output readable
    print("=" * 60)
    print("Running RAG pipeline with OpenTelemetry tracing enabled")
    print("Spans will be printed to console below")
    print("=" * 60)
    print()

    for example in qa_examples[:2]:
        result = client.run(query=example.question, top_k=3)

        print(f"\nQuestion: {result.query}")
        print(f"Answer:   {result.generation.answer}")
        print(f"Latency:  {result.latency_ms:.2f} ms")
        print(f"Traced:   {result.metadata.get('traced', False)}")
        print("Retrieved documents:")
        for rd in result.retrieved_documents:
            print(f"  [{rd.rank}] {rd.document.doc_id} (score={rd.score:.4f})")
        print()

    tracer.shutdown()


if __name__ == "__main__":
    main()
