"""Run the toy RAG pipeline on example data."""

import sys
from pathlib import Path

# Allow running from the project root without installing the package
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.datasets.jsonl_loader import JsonlLoader
from ragwatch.generators.heuristic_generator import HeuristicGenerator
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

    # Create RAG client
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    # Run on each QA example
    for example in qa_examples:
        result = client.run(query=example.question, top_k=3)

        print(f"Question: {result.query}")
        print(f"Answer:   {result.generation.answer}")
        print(f"Latency:  {result.latency_ms:.2f} ms")
        print("Retrieved documents:")
        for rd in result.retrieved_documents:
            print(f"  [{rd.rank}] {rd.document.doc_id} (score={rd.score:.4f})")
        print()


if __name__ == "__main__":
    main()
