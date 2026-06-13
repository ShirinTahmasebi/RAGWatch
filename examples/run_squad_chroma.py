"""Run RAG pipeline on SQuAD using ChromaDB retriever.

Requirements:
    pip install ragwatch[datasets,vectordb]

Usage:
    python examples/run_squad_chroma.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
from ragwatch.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.retrievers.vector.chroma_retriever import ChromaRetriever


def main() -> None:
    print("Loading SQuAD (validation, max 5 examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=5)

    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # Build embedding model and retriever
    print("Loading embedding model...")
    embedding_model = SentenceTransformerEmbeddingModel()

    retriever = ChromaRetriever(
        embedding_model=embedding_model,
        collection_name="ragwatch_squad_demo",
    )

    print("Indexing documents...")
    retriever.index(dataset.corpus)

    # Create RAG client
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)

    # Run on first 5 questions
    print()
    for example in dataset.qa_examples[:5]:
        result = client.run(query=example.question, top_k=3)

        print(f"Question: {result.query}")
        print(f"Expected: {example.answers}")
        print(f"Answer:   {result.generation.answer[:100]}...")
        print(f"Latency:  {result.latency_ms:.2f} ms")
        print(f"Retriever: {result.metadata['retriever']}")
        print("Retrieved documents:")
        for rd in result.retrieved_documents:
            print(f"  [{rd.rank}] {rd.document.doc_id} (score={rd.score:.4f})")
        print()


if __name__ == "__main__":
    main()
