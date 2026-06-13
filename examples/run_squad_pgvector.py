"""Run RAG pipeline on SQuAD using pgvector retriever with persistent PostgreSQL.

Requirements:
    pip install -e ".[datasets,vectordb]"
    docker compose up -d postgres
    cp .env.template .env

Usage:
    python examples/run_squad_pgvector.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.config.env import get_env, load_env

# Load .env before anything else
load_env()


def main() -> None:
    # Check for database URL early with a helpful message
    db_url = get_env("RAGWATCH_PGVECTOR_URL")
    if not db_url:
        print("ERROR: RAGWATCH_PGVECTOR_URL is not set.")
        print()
        print("To run this example:")
        print("  1. cp .env.template .env")
        print("  2. docker compose up -d postgres")
        print("  3. python examples/run_squad_pgvector.py")
        print()
        print("The default URL is: postgresql://ragwatch:ragwatch@localhost:5432/ragwatch")
        sys.exit(1)

    from ragwatch.clients.basic_rag_client import BasicRAGClient
    from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
    from ragwatch.embeddings.sentence_transformer import SentenceTransformerEmbeddingModel
    from ragwatch.generators.heuristic_generator import HeuristicGenerator
    from ragwatch.metrics.report import KPIEngine
    from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

    # Load SQuAD
    print("Loading SQuAD (validation, max 20 examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=20)

    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # Create embedding model and pgvector retriever
    embedding_model = SentenceTransformerEmbeddingModel()
    retriever = PgVectorRetriever(
        embedding_model=embedding_model,
        collection_name="ragwatch_squad_demo",
        connection_string=db_url,
        dataset_name="squad_validation",
        reset_collection=True,
    )

    # Index corpus
    print("Indexing corpus into pgvector...")
    retriever.index(dataset.corpus)
    print("Indexing complete.\n")

    # Build RAG client
    generator = HeuristicGenerator()
    client = BasicRAGClient(retriever=retriever, generator=generator)
    engine = KPIEngine()

    # Run on first 5 examples
    for example in dataset.qa_examples[:5]:
        result = client.run(query=example.question, top_k=3)
        report = engine.compute(result)

        print(f"Question: {result.query}")
        print(f"Expected: {example.answers}")
        print(f"Answer:   {result.generation.answer[:100]}")
        print(f"Latency:  {result.latency_ms:.3f} ms")
        print(f"Collection: {retriever._collection_name}")
        print("Retrieved documents:")
        for rd in result.retrieved_documents:
            print(f"  [{rd.rank}] {rd.document.doc_id} (score={rd.score:.4f})")
        print("KPIs:")
        for kpi in report.results[:5]:
            print(f"  {kpi.name}: {kpi.value}")
        print()


if __name__ == "__main__":
    main()
