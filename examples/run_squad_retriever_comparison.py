"""Compare retriever backends on SQuAD across multiple top-k settings.

Requirements:
    pip install -e ".[datasets]"            # TF-IDF only
    pip install -e ".[datasets,vectordb]"   # also Chroma + Qdrant

Usage:
    python examples/run_squad_retriever_comparison.py

Outputs (under outputs/comparisons/squad_retrievers/):
    combined_kpis.csv
    comparison_summary.csv
    comparison_summary.json
    experiments/<retriever>_top<k>/...

TF-IDF always runs. Chroma and Qdrant are included only if their optional
dependencies are installed. pgvector is intentionally skipped here (TODO:
add it gracefully only when RAGWATCH_PGVECTOR_URL is available).
"""

import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.core.schema import Document
from ragwatch.datasets.adapters.squad_adapter import SquadAdapter
from ragwatch.experiments.comparison import (
    ComparisonConfig,
    ComparisonRunner,
    RetrieverExperimentSpec,
    export_comparison,
)
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever


def _build_specs(
    corpus: list[Document], generator: HeuristicGenerator
) -> list[RetrieverExperimentSpec]:
    """Build retriever specs, including optional backends when available."""
    specs: list[RetrieverExperimentSpec] = []

    # --- TF-IDF (always available) ---
    tfidf = TfidfRetriever()
    tfidf.index(corpus)
    specs.append(
        RetrieverExperimentSpec(
            name="tfidf",
            client=BasicRAGClient(retriever=tfidf, generator=generator),
            retriever_name="tfidf",
            generator_name="heuristic",
        )
    )

    # --- Optional vector backends (Chroma, Qdrant) ---
    embedding_model = None
    try:
        from ragwatch.embeddings.sentence_transformer import (
            SentenceTransformerEmbeddingModel,
        )

        embedding_model = SentenceTransformerEmbeddingModel()
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping vector backends (no embedding model): {exc}")

    if embedding_model is not None:
        try:
            from ragwatch.retrievers.vector.chroma_retriever import ChromaRetriever

            chroma = ChromaRetriever(
                embedding_model=embedding_model,
                collection_name="ragwatch_squad_comparison",
            )
            chroma.index(corpus)
            specs.append(
                RetrieverExperimentSpec(
                    name="chroma",
                    client=BasicRAGClient(retriever=chroma, generator=generator),
                    retriever_name="chroma",
                    generator_name="heuristic",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping Chroma: {exc}")

        try:
            from ragwatch.retrievers.vector.qdrant_retriever import QdrantRetriever

            qdrant = QdrantRetriever(
                embedding_model=embedding_model,
                collection_name="ragwatch_squad_comparison",
            )
            qdrant.index(corpus)
            specs.append(
                RetrieverExperimentSpec(
                    name="qdrant",
                    client=BasicRAGClient(retriever=qdrant, generator=generator),
                    retriever_name="qdrant",
                    generator_name="heuristic",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping Qdrant: {exc}")

    return specs


def main() -> None:
    config = ComparisonConfig(
        comparison_name="squad_retrievers",
        dataset_name="squad_validation",
        top_k_values=[3, 5],
        max_examples=20,
    )

    # --- Load dataset ---
    print(f"Loading SQuAD (validation, max {config.max_examples} examples)...")
    adapter = SquadAdapter()
    dataset = adapter.load(split="validation", max_examples=config.max_examples)
    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # --- Build retriever specs over the same corpus ---
    generator = HeuristicGenerator()
    specs = _build_specs(dataset.corpus, generator)
    print(f"Retrievers in comparison: {[s.name for s in specs]}\n")

    # --- Run comparison ---
    runner = ComparisonRunner()
    print("Running comparison...")
    result = runner.run(
        qa_examples=dataset.qa_examples, specs=specs, config=config
    )

    # --- Export ---
    output_dir = Path("outputs/comparisons/squad_retrievers")
    export_comparison(result, output_dir)

    # --- Summary ---
    combined_path = output_dir / "combined_kpis.csv"
    with combined_path.open() as f:
        num_rows = max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus header

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"  Comparison:       {config.comparison_name}")
    print(f"  Experiment keys:  {result.experiment_keys}")
    print(f"  Num settings:     {result.num_experiments}")
    print(f"  combined_kpis rows: {num_rows}")
    print(f"  Summary CSV:      {output_dir / 'comparison_summary.csv'}")
    print()


if __name__ == "__main__":
    main()
