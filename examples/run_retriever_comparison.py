"""Compare retriever backends on a registered QA dataset across top-k settings.

This is the dataset-agnostic research entry point. Choose the dataset with the
``RAGWATCH_DATASET`` environment variable (default ``squad``):

    RAGWATCH_DATASET=squad    python examples/run_retriever_comparison.py
    RAGWATCH_DATASET=hotpotqa python examples/run_retriever_comparison.py

Requirements:
    pip install -e ".[datasets,semantic]"            # TF-IDF + semantic KPIs
    pip install -e ".[datasets,vectordb,semantic]"   # also Chroma + Qdrant + pgvector

Dataset configuration (all optional):
    RAGWATCH_DATASET                dataset name from the registry (default squad)
    RAGWATCH_DATASET_SPLIT          split name (default from the registry)
    RAGWATCH_DATASET_MAX_EXAMPLES   number of QA examples to run (default 120)

Semantic KPIs are computed by default using the local sentence-transformers
provider. The output directory is dataset- and mode-aware so different datasets
and modes never overwrite each other:
    outputs/comparisons/squad_retrievers_semantic_local/      (default)
    outputs/comparisons/hotpotqa_retrievers_semantic_local/
    outputs/comparisons/squad_retrievers_semantic_openai/     (RAGWATCH_SEMANTIC_PROVIDER=openai)
    outputs/comparisons/squad_retrievers_base/                (opt-out)

Set RAGWATCH_SEMANTIC_PROVIDER to switch provider (local | openai | azure_openai).
Set RAGWATCH_DISABLE_SEMANTIC_KPIS=true to opt out and run deterministic-only.
Set RAGWATCH_COMPARISON_OUTPUT_DIR to override the output directory verbatim.

TF-IDF always runs. Chroma and Qdrant are included only if their optional
dependencies are installed. pgvector is included only when RAGWATCH_PGVECTOR_URL
is configured and Postgres is reachable; otherwise it is skipped gracefully.
"""

import csv
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ragwatch.clients.basic_rag_client import BasicRAGClient
from ragwatch.config.env import get_env, load_env
from ragwatch.core.schema import Document
from ragwatch.datasets.registry import (
    dataset_max_examples_from_env,
    dataset_name_from_env,
    dataset_split_from_env,
    get_dataset_entry,
    load_registered_dataset,
)
from ragwatch.experiments.comparison import (
    ComparisonConfig,
    ComparisonRunner,
    RetrieverExperimentSpec,
    export_comparison,
)
from ragwatch.experiments.output_naming import (
    comparison_dir_name,
    comparison_output_dir,
)
from ragwatch.generators.heuristic_generator import HeuristicGenerator
from ragwatch.metrics.report import DBKPIEngine
from ragwatch.metrics.semantic import SemanticKPIEngine
from ragwatch.retrievers.tfidf_retriever import TfidfRetriever
from ragwatch.semantic.providers import (
    SemanticProviderConfigError,
    create_semantic_embedding_provider_from_env,
    resolve_semantic_provider_name,
    semantic_kpis_disabled,
    semantic_kpis_enabled,
)

# Load .env so RAGWATCH_PGVECTOR_URL and dataset config are available when present.
load_env()

DEFAULT_DATASET = "squad"
DEFAULT_MAX_EXAMPLES = 120
TOP_K_VALUES = [3, 5]

# Optional override for the comparison output directory. When set, it is used
# verbatim and the dataset-/mode-specific directory naming is bypassed.
ENV_COMPARISON_OUTPUT_DIR = "RAGWATCH_COMPARISON_OUTPUT_DIR"

SEMANTIC_FAILURE_MESSAGE = (
    "Semantic KPIs are enabled by default for this experiment.\n"
    "Install semantic dependencies with:\n"
    '  pip install -e ".[semantic]"\n\n'
    "For local semantic KPIs, use:\n"
    "  RAGWATCH_SEMANTIC_PROVIDER=local\n\n"
    "For OpenAI/Azure, configure the corresponding API variables in .env.\n\n"
    "To run without semantic KPIs, opt out explicitly:\n"
    "  RAGWATCH_DISABLE_SEMANTIC_KPIS=true \\\n"
    "      python examples/run_retriever_comparison.py\n\n"
    "Underlying error: {error}"
)


def create_semantic_kpi_engine() -> SemanticKPIEngine:
    """Create a semantic KPI engine from env.

    Raises ``SemanticProviderConfigError`` if dependencies or configuration are
    missing. Callers are expected to fail clearly rather than silently dropping
    semantic KPIs for this research script.
    """
    provider = create_semantic_embedding_provider_from_env()
    return SemanticKPIEngine(provider=provider)


def resolve_comparison_output_dir(
    dataset_name: str,
    semantic_enabled: bool,
    semantic_provider_name: str | None,
) -> Path:
    """Resolve the comparison output directory for the current dataset and mode.

    A ``RAGWATCH_COMPARISON_OUTPUT_DIR`` override, when set, is used verbatim.
    Otherwise the directory is chosen from the dataset name and run mode so that
    datasets, semantic runs, and base runs never overwrite each other.
    """
    override = get_env(ENV_COMPARISON_OUTPUT_DIR)
    if override:
        print(f"Using custom comparison output directory: {override}")
        return Path(override)
    return comparison_output_dir(
        dataset_name, semantic_enabled, semantic_provider_name
    )


def try_create_pgvector_spec(
    dataset_name: str,
    corpus: list[Document],
    embedding_model: object | None,
    generator: HeuristicGenerator,
) -> RetrieverExperimentSpec | None:
    """Build a pgvector spec only when Postgres is configured and reachable.

    Returns ``None`` (and prints a clear reason) when ``RAGWATCH_PGVECTOR_URL``
    is unset, an embedding model is unavailable, or connecting/indexing fails.
    On success, the spec computes a DB KPI snapshot per setting.
    """
    db_url = get_env("RAGWATCH_PGVECTOR_URL")
    if not db_url:
        print(
            "Skipping pgvector: RAGWATCH_PGVECTOR_URL is not set or Postgres is "
            "not reachable.\n"
            "To include pgvector, run:\n"
            "  cp .env.template .env\n"
            "  docker compose up -d postgres"
        )
        return None

    if embedding_model is None:
        print("Skipping pgvector: no embedding model is available.")
        return None

    try:
        from ragwatch.retrievers.vector.pgvector_retriever import PgVectorRetriever

        pgvector = PgVectorRetriever(
            embedding_model=embedding_model,
            collection_name=f"{dataset_name}_comparison_pgvector",
            connection_string=db_url,
            dataset_name=dataset_name,
            reset_collection=True,
        )
        pgvector.index(corpus)
    except Exception as exc:  # noqa: BLE001
        print(
            "Skipping pgvector: RAGWATCH_PGVECTOR_URL is not set or Postgres is "
            f"not reachable ({exc}).\n"
            "To include pgvector, run:\n"
            "  cp .env.template .env\n"
            "  docker compose up -d postgres"
        )
        return None

    return RetrieverExperimentSpec(
        name="pgvector",
        client=BasicRAGClient(retriever=pgvector, generator=generator),
        retriever_name="pgvector",
        generator_name="heuristic",
        db_kpi_engine=DBKPIEngine(),
        compute_db_kpis=True,
    )


def _build_specs(
    dataset_name: str, corpus: list[Document], generator: HeuristicGenerator
) -> list[RetrieverExperimentSpec]:
    """Build retriever specs, including optional backends when available."""
    specs: list[RetrieverExperimentSpec] = []
    collection_name = f"ragwatch_{dataset_name}_comparison"

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
                collection_name=collection_name,
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
                collection_name=collection_name,
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

    # --- Optional pgvector backend (only when Postgres is reachable) ---
    pgvector_spec = try_create_pgvector_spec(
        dataset_name, corpus, embedding_model, generator
    )
    if pgvector_spec is not None:
        specs.append(pgvector_spec)

    return specs


def main(default_dataset: str = DEFAULT_DATASET) -> None:
    # --- Resolve dataset configuration from env ---
    dataset_name = dataset_name_from_env(default_dataset)
    entry = get_dataset_entry(dataset_name)
    split = dataset_split_from_env() or entry.default_split
    max_examples = dataset_max_examples_from_env(DEFAULT_MAX_EXAMPLES)

    print(
        f"Loading dataset '{dataset_name}' (split={split}, "
        f"max {max_examples} examples)..."
    )
    dataset = load_registered_dataset(
        dataset_name, split=split, max_examples=max_examples
    )
    print(f"Corpus: {len(dataset.corpus)} documents")
    print(f"QA examples: {len(dataset.qa_examples)} questions\n")

    # --- Build retriever specs over the same corpus ---
    generator = HeuristicGenerator()
    specs = _build_specs(dataset_name, dataset.corpus, generator)
    print(f"Retrievers in comparison: {[s.name for s in specs]}\n")

    # --- Semantic KPIs are computed by default (opt out explicitly) ---
    semantic_enabled = semantic_kpis_enabled()
    provider_name: str | None = None
    if semantic_enabled:
        try:
            semantic_engine = create_semantic_kpi_engine()
        except SemanticProviderConfigError as exc:
            print(SEMANTIC_FAILURE_MESSAGE.format(error=exc))
            raise SystemExit(1)
        provider_name = resolve_semantic_provider_name()
        print(f"Semantic KPIs enabled (provider: {provider_name}).")
        for spec in specs:
            spec.semantic_kpi_engine = semantic_engine
            spec.compute_semantic_kpis = True
    else:
        print("Semantic KPIs disabled via RAGWATCH_DISABLE_SEMANTIC_KPIS.")

    # --- Make the run dataset- and mode-aware (name + output directory) ---
    comparison_name = comparison_dir_name(
        dataset_name, semantic_enabled, provider_name
    )
    output_dir = resolve_comparison_output_dir(
        dataset_name, semantic_enabled, provider_name
    )

    config = ComparisonConfig(
        comparison_name=comparison_name,
        dataset_name=f"{dataset_name}_{split}",
        top_k_values=list(TOP_K_VALUES),
        max_examples=len(dataset.qa_examples),
    )

    # --- Run comparison ---
    runner = ComparisonRunner()
    print("Running comparison...")
    result = runner.run(
        qa_examples=dataset.qa_examples, specs=specs, config=config
    )

    # --- Export ---
    export_comparison(result, output_dir)

    # --- Summary ---
    combined_path = output_dir / "combined_kpis.csv"
    with combined_path.open() as f:
        num_rows = max(sum(1 for _ in csv.reader(f)) - 1, 0)  # minus header

    semantic_mode = (
        f"semantic ({provider_name})" if semantic_enabled else "disabled"
    )

    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    print(f"  Dataset:          {dataset_name}")
    print(f"  Split:            {split}")
    print(f"  Corpus docs:      {len(dataset.corpus)}")
    print(f"  QA examples:      {len(dataset.qa_examples)}")
    print(f"  Semantic KPIs:    {semantic_mode}")
    print(f"  Comparison:       {config.comparison_name}")
    print(f"  Experiment keys:  {result.experiment_keys}")
    print(f"  Num settings:     {result.num_experiments}")
    print(f"  combined_kpis rows: {num_rows}")
    print(f"  Output dir:       {output_dir}")
    print(f"  Summary CSV:      {output_dir / 'comparison_summary.csv'}")
    print()


if __name__ == "__main__":
    main()
